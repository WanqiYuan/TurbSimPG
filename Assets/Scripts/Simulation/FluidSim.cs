using System;
using System.Collections;
using UnityEngine;

using Seb.GPUSorting;
using Unity.Mathematics;
using System.Collections.Generic;
using static Seb.Helpers.ComputeHelper;
using UnityEditor;
using UnityEngine.Rendering;
using Unity.Collections;
namespace Seb.Fluid.Simulation
{

    public enum SimulationRegionShape
    {
        Box = 0,
        SquarePyramid = 1
    }

    [System.Serializable]
    public class HeatSource
    {
        [Tooltip("When disabled, this source is uploaded at the ambient/base temperature instead of its configured temperature.")]
        public bool enabled = true;
        public Transform sourceCube;
        public float temperature = 100.0f;  // Heat source temperature
        public float influence = 1.0f;  // influence multiplier
    }

    public class FluidSim : MonoBehaviour
    {
        public event Action<FluidSim> SimulationInitCompleted;

        [Header("Time Step")] public float normalTimeScale = 1;
        public float slowTimeScale = 0.1f;
        public float maxTimestepFPS = 60; // if time-step dips lower than this fps, simulation will run slower (set to 0 to disable)
        public int iterationsPerFrame = 3;

        [Header("Simulation Settings")] public float gravity = -10;
        public float smoothingRadius = 0.2f;
        public float targetDensity = 630;
        public float sissmMu = 0.9f;
        public int sissmMaxIterations = 50;
        [Range(1e-6f, 1e-2f)]
        public float sissmEta = 1e-3f;
        [Range(0, 1)] public float collisionDamping = 0.95f;

        [Header("Simulation Region")]
        [Tooltip("Box uses the full Transform volume. Square Pyramid uses the Transform as an enclosing box, with its square base at local Y=-0.5 and apex at local Y=+0.5.")]
        public SimulationRegionShape simulationRegionShape = SimulationRegionShape.Box;

        public float[] partialSumsCPU;

        [Header("Thermal Turbulence Settings")]
        public HeatSource[] heatSources;
        public float baseTemperature = 20.0f;
        public float temperatureDiffusion = 1.0f;  //A multiplier for particle-particle heat diffusion
        public float thermalExpansion = 2.0f;  // A multiplier for buoyancy effect
        public float heatTransferRate = 1.0f;  //A multiplier for heat transfer between particles and heat sources
        public float tempGradientForce = 1.0f;  // A multiplier for temperature gradient force(convection)


        [Header("Neighbor List Settings")]
        public int Kmax = 32; // Maximum neighbors per particle

        [Header("Temperature Bucket Grid")]
        [Min(32)]
        [Tooltip("Maximum memory, in MiB, used by the dense temperature bucket-entry buffer. " +
                 "The bucket cell size is increased automatically when needed.")]
        public int maxBucketBufferSizeMB = 256;

        [Header("Volumetric Render Settings")] public bool renderToTex3D;
        public int densityTextureRes;

        [Header("References")] public ComputeShader compute;
        public Spawner3D spawner;

        [HideInInspector] public RenderTexture DensityMap;
        [HideInInspector] public RenderTexture TemperatureMap;
        public Vector3 Scale => transform.localScale;

        // Buffers

        public ComputeBuffer positionBuffer { get; private set; }
        public ComputeBuffer velocityBuffer { get; private set; }
        public ComputeBuffer densityBuffer { get; private set; }
        public ComputeBuffer predictedPositionsBuffer;
        public ComputeBuffer spatialKeys { get; private set; }
        public ComputeBuffer spatialOffsets { get; private set; }
        public ComputeBuffer sortedIndices { get; private set; }
        public ComputeBuffer debugBuffer { get; private set; }
        public ComputeBuffer temperatureBuffer { get; private set; }

        public ComputeBuffer heatSourcesBuffer { get; private set; }
        public ComputeBuffer neighborListBuffer { get; private set; }
        public ComputeBuffer neighborCountBuffer { get; private set; }

        public ComputeBuffer predictedPositionsStarBuffer { get; private set; }

        public ComputeBuffer predictedPositionsNextBuffer { get; private set; }
        public ComputeBuffer psiPerParticleBuffer { get; private set; }
        public ComputeBuffer psiPartialSumsBuffer { get; private set; }



        ComputeBuffer sortTarget_positionBuffer;
        ComputeBuffer sortTarget_velocityBuffer;
        ComputeBuffer sortTarget_predictedPositionsBuffer;
        ComputeBuffer sortTarget_predictedPositionsStarBuffer;
        ComputeBuffer sortTarget_TemperatureBuffer;

        // Bucket grid for temperature pre-pass
        ComputeBuffer bucketCellCountsBuffer;
        ComputeBuffer bucketCellEntriesBuffer;
        const int BucketMaxPerCell = 64;
        Vector3Int bucketGridRes;
        float bucketGridCellSize;

        // Kernel IDs
        const int externalForcesKernel = 0;
        const int spatialHashKernel = 1;
        const int reorderKernel = 2;
        const int reorderCopybackKernel = 3;
        const int densityKernel = 4;
        const int updatePositionsKernel = 5;
        const int renderKernel = 6;
        const int updateTemperatureKernel = 7;
        const int sissmUpdatePredictedPositionsKernel = 8;
        const int reducePsiKernel = 9;
        const int copyPredictedPositionsKernel = 10;
        const int reorderPredictedPositionsStarKernel = 11;
        const int reorderCopyBackPredictedPositionsStarKernel = 12;
        const int buildNeighbourListKernel = 13;
        const int clearBucketGridKernel = 14;
        const int buildBucketGridKernel = 15;

        // Sorting
        GPUCountSort gpuSort;
        SpatialOffsetCalculator spatialOffsetsCalc;

        // State
        private bool isPaused;
        public bool IsPaused { get { return isPaused; } set { isPaused = value; } }
        public float CurrentFrameSimulationDeltaTime
        {
            get
            {
                if (isPaused)
                {
                    return 0f;
                }

                float maxDeltaTime = maxTimestepFPS > 0 ? 1f / maxTimestepFPS : float.PositiveInfinity;
                return Mathf.Min(Time.deltaTime * ActiveTimeScale, maxDeltaTime);
            }
        }
        bool pauseNextFrame;
        float smoothRadiusOld;
        float simTimer;
        public float SimTime => simTimer;
        bool inSlowMode;
        Spawner3D.SpawnData spawnData;
        Dictionary<ComputeBuffer, string> bufferNameLookup;

        const int ReduceThreadGroupSize = 256;

        // Structure to match the compute shader
        struct HeatSourceData
        {
            public Matrix4x4 localToWorld;
            public Matrix4x4 worldToLocal;
            public float temperature;
            public float influence;
            public Vector3 padding;

            public static int Size()
            {
                return sizeof(float) * 32 + sizeof(float) * 2 + sizeof(float) * 3;
            }
        }

        void Start()
        {
            Debug.Log("Controls: Space = Play/Pause, Q = SlowMode, R = Reset");
            isPaused = false;

            Initialize();
        }

        void Initialize()
        {
            spawnData = spawner.GetSpawnData(transform, simulationRegionShape);

            if (spawnData.points.Length == 0)
            {
                Debug.LogError("FluidSim: The selected spawn regions contain no particles inside the simulation region.", this);
                return;
            }


            // Create buffers
            int numParticles = spawnData.points.Length;
            positionBuffer = CreateStructuredBuffer<float3>(numParticles);
            predictedPositionsBuffer = CreateStructuredBuffer<float3>(numParticles);
            predictedPositionsStarBuffer = CreateStructuredBuffer<float3>(numParticles);
            predictedPositionsNextBuffer = CreateStructuredBuffer<float3>(numParticles);
            psiPerParticleBuffer = CreateStructuredBuffer<float>(numParticles);
            psiPartialSumsBuffer = CreateStructuredBuffer<float>((numParticles + ReduceThreadGroupSize - 1) / ReduceThreadGroupSize);
            velocityBuffer = CreateStructuredBuffer<float3>(numParticles);
            densityBuffer = CreateStructuredBuffer<float2>(numParticles);
            spatialKeys = CreateStructuredBuffer<uint>(numParticles);
            spatialOffsets = CreateStructuredBuffer<uint>(numParticles);
            sortedIndices = CreateStructuredBuffer<uint>(numParticles);
            debugBuffer = CreateStructuredBuffer<float3>(numParticles);

            temperatureBuffer = CreateStructuredBuffer<float>(numParticles);
            heatSourcesBuffer = CreateStructuredBuffer<HeatSourceData>(heatSources.Length);

            neighborListBuffer = CreateStructuredBuffer<int>(numParticles * Kmax);
            neighborCountBuffer = CreateStructuredBuffer<int>(numParticles);

            sortTarget_positionBuffer = CreateStructuredBuffer<float3>(numParticles);
            sortTarget_predictedPositionsBuffer = CreateStructuredBuffer<float3>(numParticles);
            sortTarget_predictedPositionsStarBuffer = CreateStructuredBuffer<float3>(numParticles);
            sortTarget_TemperatureBuffer = CreateStructuredBuffer<float>(numParticles);
            sortTarget_velocityBuffer = CreateStructuredBuffer<float3>(numParticles);

            bufferNameLookup = new Dictionary<ComputeBuffer, string>
            {
                { positionBuffer, "Positions" },
                { predictedPositionsBuffer, "PredictedPositions" },
                { predictedPositionsStarBuffer,"PredictedPositionsStar"},
                { predictedPositionsNextBuffer,"PredictedPositionsNext"},
                { psiPerParticleBuffer, "PsiPerParticle" },
                { psiPartialSumsBuffer, "PsiPartialSums" },
                { velocityBuffer, "Velocities" },
                { densityBuffer, "Densities" },
                { spatialKeys, "SpatialKeys" },
                { spatialOffsets, "SpatialOffsets" },
                { sortedIndices, "SortedIndices" },
                { sortTarget_positionBuffer, "SortTarget_Positions" },
                { sortTarget_predictedPositionsBuffer, "SortTarget_PredictedPositions" },
                { sortTarget_predictedPositionsStarBuffer, "SortTarget_PredictedPositionsStar" },
                { sortTarget_TemperatureBuffer, "SortTarget_Temperature" },
                { sortTarget_velocityBuffer, "SortTarget_Velocities" },
                { debugBuffer, "Debug" },
                { temperatureBuffer,"Temperature" },
                { heatSourcesBuffer,"HeatSources" },
                { neighborListBuffer, "NeighborList" },
                { neighborCountBuffer, "NeighborCount" }
            };
            partialSumsCPU = new float[(numParticles + ReduceThreadGroupSize - 1) / ReduceThreadGroupSize];
            // Set buffer data
            SetInitialBufferData(spawnData);

            // External forces kernel
            SetBuffers(compute, externalForcesKernel, bufferNameLookup, new ComputeBuffer[]
            {
                spatialKeys,
                spatialOffsets,
                positionBuffer,
                predictedPositionsBuffer,
                predictedPositionsStarBuffer,
                velocityBuffer,
                temperatureBuffer
            });

            // Set up temperature update kernel
            SetBuffers(compute, updateTemperatureKernel, bufferNameLookup, new ComputeBuffer[]
            {
                velocityBuffer,
                positionBuffer,
                temperatureBuffer,
                heatSourcesBuffer,
                neighborListBuffer,
                neighborCountBuffer
            });

            // Bucket grid kernels
            SetBuffers(compute, clearBucketGridKernel, bufferNameLookup, new ComputeBuffer[]
            {
            });
            SetBuffers(compute, buildBucketGridKernel, bufferNameLookup, new ComputeBuffer[]
            {
                positionBuffer
            });

            // Spatial hash kernel
            SetBuffers(compute, spatialHashKernel, bufferNameLookup, new ComputeBuffer[]
            {
                spatialKeys,
                spatialOffsets,
                predictedPositionsBuffer,
                sortedIndices
            });

            // Reorder kernel
            SetBuffers(compute, reorderKernel, bufferNameLookup, new ComputeBuffer[]
            {
                positionBuffer,
                sortTarget_positionBuffer,
                predictedPositionsBuffer,
                sortTarget_predictedPositionsBuffer,
                velocityBuffer,
                sortTarget_velocityBuffer,
                sortedIndices
            });

            // Reorder copyback kernel
            SetBuffers(compute, reorderCopybackKernel, bufferNameLookup, new ComputeBuffer[]
            {
                positionBuffer,
                sortTarget_positionBuffer,
                predictedPositionsBuffer,
                sortTarget_predictedPositionsBuffer,
                velocityBuffer,
                sortTarget_velocityBuffer,
                sortedIndices
            });

            // Density kernel
            SetBuffers(compute, densityKernel, bufferNameLookup, new ComputeBuffer[]
            {
                predictedPositionsBuffer,
                densityBuffer,
                neighborListBuffer,
                neighborCountBuffer
            });

            //Update SISSM predicted positions kernel
            SetBuffers(compute, sissmUpdatePredictedPositionsKernel, bufferNameLookup, new ComputeBuffer[]
            {
                predictedPositionsBuffer,
                predictedPositionsStarBuffer,
                predictedPositionsNextBuffer,
                psiPerParticleBuffer,
                densityBuffer,
                neighborListBuffer,
                neighborCountBuffer
            });

            //SISSM psi reduce kernel
            SetBuffers(compute, reducePsiKernel, bufferNameLookup, new ComputeBuffer[]
            {
                psiPerParticleBuffer,
                psiPartialSumsBuffer
            });

            //Copy predicted positions kernel
            SetBuffers(compute, copyPredictedPositionsKernel, bufferNameLookup, new ComputeBuffer[]
            {
                predictedPositionsBuffer,
                predictedPositionsNextBuffer
            });

            //Reorder PredictedPositionsStar kernel  
            SetBuffers(compute, reorderPredictedPositionsStarKernel, bufferNameLookup, new ComputeBuffer[]
            {
                predictedPositionsStarBuffer,
                sortTarget_predictedPositionsStarBuffer,
                temperatureBuffer,
                sortTarget_TemperatureBuffer,
                sortedIndices
            });

            //Reorder copyback PredictedPositionsStar kernel
            SetBuffers(compute, reorderCopyBackPredictedPositionsStarKernel, bufferNameLookup, new ComputeBuffer[]
            {
                predictedPositionsStarBuffer,
                sortTarget_TemperatureBuffer,
                temperatureBuffer,
                sortTarget_predictedPositionsStarBuffer
            });

            // Update positions kernel
            SetBuffers(compute, updatePositionsKernel, bufferNameLookup, new ComputeBuffer[]
            {
                predictedPositionsBuffer,
                positionBuffer,
                velocityBuffer
            });

            // Render to 3d tex kernel
            SetBuffers(compute, renderKernel, bufferNameLookup, new ComputeBuffer[]
            {
                predictedPositionsBuffer,
                densityBuffer,
                spatialKeys,
                spatialOffsets,
                temperatureBuffer,
            });


            // Build Neighbour List kernel
            SetBuffers(compute, buildNeighbourListKernel, bufferNameLookup, new ComputeBuffer[]
            {
                predictedPositionsBuffer,
                spatialKeys,
                spatialOffsets,
                neighborListBuffer,
                neighborCountBuffer
            });

            compute.SetInt("numParticles", positionBuffer.count);
            compute.SetInt("Kmax", Kmax);

            gpuSort = new GPUCountSort(spatialKeys, sortedIndices, (uint)(spatialKeys.count - 1));
            spatialOffsetsCalc = new SpatialOffsetCalculator(spatialKeys, spatialOffsets);

            UpdateBucketGridSettings();
            UpdateSmoothingConstants();

            // Run single frame of sim with deltaTime = 0 to initialize density texture
            // (so that display can work even if paused at start)
            if (renderToTex3D)
            {
                RunSimulationFrame(0);
            }

            SimulationInitCompleted?.Invoke(this);
        }

        void Update()
        {
            if (heatSources != null && heatSources.Length > 0)
            {
                UpdateHeatSourcesBuffer();
            }

            // Run simulation
            if (!isPaused)
            {
                RunSimulationFrame(CurrentFrameSimulationDeltaTime);
            }

            if (pauseNextFrame)
            {
                isPaused = true;
                pauseNextFrame = false;
            }

            HandleInput();
        }

        void UpdateBucketGridSettings()
        {
            Vector3 origin = transform.position - transform.localScale * 0.5f;
            Vector3 size = transform.localScale;
            float requestedCellSize = Mathf.Max(smoothingRadius, 0.0001f);
            float cellSize = requestedCellSize;

            const long UnityMaxComputeBufferBytes = 2147483648L;
            long requestedBufferBytes = (long)Mathf.Max(32, maxBucketBufferSizeMB) * 1024L * 1024L;
            long maxEntryBufferBytes = Math.Min(requestedBufferBytes, UnityMaxComputeBufferBytes);
            long maxCellCount = Math.Max(1L, maxEntryBufferBytes / (sizeof(uint) * BucketMaxPerCell));

            Vector3Int newRes = CalculateBucketGridResolution(size, cellSize);
            long newCellCount = CalculateBucketCellCount(newRes);

            // A cell size >= smoothingRadius still guarantees that all particles within the
            // smoothing radius are in the current cell or one of its 26 adjacent cells.
            // Increase the cell size only as much as needed to fit the configured memory budget.
            for (int i = 0; newCellCount > maxCellCount && i < 16; i++)
            {
                double scale = Math.Cbrt((double)newCellCount / maxCellCount);
                cellSize *= Mathf.Max(1.001f, (float)scale * 1.001f);
                newRes = CalculateBucketGridResolution(size, cellSize);
                newCellCount = CalculateBucketCellCount(newRes);
            }

            if (newCellCount > maxCellCount)
            {
                throw new InvalidOperationException(
                    $"Unable to fit the temperature bucket grid inside {maxBucketBufferSizeMB} MiB. " +
                    $"Resolution={newRes}, cells={newCellCount:N0}.");
            }

            if (bucketCellCountsBuffer == null ||
                bucketGridRes != newRes ||
                !Mathf.Approximately(bucketGridCellSize, cellSize))
            {
                bucketCellCountsBuffer?.Release();
                bucketCellEntriesBuffer?.Release();

                bucketGridRes = newRes;
                bucketGridCellSize = cellSize;
                int cellCount = checked((int)newCellCount);
                int entryCount = checked(cellCount * BucketMaxPerCell);
                bucketCellCountsBuffer = new ComputeBuffer(cellCount, sizeof(uint));
                bucketCellEntriesBuffer = new ComputeBuffer(entryCount, sizeof(uint));

                if (cellSize > requestedCellSize * 1.001f)
                {
                    float entryBufferMiB = entryCount * sizeof(uint) / (1024f * 1024f);
                    Debug.LogWarning(
                        $"FluidSim: temperature bucket grid was limited to {entryBufferMiB:F1} MiB. " +
                        $"Cell size increased from {requestedCellSize:F5} to {cellSize:F5}; " +
                        $"resolution is {bucketGridRes.x}x{bucketGridRes.y}x{bucketGridRes.z}.",
                        this);
                }

                bufferNameLookup[bucketCellCountsBuffer] = "BucketCellCounts";
                bufferNameLookup[bucketCellEntriesBuffer] = "BucketCellEntries";

                compute.SetBuffer(clearBucketGridKernel, "BucketCellCounts", bucketCellCountsBuffer);
                compute.SetBuffer(buildBucketGridKernel, "BucketCellCounts", bucketCellCountsBuffer);
                compute.SetBuffer(buildBucketGridKernel, "BucketCellEntries", bucketCellEntriesBuffer);
                compute.SetBuffer(buildBucketGridKernel, "Positions", positionBuffer);
                compute.SetBuffer(updateTemperatureKernel, "BucketCellCounts", bucketCellCountsBuffer);
                compute.SetBuffer(updateTemperatureKernel, "BucketCellEntries", bucketCellEntriesBuffer);
            }

            compute.SetInts("bucketGridRes", bucketGridRes.x, bucketGridRes.y, bucketGridRes.z);
            compute.SetVector("bucketGridOrigin", origin);
            compute.SetFloat("bucketGridCellSize", cellSize);
        }

        static Vector3Int CalculateBucketGridResolution(Vector3 size, float cellSize)
        {
            return new Vector3Int(
                Mathf.Max(1, Mathf.CeilToInt(Mathf.Abs(size.x) / cellSize)),
                Mathf.Max(1, Mathf.CeilToInt(Mathf.Abs(size.y) / cellSize)),
                Mathf.Max(1, Mathf.CeilToInt(Mathf.Abs(size.z) / cellSize))
            );
        }

        static long CalculateBucketCellCount(Vector3Int resolution)
        {
            return (long)resolution.x * resolution.y * resolution.z;
        }


        private void UpdateHeatSourcesBuffer()
        {
            if (heatSourcesBuffer == null || heatSources == null) return;

            HeatSourceData[] heatSourcesData = new HeatSourceData[Mathf.Max(1, heatSources.Length)];

            if (heatSources.Length > 0)
            {
                for (int i = 0; i < heatSources.Length; i++)
                {
                    if (heatSources[i].sourceCube != null)
                    {
                        heatSourcesData[i] = new HeatSourceData
                        {
                            localToWorld = heatSources[i].sourceCube.localToWorldMatrix,
                            worldToLocal = heatSources[i].sourceCube.worldToLocalMatrix,
                            temperature = heatSources[i].enabled ? heatSources[i].temperature : baseTemperature,
                            influence = heatSources[i].influence
                        };
                    }
                }
            }
            else
            {
                heatSourcesData[0] = new HeatSourceData
                {
                    temperature = baseTemperature,
                    influence = 0
                };
            }

            heatSourcesBuffer.SetData(heatSourcesData);
            compute.SetInt("heatSourceCount", heatSources?.Length ?? 0);
        }


        void RunSimulationFrame(float frameDeltaTime)
        {
            float subStepDeltaTime = frameDeltaTime / iterationsPerFrame;
            UpdateSettings(subStepDeltaTime, frameDeltaTime);

            // Simulation sub-steps
            for (int i = 0; i < iterationsPerFrame; i++)
            {
                simTimer += subStepDeltaTime;
                RunSimulationStep();
            }

            if (renderToTex3D)
            {
                UpdateDensityMap();
            }
        }

        void UpdateDensityMap()
        {
            float maxAxis = Mathf.Max(transform.localScale.x, transform.localScale.y, transform.localScale.z);
            int w = Mathf.RoundToInt(transform.localScale.x / maxAxis * densityTextureRes);
            int h = Mathf.RoundToInt(transform.localScale.y / maxAxis * densityTextureRes);
            int d = Mathf.RoundToInt(transform.localScale.z / maxAxis * densityTextureRes);
            CreateRenderTexture3D(ref DensityMap, w, h, d, UnityEngine.Experimental.Rendering.GraphicsFormat.R16_SFloat, TextureWrapMode.Clamp);
            //Debug.Log(w + " " + h + "  " + d);
            compute.SetTexture(renderKernel, "DensityMap", DensityMap);
            CreateRenderTexture3D(ref TemperatureMap, w, h, d, UnityEngine.Experimental.Rendering.GraphicsFormat.R32_SFloat, TextureWrapMode.Clamp);
            compute.SetTexture(renderKernel, "TemperatureMap", TemperatureMap);
            compute.SetInts("densityMapSize", DensityMap.width, DensityMap.height, DensityMap.volumeDepth);
            Dispatch(compute, DensityMap.width, DensityMap.height, DensityMap.volumeDepth, renderKernel);
        }

        [ContextMenu("Refresh Density Map At Current State")]
        public void RefreshDensityMapAtCurrentState()
        {
            if (!Application.isPlaying || positionBuffer == null)
            {
                Debug.LogError("FluidSim: Density map can only be refreshed after the simulation has initialized in Play Mode.", this);
                return;
            }

            if (!renderToTex3D)
            {
                Debug.LogError("FluidSim: Enable renderToTex3D before refreshing the density map.", this);
                return;
            }

            UpdateDensityMap();
            Debug.Log(
                $"FluidSim: Refreshed DensityMap at the current simulation state " +
                $"({DensityMap.width}x{DensityMap.height}x{DensityMap.volumeDepth}).",
                this);
        }

        void RunSimulationStep()
        {
            int N = positionBuffer.count;
            int reduceGroups = (N + ReduceThreadGroupSize - 1) / ReduceThreadGroupSize;

            if (psiPartialSumsBuffer == null || psiPartialSumsBuffer.count != reduceGroups)
            {
                psiPartialSumsBuffer?.Release();
                psiPartialSumsBuffer = new ComputeBuffer(reduceGroups, sizeof(float));
                // Update buffer name lookup for the new buffer
                if (bufferNameLookup.ContainsKey(psiPartialSumsBuffer))
                    bufferNameLookup[psiPartialSumsBuffer] = "PsiPartialSums";
                else
                    bufferNameLookup.Add(psiPartialSumsBuffer, "PsiPartialSums");
            }
            if (partialSumsCPU == null || partialSumsCPU.Length != reduceGroups)
                partialSumsCPU = new float[reduceGroups];

            UpdateBucketGridSettings();
            int totalCells = bucketGridRes.x * bucketGridRes.y * bucketGridRes.z;
            int clearGroups = (totalCells + ReduceThreadGroupSize - 1) / ReduceThreadGroupSize;
            compute.Dispatch(clearBucketGridKernel, clearGroups, 1, 1);
            Dispatch(compute, N, kernelIndex: buildBucketGridKernel);

            Dispatch(compute, N, kernelIndex: updateTemperatureKernel);
            Dispatch(compute, N, kernelIndex: externalForcesKernel);

            // Build spatial hash with predicted positions
            Dispatch(compute, N, kernelIndex: spatialHashKernel);
            gpuSort.Run();
            spatialOffsetsCalc.Run(false);

            // Align buffer and particle data
            Dispatch(compute, N, kernelIndex: reorderKernel);
            Dispatch(compute, N, kernelIndex: reorderPredictedPositionsStarKernel);
            Dispatch(compute, N, kernelIndex: reorderCopybackKernel);
            Dispatch(compute, N, kernelIndex: reorderCopyBackPredictedPositionsStarKernel);

            // Build neighbor list for each particle after reordering
            Dispatch(compute, N, kernelIndex: buildNeighbourListKernel);

            // error threshold
            float psi0 = -1f, psiRef = float.PositiveInfinity, psiPrev = float.PositiveInfinity;
            const float tiny = 1e-12f;

            // SISSM sub iterations
            for (int k = 0; k < sissmMaxIterations; ++k)
            {
                    //Update densities with neighbor list
                    Dispatch(compute, N, kernelIndex: densityKernel);

                    // SISSM
                    Dispatch(compute, N, kernelIndex: sissmUpdatePredictedPositionsKernel);

                    //Compute Psi
                    compute.SetInt("reduceCount", N);
                    compute.Dispatch(reducePsiKernel, reduceGroups, 1, 1);

                    psiPartialSumsBuffer.GetData(partialSumsCPU);
                    float psiK = 0f; for (int g = 0; g < reduceGroups; g++) psiK += partialSumsCPU[g];

                    if (k == 0) { psi0 = psiK; psiRef = psiK; }

                    bool stop = false;
                    if (k >= 1)
                    {
                        float denom = Mathf.Max(psi0 - psiRef, tiny);
                        float epsK = (psiK - psiRef) / denom;
                        float relDrop = Mathf.Abs(psiPrev - psiK) / Mathf.Max(Mathf.Abs(psiPrev), tiny);
                        stop = (epsK < sissmEta) || (relDrop < sissmEta * 0.1f);
                    }
                    psiRef = Mathf.Min(psiRef, psiK);
                    psiPrev = psiK;

                    // 6) Copy the iteration results back to PredictedPositions
                    Dispatch(compute, N, kernelIndex: copyPredictedPositionsKernel);

                if (stop) break;
            }

            // Update position
            Dispatch(compute, N, kernelIndex: updatePositionsKernel);

        }

        void UpdateSmoothingConstants()
        {
            float r = smoothingRadius;
            float spikyPow2 = 15 / (2 * Mathf.PI * Mathf.Pow(r, 5));
            float spikyPow3 = 15 / (Mathf.PI * Mathf.Pow(r, 6));
            float spikyPow2Grad = 15 / (Mathf.PI * Mathf.Pow(r, 5));
            float spikyPow3Grad = 45 / (Mathf.PI * Mathf.Pow(r, 6));

            compute.SetFloat("K_SpikyPow2", spikyPow2);
            compute.SetFloat("K_SpikyPow3", spikyPow3);
            compute.SetFloat("K_SpikyPow2Grad", spikyPow2Grad);
            compute.SetFloat("K_SpikyPow3Grad", spikyPow3Grad);
        }



        void UpdateSettings(float stepDeltaTime, float frameDeltaTime)
        {
            if (smoothingRadius != smoothRadiusOld)
            {
                smoothRadiusOld = smoothingRadius;
                UpdateSmoothingConstants();
            }

            Vector3 simBoundsSize = transform.localScale;
            Vector3 simBoundsCentre = transform.position;

            compute.SetFloat("deltaTime", stepDeltaTime);
            compute.SetFloat("sissmMu", sissmMu);
            compute.SetFloat("whiteParticleDeltaTime", frameDeltaTime);
            compute.SetFloat("simTime", simTimer);
            compute.SetFloat("gravity", gravity);
            compute.SetFloat("collisionDamping", collisionDamping);
            compute.SetFloat("smoothingRadius", smoothingRadius);
            compute.SetFloat("targetDensity", targetDensity);
            compute.SetVector("boundsSize", simBoundsSize);
            compute.SetVector("centre", simBoundsCentre);
            compute.SetInt("simulationRegionShape", (int)simulationRegionShape);

            compute.SetMatrix("localToWorld", transform.localToWorldMatrix);
            compute.SetMatrix("worldToLocal", transform.worldToLocalMatrix);

            // Thermal distortion settings
            compute.SetFloat("temperatureDiffusion", temperatureDiffusion);
            compute.SetFloat("thermalExpansion", thermalExpansion);
            compute.SetFloat("baseTemperature", baseTemperature);
            compute.SetFloat("heatTransferRate", heatTransferRate);
            compute.SetFloat("tempGradientForce", tempGradientForce);

            // Keep the clamp ceiling independent of the on/off state. Otherwise
            // changing a source to baseTemperature would erase all existing heat
            // in a single step instead of allowing it to dissipate naturally.
            float maximumTemperature = baseTemperature;
            if (heatSources != null)
            {
                for (int i = 0; i < heatSources.Length; i++)
                {
                    if (heatSources[i] != null)
                    {
                        maximumTemperature = Mathf.Max(maximumTemperature, heatSources[i].temperature);
                    }
                }
            }
            compute.SetFloat("maximumTemperature", maximumTemperature);

            if (heatSources != null && heatSources.Length > 0)
            {
                compute.SetInt("heatSourceCount", heatSources.Length);
                UpdateHeatSourcesBuffer();
            }
            else
            {
                compute.SetInt("heatSourceCount", 0);
            }

        }

        void SetInitialBufferData(Spawner3D.SpawnData spawnData)
        {
            positionBuffer.SetData(spawnData.points);
            predictedPositionsBuffer.SetData(spawnData.points);
            predictedPositionsStarBuffer.SetData(spawnData.points);
            velocityBuffer.SetData(spawnData.velocities);


            // Initialize temperatures to base temperature
            float[] initialTemps = new float[spawnData.points.Length];
            for (int i = 0; i < spawnData.points.Length; i++)
            {
                initialTemps[i] = baseTemperature;
            }
            temperatureBuffer.SetData(initialTemps);

            debugBuffer.SetData(new float3[debugBuffer.count]);
            simTimer = 0;
        }

        void HandleInput()
        {
            if (Input.GetKeyDown(KeyCode.Space))
            {
                isPaused = !isPaused;
            }

            if (Input.GetKeyDown(KeyCode.RightArrow))
            {
                isPaused = false;
                pauseNextFrame = true;
            }

            if (Input.GetKeyDown(KeyCode.R))
            {
                pauseNextFrame = true;
                SetInitialBufferData(spawnData);
                // Run single frame of sim with deltaTime = 0 to initialize density texture
                // (so that display can work even if paused at start)
                if (renderToTex3D)
                {
                    RunSimulationFrame(0);
                }
            }

            if (Input.GetKeyDown(KeyCode.Q))
            {
                inSlowMode = !inSlowMode;
            }
        }

        private float ActiveTimeScale => inSlowMode ? slowTimeScale : normalTimeScale;

        void OnDestroy()
        {
            foreach (var kvp in bufferNameLookup)
            {
                Release(kvp.Key);
            }

            gpuSort.Release();
        }


        public struct ParticleData
        {
            public Vector3[] positions;
            public Vector3[] velocities;
            public float[] temperatures;
        }

        public ParticleData GetParticleData()
        {
            if (positionBuffer == null || velocityBuffer == null || temperatureBuffer == null)
            {
                Debug.LogError("Buffers are null, cannot get particle data.");
                return new ParticleData();
            }

            int count = positionBuffer.count;
            ParticleData data = new ParticleData
            {
                positions = new Vector3[count],
                velocities = new Vector3[count],
                temperatures = new float[count]
            };

            positionBuffer.GetData(data.positions);
            velocityBuffer.GetData(data.velocities);
            temperatureBuffer.GetData(data.temperatures);

            return data;
        }



        void OnDrawGizmos()
        {
            // Draw Simulation Bounds
            var m = Gizmos.matrix;
            Gizmos.matrix = transform.localToWorldMatrix;
            Gizmos.color = new Color(0, 1, 0, 0.5f);
            if (simulationRegionShape == SimulationRegionShape.SquarePyramid)
            {
                DrawWireSquarePyramid();
            }
            else
            {
                Gizmos.DrawWireCube(Vector3.zero, Vector3.one);
            }

            // Draw Heat Sources
            if (heatSources != null)
            {
                foreach (var hs in heatSources)
                {
                    if (hs != null && hs.sourceCube != null)
                    {
                        Gizmos.matrix = hs.sourceCube.localToWorldMatrix;
                        Gizmos.color = new Color(1.0f, 0.2f, 0.0f, 0.5f); // Semi-transparent red/orange
                        Gizmos.DrawWireCube(Vector3.zero, Vector3.one);

                        // Optional: draw a solid cube with lower opacity for better visibility
                        Gizmos.color = new Color(1.0f, 0.2f, 0.0f, 0.1f);
                        Gizmos.DrawCube(Vector3.zero, Vector3.one);
                    }
                }
            }

            Gizmos.matrix = m;
        }

        private static void DrawWireSquarePyramid()
        {
            Vector3 bottom0 = new Vector3(-0.5f, -0.5f, -0.5f);
            Vector3 bottom1 = new Vector3(0.5f, -0.5f, -0.5f);
            Vector3 bottom2 = new Vector3(0.5f, -0.5f, 0.5f);
            Vector3 bottom3 = new Vector3(-0.5f, -0.5f, 0.5f);
            Vector3 apex = new Vector3(0f, 0.5f, 0f);

            Gizmos.DrawLine(bottom0, bottom1);
            Gizmos.DrawLine(bottom1, bottom2);
            Gizmos.DrawLine(bottom2, bottom3);
            Gizmos.DrawLine(bottom3, bottom0);
            Gizmos.DrawLine(bottom0, apex);
            Gizmos.DrawLine(bottom1, apex);
            Gizmos.DrawLine(bottom2, apex);
            Gizmos.DrawLine(bottom3, apex);
        }
    }
}
