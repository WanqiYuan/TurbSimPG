Shader "Fluid/URPRaymarching"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
    }
    SubShader
    {
        Tags { "RenderType" = "Opaque" "RenderPipeline" = "UniversalPipeline" }
        Cull Off ZWrite Off ZTest Always

        Pass
        {
            Name "URP Raymarching"

            HLSLPROGRAM
            #pragma vertex Vert
            #pragma fragment Frag
            #pragma target 5.0

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"
            #include "Packages/com.unity.render-pipelines.core/Runtime/Utilities/Blit.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/DeclareDepthTexture.hlsl"

            struct Ray
            {
                float3 position;
                float3 direction;
                float3 energy;
            };

            struct RayHit
            {
                float3 position;
                float  distance;
                float3 normal;
                float3 color;
                float3 rayDirection;
                float2 screenUV;
                bool   didHit;
            };

            struct DebugHeader
            {
                int valid;
                int pixelX;
                int pixelY;
                int stepCount;
                float3 rayOrigin;
                float3 rayDir;
                float dstToBox;
                float3 entryPos;
                float3 entryDir;
            };

            struct DebugStep
            {
                int stepIndex;
                int eventType;
                float accumulatedDistance;
                float stepSize;
                float3 worldPos;
                float3 worldDir;
                float3 uvw;
                float sampledDensity;
                float sampledIor;
                float3 grad;
            };

            struct DebugPixelSelectionData
            {
                int pixelX;
                int pixelY;
            };

            RayHit InitializeRayHit()
            {
                RayHit hit;
                hit.position = float3(0, 0, 0);
                hit.distance = 0;
                hit.normal = float3(0, 0, 0);
                hit.color = float3(0, 0, 0);
                hit.rayDirection = float3(0, 0, 0);
                hit.screenUV = float2(0, 0);
                hit.didHit = false;
                return hit;
            }

            TEXTURE3D(DensityMap);
            SamplerState linearClampSampler;

            float indexOfRefraction;
            float3 extinctionCoeff;
            float3 boundsSize;
            float volumeValueOffset;
            float densityMultiplier;
            float viewMarchStepSize;
            float lightStepSize;
            float K;
            int _FilterHighDensityDerivedIoR;
            float _DensityDerivedIoRUpperThreshold;
            float3 _VolumeResolution;
            int _UseAdaptiveStepSize;
            float3 _VolumePosition;

            int _UseIoRVolume;
            float3 _IORTextureBounds;
            int _SimulationRegionShape;

            int _EnableDebugPathTrace;
            int _DebugPixelCount;
            int _DebugMaxSteps;
            int _DebugStepCapacity;
            int _DebugFullScreenBatch;
            int _DebugBatchStartPixel;
            int _DebugTargetWidth;
            StructuredBuffer<DebugPixelSelectionData> _DebugPixelBuffer;
            RWStructuredBuffer<DebugHeader> _DebugHeaderBuffer : register(u2);
            RWStructuredBuffer<DebugStep> _DebugStepBuffer : register(u3);

            static const float TinyNudge = 0.01;
            static const float DebugEdgeEpsilon = 0.0001;

            float3 dirToSun;

            float3 ComputeNDCWithZ(float3 positionWS, float4x4 worldToClip)
            {
                float4 posCS = mul(worldToClip, float4(positionWS, 1.0));
                posCS.xyz /= posCS.w;
                posCS.xy = posCS.xy * 0.5 + 0.5;
                #if UNITY_UV_STARTS_AT_TOP
                    posCS.y = 1.0 - posCS.y;
                #endif
                return posCS.xyz;
            }

            static const float SS_MARCHING_THICKNESS = 0.5;
            static const int SS_MAX_STEPS = 1000;

            float2 RayBoxDst(float3 boundsMin, float3 boundsMax, float3 rayOrigin, float3 rayDir)
            {
                float3 invRayDir = 1.0 / rayDir;
                float3 t0 = (boundsMin - rayOrigin) * invRayDir;
                float3 t1 = (boundsMax - rayOrigin) * invRayDir;
                float3 tmin = min(t0, t1);
                float3 tmax = max(t0, t1);
                float dstA = max(max(tmin.x, tmin.y), tmin.z);
                float dstB = min(tmax.x, min(tmax.y, tmax.z));
                float dstToBox = max(0, dstA);
                float dstInsideBox = max(0, dstB - dstToBox);
                return float2(dstToBox, dstInsideBox);
            }

            float3 ComputeActiveUVW(float3 pos)
            {
                float3 localPos = pos - _VolumePosition;
                return (localPos + boundsSize * 0.5) / boundsSize;
            }

            float3 ComputeSampleUVW(float3 pos)
            {
                float3 localPos = pos - _VolumePosition;
                float3 sampleBounds = (_UseIoRVolume > 0) ? _IORTextureBounds : boundsSize;
                return (localPos + sampleBounds * 0.5) / sampleBounds;
            }

            bool IsInsideActiveVolume(float3 pos)
            {
                float3 uvw = ComputeActiveUVW(pos);
                if (!all(uvw > DebugEdgeEpsilon) || !all(uvw < 1.0 - DebugEdgeEpsilon))
                {
                    return false;
                }

                if (_SimulationRegionShape == 1)
                {
                    float3 localNormalized = uvw - 0.5;
                    float halfExtent = 0.5 * (0.5 - localNormalized.y);
                    return abs(localNormalized.x) < halfExtent &&
                           abs(localNormalized.z) < halfExtent;
                }

                return true;
            }

            float SampleDensity(float3 pos)
            {
                float3 uvw = ComputeActiveUVW(pos);
                bool isEdge = any(uvw >= 1 - DebugEdgeEpsilon || uvw <= DebugEdgeEpsilon);
                if (isEdge)
                {
                    return -volumeValueOffset;
                }

                if (_SimulationRegionShape == 1)
                {
                    float3 localNormalized = uvw - 0.5;
                    float halfExtent = 0.5 * (0.5 - localNormalized.y);
                    if (abs(localNormalized.x) > halfExtent ||
                        abs(localNormalized.z) > halfExtent)
                    {
                        return -volumeValueOffset;
                    }
                }

                return DensityMap.SampleLevel(linearClampSampler, uvw, 0).r - volumeValueOffset;
            }

            float SampleIoR(float3 pos)
            {
                float3 uvw = ComputeSampleUVW(pos);
                if (any(uvw >= 1 - DebugEdgeEpsilon || uvw <= DebugEdgeEpsilon))
                {
                    return 1.0 - volumeValueOffset;
                }

                return DensityMap.SampleLevel(linearClampSampler, uvw, 0).r - volumeValueOffset;
            }

            float DensityToIoR(float density)
            {
                float ior = 1.0 + K * max(0, density);
                return (_FilterHighDensityDerivedIoR > 0 && ior > _DensityDerivedIoRUpperThreshold)
                    ? 1.0003
                    : ior;
            }

            float3 Transmittance(float thickness)
            {
                return exp(-thickness * extinctionCoeff);
            }

            float CalculateDensityAlongRay(float3 rayPos, float3 rayDir, float stepSizeParam)
            {
                if (dot(rayDir, rayDir) < 0.9)
                {
                    return 0;
                }

                float3 boxMin = _VolumePosition - boundsSize * 0.5;
                float3 boxMax = _VolumePosition + boundsSize * 0.5;
                float2 boundsDstInfo = RayBoxDst(boxMin, boxMax, rayPos, rayDir);
                float dstToBounds = boundsDstInfo.x;
                float dstInsideBounds = boundsDstInfo.y;
                if (dstInsideBounds <= 0)
                {
                    return 0;
                }

                float dstTravelled = 0;
                float opticalDepth = 0;
                float nudge = stepSizeParam * 0.5;
                float3 entryPoint = rayPos + rayDir * (dstToBounds + nudge);
                dstInsideBounds -= (nudge + TinyNudge);

                while (dstTravelled < dstInsideBounds)
                {
                    float3 samplePos = entryPoint + rayDir * dstTravelled;
                    float density;
                    if (_UseIoRVolume > 0)
                    {
                        float ior = max(0, SampleIoR(samplePos));
                        density = (K > 0) ? (ior / K) : 0;
                    }
                    else
                    {
                        density = SampleDensity(samplePos);
                    }

                    density *= densityMultiplier * stepSizeParam;
                    if (density > 0)
                    {
                        opticalDepth += density;
                    }

                    dstTravelled += stepSizeParam;
                }

                return opticalDepth;
            }

            float CalculateDensityAlongRayLight(float3 rayPos, float3 rayDir)
            {
                return CalculateDensityAlongRay(rayPos, rayDir, lightStepSize);
            }

            int FindDebugTraceIndex(int2 pixelCoord)
            {
                if (_EnableDebugPathTrace <= 0)
                {
                    return -1;
                }

                if (_DebugFullScreenBatch > 0 && _DebugTargetWidth > 0)
                {
                    int linearPixelIndex = pixelCoord.y * _DebugTargetWidth + pixelCoord.x;
                    int batchIndex = linearPixelIndex - _DebugBatchStartPixel;
                    return batchIndex >= 0 && batchIndex < _DebugPixelCount ? batchIndex : -1;
                }

                [loop]
                for (int i = 0; i < _DebugPixelCount; i++)
                {
                    DebugPixelSelectionData selection = _DebugPixelBuffer[i];
                    if (all(pixelCoord == int2(selection.pixelX, selection.pixelY)))
                    {
                        return i;
                    }
                }

                return -1;
            }

            void DebugStoreHeader(int traceIndex, int2 pixelCoord, int validState, float3 rayOrigin, float3 rayDir, float dstToBox, float3 entryPos)
            {
                if (_EnableDebugPathTrace <= 0 || traceIndex < 0 || traceIndex >= _DebugPixelCount)
                {
                    return;
                }

                DebugHeader header;
                header.valid = validState;
                header.pixelX = pixelCoord.x;
                header.pixelY = pixelCoord.y;
                header.stepCount = 0;
                header.rayOrigin = rayOrigin;
                header.rayDir = rayDir;
                header.dstToBox = dstToBox;
                header.entryPos = entryPos;
                header.entryDir = normalize(rayDir);
                _DebugHeaderBuffer[traceIndex] = header;
            }

            void DebugUpdateStepCount(int traceIndex, int stepCount)
            {
                if (traceIndex < 0 || traceIndex >= _DebugPixelCount)
                {
                    return;
                }

                DebugHeader header = _DebugHeaderBuffer[traceIndex];
                header.stepCount = stepCount;
                _DebugHeaderBuffer[traceIndex] = header;
            }

            void DebugStoreStep(
                int traceIndex,
                inout int debugStepCount,
                int eventType,
                float accumulatedDistance,
                float stepSize,
                float3 worldPos,
                float3 worldDir,
                float3 uvw,
                float sampledDensity,
                float sampledIor,
                float3 grad)
            {
                if (_EnableDebugPathTrace <= 0 || traceIndex < 0 || traceIndex >= _DebugPixelCount || debugStepCount >= _DebugStepCapacity)
                {
                    return;
                }

                DebugStep step;
                step.stepIndex = debugStepCount;
                step.eventType = eventType;
                step.accumulatedDistance = accumulatedDistance;
                step.stepSize = stepSize;
                step.worldPos = worldPos;
                step.worldDir = worldDir;
                step.uvw = uvw;
                step.sampledDensity = sampledDensity;
                step.sampledIor = sampledIor;
                step.grad = grad;
                _DebugStepBuffer[traceIndex * _DebugStepCapacity + debugStepCount] = step;

                debugStepCount++;
                DebugUpdateStepCount(traceIndex, debugStepCount);
            }

            void DebugComputeCurrentMedium(
                float3 currentPos,
                float3 rayDir,
                out float currentDensity,
                out float currentIoR,
                out float3 gradIoR)
            {
                float3 oneVoxel = boundsSize / float3(
                    _VolumeResolution.x - 1,
                    _VolumeResolution.y - 1,
                    _VolumeResolution.z - 1);
                float3 gradientSampleDistance = oneVoxel;
                float3 offsetX = float3(gradientSampleDistance.x, 0, 0);
                float3 offsetY = float3(0, gradientSampleDistance.y, 0);
                float3 offsetZ = float3(0, 0, gradientSampleDistance.z);

                if (_UseIoRVolume > 0)
                {
                    currentIoR = 1.0 + max(0, SampleIoR(currentPos));

                    float iorL = 1.0 + max(0, SampleIoR(currentPos - offsetX));
                    float iorR = 1.0 + max(0, SampleIoR(currentPos + offsetX));
                    float iorD = 1.0 + max(0, SampleIoR(currentPos - offsetY));
                    float iorU = 1.0 + max(0, SampleIoR(currentPos + offsetY));
                    float iorB = 1.0 + max(0, SampleIoR(currentPos - offsetZ));
                    float iorF = 1.0 + max(0, SampleIoR(currentPos + offsetZ));

                    gradIoR = float3(
                        (iorR - iorL) / (2 * offsetX.x),
                        (iorU - iorD) / (2 * offsetY.y),
                        (iorF - iorB) / (2 * offsetZ.z));

                    currentDensity = (K > 0) ? max(0, (currentIoR - 1.0) / K) : 0;
                }
                else
                {
                    float density = max(0, SampleDensity(currentPos));
                    currentIoR = DensityToIoR(density);

                    float densL = max(0, SampleDensity(currentPos - offsetX));
                    float densR = max(0, SampleDensity(currentPos + offsetX));
                    float densD = max(0, SampleDensity(currentPos - offsetY));
                    float densU = max(0, SampleDensity(currentPos + offsetY));
                    float densB = max(0, SampleDensity(currentPos - offsetZ));
                    float densF = max(0, SampleDensity(currentPos + offsetZ));

                    float iorL = DensityToIoR(densL);
                    float iorR = DensityToIoR(densR);
                    float iorD = DensityToIoR(densD);
                    float iorU = DensityToIoR(densU);
                    float iorB = DensityToIoR(densB);
                    float iorF = DensityToIoR(densF);
                    gradIoR = float3(
                        (iorR - iorL) / (2 * offsetX.x),
                        (iorU - iorD) / (2 * offsetY.y),
                        (iorF - iorB) / (2 * offsetZ.z));
                    currentDensity = density;
                }

            }

            RayHit ScreenSpaceCurvedRayMarching(
                float3 rayOrigin,
                float3 initialRayDir,
                float entryDist,
                float maxDist,
                float stepSizeParam,
                int debugTraceIndex,
                bool debugActive,
                float3 entryPos)
            {
                RayHit result = InitializeRayHit();

                float3 currentPos = rayOrigin;
                float3 rayDir = initialRayDir;
                float traveled = 0;
                float opticalDepth = 0;
                float segment = min(stepSizeParam, maxDist);

                float lastDepthDiff = 0.0;
                float3 lastRayPositionWS = currentPos;
                bool startBinarySearch = false;
                float currStepSize = segment;
                float4x4 worldToClip = GetWorldToHClipMatrix();

                int debugStepCount = 0;
                bool debugExitLogged = false;
                bool debugMaxStepEventLogged = false;
                float debugEntryOffset = distance(rayOrigin, entryPos);

                if (debugActive)
                {
                    float entryDensity;
                    float entryIoR;
                    float3 entryGradIoR;
                    DebugComputeCurrentMedium(rayOrigin, initialRayDir, entryDensity, entryIoR, entryGradIoR);
                    DebugStoreStep(
                        debugTraceIndex,
                        debugStepCount,
                        1,
                        0.0,
                        0.0,
                        entryPos,
                        normalize(initialRayDir),
                        ComputeSampleUVW(entryPos),
                        entryDensity,
                        entryIoR,
                        entryGradIoR);
                }

                int renderStepCount = 0;

                [loop]
                for (int i = 0; i < SS_MAX_STEPS && traveled < maxDist; i++)
                {
                    renderStepCount = i + 1;

                    float currentDensity;
                    float currentIoR;
                    float3 gradIoR;
                    DebugComputeCurrentMedium(currentPos, rayDir, currentDensity, currentIoR, gradIoR);

                    if (_UseAdaptiveStepSize > 0 && !startBinarySearch)
                    {
                        float theta_threshold = 0.008;
                        float epi = 0.000001;
                        float curvature_mag = length(gradIoR - dot(gradIoR, rayDir) * rayDir);
                        float h = (currentIoR * theta_threshold) / (curvature_mag + epi);
                        if (h < 0.001)
                        {
                            segment = 0.001;
                        }
                        else if (h > 0.09)
                        {
                            segment = 0.09;
                        }
                        else
                        {
                            segment = h;
                        }

                        segment = min(segment, maxDist - traveled);
                        currStepSize = segment;
                    }
                    else if (!startBinarySearch)
                    {
                        segment = min(stepSizeParam, maxDist - traveled);
                        currStepSize = segment;
                    }

                    if (debugActive && !debugMaxStepEventLogged)
                    {
                        if (i < _DebugMaxSteps)
                        {
                            DebugStoreStep(
                                debugTraceIndex,
                                debugStepCount,
                                0,
                                traveled + debugEntryOffset,
                                currStepSize,
                                currentPos,
                                rayDir,
                                ComputeSampleUVW(currentPos),
                                currentDensity,
                                currentIoR,
                                gradIoR);
                        }
                        else
                        {
                            DebugStoreStep(
                                debugTraceIndex,
                                debugStepCount,
                                4,
                                traveled + debugEntryOffset,
                                0.0,
                                currentPos,
                                rayDir,
                                ComputeSampleUVW(currentPos),
                                currentDensity,
                                currentIoR,
                                gradIoR);
                            debugMaxStepEventLogged = true;
                        }
                    }

                    
                    //Original Unity update:
                    float3 nextPos = currentPos + rayDir * currStepSize;
                    float nextDensity = max(0, SampleDensity(nextPos));
                    float nextIoR = (_UseIoRVolume > 0)
                        ? (1.0 + max(0, SampleIoR(nextPos)))
                        : DensityToIoR(nextDensity);
                    float3 newDir = normalize((currentIoR * rayDir + gradIoR * currStepSize) / nextIoR);

                    lastRayPositionWS = currentPos;
                    rayDir = newDir;
                    currentPos += rayDir * currStepSize;
                    traveled += currStepSize;
                    

                    if (debugActive && !debugExitLogged && !IsInsideActiveVolume(currentPos))
                    {
                        float exitDensity;
                        float exitIoR;
                        float3 exitGradIoR;
                        DebugComputeCurrentMedium(currentPos, rayDir, exitDensity, exitIoR, exitGradIoR);
                        DebugStoreStep(
                            debugTraceIndex,
                            debugStepCount,
                            2,
                            traveled + debugEntryOffset,
                            currStepSize,
                            currentPos,
                            rayDir,
                            ComputeSampleUVW(currentPos),
                            exitDensity,
                            exitIoR,
                            exitGradIoR);
                        debugExitLogged = true;
                    }

                    float3 rayPositionNDC = ComputeNDCWithZ(currentPos, worldToClip);

                    #if (UNITY_REVERSED_Z == 0)
                        rayPositionNDC.z = rayPositionNDC.z * 0.5 + 0.5;
                    #endif

                    bool isScreenSpace = rayPositionNDC.x > 0.0 && rayPositionNDC.y > 0.0
                                      && rayPositionNDC.x < 1.0 && rayPositionNDC.y < 1.0;
                    if (!isScreenSpace)
                    {
                        if (debugActive)
                        {
                            float boundaryDensity;
                            float boundaryIoR;
                            float3 boundaryGradIoR;
                            DebugComputeCurrentMedium(currentPos, rayDir, boundaryDensity, boundaryIoR, boundaryGradIoR);
                            DebugStoreStep(
                                debugTraceIndex,
                                debugStepCount,
                                3,
                                traveled + debugEntryOffset,
                                currStepSize,
                                currentPos,
                                rayDir,
                                ComputeSampleUVW(currentPos),
                                boundaryDensity,
                                boundaryIoR,
                                boundaryGradIoR);
                        }

                        break;
                    }

                    float deviceDepth = SampleSceneDepth(rayPositionNDC.xy);
                    float sceneDepth = LinearEyeDepth(deviceDepth, _ZBufferParams);
                    float hitDepth = LinearEyeDepth(rayPositionNDC.z, _ZBufferParams);
                    float depthDiff = sceneDepth - hitDepth;
                    bool isSky = (deviceDepth == UNITY_RAW_FAR_CLIP_VALUE);
                    float Sign = (depthDiff >= 0) ? 1.0 : -1.0;

                    if (!startBinarySearch && Sign < 0 && !isSky)
                    {
                        startBinarySearch = true;
                    }

                    if (startBinarySearch)
                    {
                        currStepSize *= 0.5;
                        float currSign = (currStepSize >= 0) ? 1.0 : -1.0;
                        currStepSize = (currSign == Sign) ? currStepSize : -currStepSize;
                    }

                    bool hitSuccessful = (depthDiff <= 0.0) && (depthDiff >= -SS_MARCHING_THICKNESS) && !isSky;

                    if (hitSuccessful)
                    {
                        float3 hitPos = currentPos;
                        float2 hitUV = rayPositionNDC.xy;

                        if (Sign != ((lastDepthDiff >= 0) ? 1.0 : -1.0))
                        {
                            float t = lastDepthDiff / (lastDepthDiff - depthDiff);
                            hitPos = lerp(lastRayPositionWS, currentPos, t);
                            float3 interpNDC = ComputeNDCWithZ(hitPos, worldToClip);
                            #if (UNITY_REVERSED_Z == 0)
                                interpNDC.z = interpNDC.z * 0.5 + 0.5;
                            #endif
                            hitUV = interpNDC.xy;
                        }

                        float3 sceneColor = SAMPLE_TEXTURE2D_X(_BlitTexture, sampler_LinearClamp, hitUV).rgb;
                        float3 transmittance = Transmittance(opticalDepth);

                        result.didHit = true;
                        result.color = sceneColor * transmittance;
                        result.position = hitPos;
                        result.distance = entryDist + traveled;
                        result.screenUV = hitUV;
                        result.rayDirection = rayDir;
                        return result;
                    }

                    if (!startBinarySearch)
                    {
                        currStepSize += currStepSize * 0.05;
                    }

                    lastDepthDiff = depthDiff;
                }

                if (debugActive && !debugExitLogged && traveled >= maxDist)
                {
                    float exitDensity;
                    float exitIoR;
                    float3 exitGradIoR;
                    DebugComputeCurrentMedium(currentPos, rayDir, exitDensity, exitIoR, exitGradIoR);
                    DebugStoreStep(
                        debugTraceIndex,
                        debugStepCount,
                        2,
                        traveled + debugEntryOffset,
                        0.0,
                        currentPos,
                        rayDir,
                        ComputeSampleUVW(currentPos),
                        exitDensity,
                        exitIoR,
                        exitGradIoR);
                    debugExitLogged = true;
                }

                if (debugActive && !debugMaxStepEventLogged && renderStepCount >= SS_MAX_STEPS && traveled < maxDist)
                {
                    float limitDensity;
                    float limitIoR;
                    float3 limitGradIoR;
                    DebugComputeCurrentMedium(currentPos, rayDir, limitDensity, limitIoR, limitGradIoR);
                    DebugStoreStep(
                        debugTraceIndex,
                        debugStepCount,
                        4,
                        traveled + debugEntryOffset,
                        0.0,
                        currentPos,
                        rayDir,
                        ComputeSampleUVW(currentPos),
                        limitDensity,
                        limitIoR,
                        limitGradIoR);
                }

                float3 exitNDC = ComputeNDCWithZ(currentPos, worldToClip);
                float2 exitUV = saturate(exitNDC.xy);
                float3 bgColor = SAMPLE_TEXTURE2D_X(_BlitTexture, sampler_LinearClamp, exitUV).rgb;
                float3 transmittance = Transmittance(opticalDepth);

                result.color = bgColor * transmittance;
                result.position = currentPos;
                result.distance = entryDist + traveled;
                result.screenUV = exitUV;
                result.rayDirection = rayDir;
                return result;
            }

            RayHit ScreenSpaceRayMarchFluid(float2 uv, float stepSizeParam, bool debugActive, int debugTraceIndex, int2 pixelCoord)
            {
                RayHit result = InitializeRayHit();

                float3 rayPos = GetCameraPositionWS();

                float2 ndc = uv * 2.0 - 1.0;
                #if UNITY_UV_STARTS_AT_TOP
                    ndc.y = -ndc.y;
                #endif
                float4 clipPos = float4(ndc, 0.5, 1.0);
                float4 worldPos = mul(UNITY_MATRIX_I_VP, clipPos);
                worldPos.xyz /= worldPos.w;
                float3 rayDir = normalize(worldPos.xyz - rayPos);

                result.rayDirection = rayDir;

                float3 boxMin = _VolumePosition - boundsSize * 0.5;
                float3 boxMax = _VolumePosition + boundsSize * 0.5;
                float2 boundsInfo = RayBoxDst(boxMin, boxMax, rayPos, rayDir);
                float entryDist = boundsInfo.x;
                float insideDist = boundsInfo.y;
                float3 entryPos = rayPos + rayDir * entryDist;

                if (debugActive)
                {
                    DebugStoreHeader(debugTraceIndex, pixelCoord, -1, rayPos, rayDir, entryDist, entryPos);
                }

                if (insideDist <= 0)
                {
                    result.color = SAMPLE_TEXTURE2D_X(_BlitTexture, sampler_LinearClamp, uv).rgb;
                    return result;
                }

                float3 entryNDC = ComputeNDCWithZ(entryPos, GetWorldToHClipMatrix());
                float deviceDepthAtEntry = SampleSceneDepth(uv);
                float sceneDepthAtEntry = LinearEyeDepth(deviceDepthAtEntry, _ZBufferParams);
                float entryLinearDepth = LinearEyeDepth(entryNDC.z, _ZBufferParams);

                if (sceneDepthAtEntry < entryLinearDepth && deviceDepthAtEntry != UNITY_RAW_FAR_CLIP_VALUE)
                {
                    if (debugActive)
                    {
                        DebugStoreHeader(debugTraceIndex, pixelCoord, -2, rayPos, rayDir, entryDist, entryPos);
                    }

                    result.color = SAMPLE_TEXTURE2D_X(_BlitTexture, sampler_LinearClamp, uv).rgb;
                    return result;
                }

                if (debugActive)
                {
                    DebugStoreHeader(debugTraceIndex, pixelCoord, 1, rayPos, rayDir, entryDist, entryPos);
                }

                float3 startPos = rayPos + rayDir * (entryDist + TinyNudge);
                float maxDist = insideDist - TinyNudge * 2;

                return ScreenSpaceCurvedRayMarching(startPos, rayDir, entryDist, maxDist, stepSizeParam, debugTraceIndex, debugActive, entryPos);
            }

            float4 Frag(Varyings input) : SV_Target
            {
                int2 pixelCoord = int2(input.positionCS.xy);
                int debugTraceIndex = FindDebugTraceIndex(pixelCoord);
                bool debugActive = debugTraceIndex >= 0;
                RayHit result = ScreenSpaceRayMarchFluid(input.texcoord, viewMarchStepSize, debugActive, debugTraceIndex, pixelCoord);
                return float4(result.color, 1);
            }

            ENDHLSL
        }
    }
}
