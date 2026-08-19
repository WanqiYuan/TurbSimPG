using Seb.Fluid.Simulation;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

/// <summary>
/// URP renderer feature for the live FluidSim density volume.
/// </summary>
public class RaymarchingRenderFeature : ScriptableRendererFeature
{
    [Header("Shader")]
    [Tooltip("The raymarching material (assign the URPRaymarching shader).")]
    [SerializeField] private Material m_RaymarchMaterial;

    private const string RaymarchingShaderName = "Fluid/URPRaymarching";

    private static readonly int DirToSunId = Shader.PropertyToID("dirToSun");
    private static readonly int DensityMapId = Shader.PropertyToID("DensityMap");
    private static readonly int BoundsSizeId = Shader.PropertyToID("boundsSize");
    private static readonly int VolumeValueOffsetId = Shader.PropertyToID("volumeValueOffset");
    private static readonly int IndexOfRefractionId = Shader.PropertyToID("indexOfRefraction");
    private static readonly int DensityMultiplierId = Shader.PropertyToID("densityMultiplier");
    private static readonly int GladstoneDaleKId = Shader.PropertyToID("K");
    private static readonly int FilterHighDensityDerivedIoRId = Shader.PropertyToID("_FilterHighDensityDerivedIoR");
    private static readonly int DensityDerivedIoRUpperThresholdId = Shader.PropertyToID("_DensityDerivedIoRUpperThreshold");
    private static readonly int ViewMarchStepSizeId = Shader.PropertyToID("viewMarchStepSize");
    private static readonly int LightStepSizeId = Shader.PropertyToID("lightStepSize");
    private static readonly int ExtinctionCoeffId = Shader.PropertyToID("extinctionCoeff");
    private static readonly int UseAdaptiveStepSizeId = Shader.PropertyToID("_UseAdaptiveStepSize");
    private static readonly int VolumeResolutionId = Shader.PropertyToID("_VolumeResolution");
    private static readonly int VolumePositionId = Shader.PropertyToID("_VolumePosition");
    private static readonly int UseIoRVolumeId = Shader.PropertyToID("_UseIoRVolume");
    private static readonly int IorTextureBoundsId = Shader.PropertyToID("_IORTextureBounds");
    private static readonly int SimulationRegionShapeId = Shader.PropertyToID("_SimulationRegionShape");
    private static readonly int EnableDebugPathTraceId = Shader.PropertyToID("_EnableDebugPathTrace");

    private RaymarchingPass raymarchingPass;
    private RaymarchingSceneReferences sceneReferences;

    public Material raymarchMaterial
    {
        get => m_RaymarchMaterial;
        set
        {
            if (value != null && value.shader == Shader.Find(RaymarchingShaderName))
            {
                m_RaymarchMaterial = value;
            }
        }
    }

    public override void Create()
    {
        if (m_RaymarchMaterial == null)
        {
            Debug.LogError("Raymarching material is missing!");
            return;
        }

        raymarchingPass ??= new RaymarchingPass(m_RaymarchMaterial)
        {
            renderPassEvent = RenderPassEvent.AfterRenderingPostProcessing
        };
    }

    public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
    {
        RaymarchingVolume volume = VolumeManager.instance.stack.GetComponent<RaymarchingVolume>();
        if (volume == null || !volume.IsActive())
        {
            return;
        }

        sceneReferences ??= Object.FindFirstObjectByType<RaymarchingSceneReferences>();

        FluidSim simulation = sceneReferences != null ? sceneReferences.sim : null;
        simulation ??= Object.FindFirstObjectByType<FluidSim>();

        // The renderer can run before FluidSim.Start initializes the 3D texture.
        // Skip those early frames instead of reporting a false missing-reference error.
        if (simulation == null || simulation.DensityMap == null)
        {
            return;
        }

        SetPerFrameShaderParams(volume, simulation);
        renderer.EnqueuePass(raymarchingPass);
    }

    private void SetPerFrameShaderParams(RaymarchingVolume volume, FluidSim simulation)
    {
        if (sceneReferences != null && sceneReferences.mainLight != null)
        {
            m_RaymarchMaterial.SetVector(DirToSunId, -sceneReferences.mainLight.transform.forward);
        }

        m_RaymarchMaterial.SetTexture(DensityMapId, simulation.DensityMap);
        m_RaymarchMaterial.SetVector(BoundsSizeId, simulation.Scale);
        m_RaymarchMaterial.SetFloat(VolumeValueOffsetId, volume.densityOffset.value);
        m_RaymarchMaterial.SetFloat(IndexOfRefractionId, volume.indexOfRefraction.value);
        m_RaymarchMaterial.SetFloat(DensityMultiplierId, volume.densityMultiplier.value / 1000f);
        m_RaymarchMaterial.SetFloat(GladstoneDaleKId, volume.gladstoneDaleK.value);
        m_RaymarchMaterial.SetInt(FilterHighDensityDerivedIoRId, volume.filterHighDensityDerivedIoR.value ? 1 : 0);
        m_RaymarchMaterial.SetFloat(DensityDerivedIoRUpperThresholdId, volume.densityDerivedIoRUpperThreshold.value);
        m_RaymarchMaterial.SetFloat(ViewMarchStepSizeId, volume.stepSize.value);
        m_RaymarchMaterial.SetFloat(LightStepSizeId, volume.lightStepSize.value);
        m_RaymarchMaterial.SetVector(ExtinctionCoeffId, volume.extinctionCoefficients.value);
        m_RaymarchMaterial.SetInt(UseAdaptiveStepSizeId, volume.useAdaptiveStepSize.value ? 1 : 0);
        m_RaymarchMaterial.SetVector(VolumeResolutionId,
            new Vector3(simulation.DensityMap.width, simulation.DensityMap.height, simulation.DensityMap.volumeDepth));
        m_RaymarchMaterial.SetVector(VolumePositionId, simulation.transform.position);
        m_RaymarchMaterial.SetInt(UseIoRVolumeId, 0);
        m_RaymarchMaterial.SetVector(IorTextureBoundsId, simulation.Scale);
        m_RaymarchMaterial.SetInt(SimulationRegionShapeId, (int)simulation.simulationRegionShape);
        m_RaymarchMaterial.SetInt(EnableDebugPathTraceId, 0);
    }

    private sealed class RaymarchingPass : ScriptableRenderPass
    {
        private const string ProfileTag = "URP Raymarching";
        private readonly Material material;

        public RaymarchingPass(Material material)
        {
            this.material = material;
            ConfigureInput(ScriptableRenderPassInput.Depth);
        }

        private sealed class PassData
        {
            internal Material material;
            internal TextureHandle cameraColor;
            internal TextureHandle cameraDepth;
            internal TextureHandle intermediate;
        }

        private static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            CommandBuffer commandBuffer = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
            Blitter.BlitCameraTexture(commandBuffer, data.cameraColor, data.intermediate);
            Blitter.BlitCameraTexture(commandBuffer, data.intermediate, data.cameraColor, data.material, 0);
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>(ProfileTag, out PassData passData);
            builder.AllowPassCulling(false);

            UniversalResourceData resourceData = frameData.Get<UniversalResourceData>();
            UniversalCameraData cameraData = frameData.Get<UniversalCameraData>();
            RenderTextureDescriptor descriptor = cameraData.cameraTargetDescriptor;
            descriptor.depthBufferBits = 0;

            passData.material = material;
            passData.cameraColor = resourceData.activeColorTexture;
            passData.cameraDepth = resourceData.activeDepthTexture;
            passData.intermediate = UniversalRenderer.CreateRenderGraphTexture(
                renderGraph, descriptor, "raymarchingIntermediate", false, FilterMode.Point, TextureWrapMode.Clamp);

            builder.UseTexture(passData.cameraColor, AccessFlags.ReadWrite);
            builder.UseTexture(passData.cameraDepth, AccessFlags.Read);
            builder.UseTexture(passData.intermediate, AccessFlags.ReadWrite);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => ExecutePass(data, context));
        }
    }
}
