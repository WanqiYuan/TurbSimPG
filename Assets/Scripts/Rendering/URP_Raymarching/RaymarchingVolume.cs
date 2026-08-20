using System;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

[Serializable, VolumeComponentMenu("Rendering/Raymarching (URP)"), SupportedOnRenderPipeline(typeof(UniversalRenderPipelineAsset))]
public class RaymarchingVolume : VolumeComponent, IPostProcessComponent
{
    /// <summary>
    /// Enables fluid raymarching post-processing.
    /// </summary>
    [Header("General"), Tooltip("Enables fluid raymarching.")]
    public BoolParameter state = new BoolParameter(false, BoolParameter.DisplayType.EnumPopup, overrideState: true);

    // ── Ray Marching ──────────────────────────────────────────────────

    [Header("Ray Marching")]
    [Tooltip("Step size for view-ray marching inside the fluid volume.")]
    public ClampedFloatParameter stepSize = new ClampedFloatParameter(0.02f, 0.001f, 0.2f, overrideState: false);

    [Tooltip("Step size for light-ray marching (shadow / transmittance).")]
    public ClampedFloatParameter lightStepSize = new ClampedFloatParameter(0.4f, 0.01f, 1.0f, overrideState: false);

    [Tooltip("Enable adaptive step size based on local IoR curvature.")]
    public BoolParameter useAdaptiveStepSize = new BoolParameter(false, overrideState: false);

    // ── Fluid Volume ─────────────────────────────────────────────────

    [Header("Fluid Volume")]
    [Tooltip("Density offset subtracted from raw volume data.")]
    public FloatParameter densityOffset = new FloatParameter(150f, overrideState: false);

    [Tooltip("Multiplier applied to density values.")]
    public FloatParameter densityMultiplier = new FloatParameter(0.001f, overrideState: false);

    [Tooltip("Extinction coefficients (RGB) for Beer-Lambert absorption.")]
    public Vector3Parameter extinctionCoefficients = new Vector3Parameter(Vector3.zero, overrideState: false);

    // ── Refraction ───────────────────────────────────────────────────

    [Header("Refraction")]
    [Tooltip("Index of refraction of the fluid medium.")]
    public ClampedFloatParameter indexOfRefraction = new ClampedFloatParameter(1.33f, 1.0f, 2.0f, overrideState: false);

    [Tooltip("Gladstone-Dale constant K used for curved ray bending.")]
    public FloatParameter gladstoneDaleK = new FloatParameter(0.000003f, overrideState: false);

    [Tooltip("Replace density-derived IoR values above the upper threshold with the ambient-air IoR (1.0003). This does not modify pre-converted IoR volumes.")]
    public BoolParameter filterHighDensityDerivedIoR = new BoolParameter(false, overrideState: false);

    [Tooltip("Density-derived IoR values above this threshold are treated as anomalous when the high-IoR filter is enabled.")]
    public ClampedFloatParameter densityDerivedIoRUpperThreshold =
        new ClampedFloatParameter(1.0003f, 1.0003f, 2.0f, overrideState: false);

    // ── IPostProcessComponent ────────────────────────────────────────

    public bool IsActive() => state.value;

    // Unused since 2023.1
    public bool IsTileCompatible() => false;
}
