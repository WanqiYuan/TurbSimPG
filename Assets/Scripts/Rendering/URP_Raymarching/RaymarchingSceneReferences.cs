using UnityEngine;
using Seb.Fluid.Simulation;

/// <summary>
/// MonoBehaviour that holds scene-object references for the URP Raymarching
/// render feature.  Attach this to any GameObject in the scene and drag-assign
/// the references in the Inspector.  The RaymarchingRenderFeature will
/// discover it at runtime via FindObjectOfType.
/// </summary>
public class RaymarchingSceneReferences : MonoBehaviour
{
    [Header("Fluid Simulation")]
    [Tooltip("The FluidSim component providing the live DensityMap.")]
    public FluidSim sim;

    [Header("Scene Objects")]
    [Tooltip("Main directional light for shading.")]
    public Light mainLight;
}
