using System;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Serialization;

namespace Seb.Fluid.Simulation
{

	public class Spawner3D : MonoBehaviour
	{
		public int particleSpawnDensity = 600;
		public float3 initialVel;
		public float jitterStrength;
		public bool showSpawnBounds;
		public SpawnRegion[] spawnRegions;

		[Header("Debug Info")] public int debug_num_particles;
		public float debug_spawn_volume;


		public SpawnData GetSpawnData()
		{
			return GetSpawnData(null, SimulationRegionShape.Box);
		}

		public SpawnData GetSpawnData(Transform simulationTransform, SimulationRegionShape simulationRegionShape)
		{
			List<float3> allPoints = new();
			List<float3> allVelocities = new();

			foreach (SpawnRegion region in spawnRegions)
			{
				int3 particlesPerAxis = region.CalculateParticleCountPerAxis(particleSpawnDensity);
				(float3[] points, float3[] velocities) = SpawnCube(particlesPerAxis, region.centre, region.Size);
				for (int i = 0; i < points.Length; i++)
				{
					if (simulationTransform == null ||
						simulationRegionShape == SimulationRegionShape.Box ||
						IsInsideSquarePyramid(points[i], simulationTransform))
					{
						allPoints.Add(points[i]);
						allVelocities.Add(velocities[i]);
					}
				}
			}

			debug_num_particles = allPoints.Count;

			return new SpawnData() { points = allPoints.ToArray(), velocities = allVelocities.ToArray() };
		}

		static bool IsInsideSquarePyramid(float3 worldPoint, Transform simulationTransform)
		{
			Vector3 localPoint = simulationTransform.InverseTransformPoint((Vector3)worldPoint);
			if (localPoint.y < -0.5f || localPoint.y > 0.5f)
			{
				return false;
			}

			float halfExtent = 0.5f * (0.5f - localPoint.y);
			return Mathf.Abs(localPoint.x) <= halfExtent && Mathf.Abs(localPoint.z) <= halfExtent;
		}

		(float3[] p, float3[] v) SpawnCube(int3 numPerAxis, Vector3 centre, Vector3 size)
		{
			int numPoints = numPerAxis.x * numPerAxis.y * numPerAxis.z;
			float3[] points = new float3[numPoints];
			float3[] velocities = new float3[numPoints];

			int i = 0;

			for (int x = 0; x < numPerAxis.x; x++)
			{
				for (int y = 0; y < numPerAxis.y; y++)
				{
					for (int z = 0; z < numPerAxis.z; z++)
					{
						float tx = numPerAxis.x > 1 ? x / (numPerAxis.x - 1f) : 0.5f;
						float ty = numPerAxis.y > 1 ? y / (numPerAxis.y - 1f) : 0.5f;
						float tz = numPerAxis.z > 1 ? z / (numPerAxis.z - 1f) : 0.5f;

						float px = (tx - 0.5f) * size.x + centre.x;
						float py = (ty - 0.5f) * size.y + centre.y;
						float pz = (tz - 0.5f) * size.z + centre.z;
						float3 jitter = UnityEngine.Random.insideUnitSphere * jitterStrength;
						points[i] = new float3(px, py, pz) + jitter;
						velocities[i] = initialVel;
						i++;
					}
				}
			}

			return (points, velocities);
		}



		void OnValidate()
		{
			debug_spawn_volume = 0;
			debug_num_particles = 0;

			if (spawnRegions != null)
			{
				for (int i = 0; i < spawnRegions.Length; i++)
				{
					SpawnRegion region = spawnRegions[i];
					region.Validate();
					spawnRegions[i] = region;

					debug_spawn_volume += region.Volume;
					int3 numPerAxis = region.CalculateParticleCountPerAxis(particleSpawnDensity);
					debug_num_particles += numPerAxis.x * numPerAxis.y * numPerAxis.z;
				}
			}
		}

		void OnDrawGizmos()
		{
			if (showSpawnBounds && !Application.isPlaying)
			{
				foreach (SpawnRegion region in spawnRegions)
				{
					Gizmos.color = region.debugDisplayCol;
					Gizmos.DrawWireCube(region.centre, region.Size);
				}
			}
		}

		[System.Serializable]
		public struct SpawnRegion
		{
			public Vector3 centre;

			[InspectorName("Size")]
			[Tooltip("Particle spawn region dimensions along the X, Y, and Z axes.")]
			public Vector3 dimensions;

			[FormerlySerializedAs("size")]
			[SerializeField, HideInInspector]
			float legacyUniformSize;

			public Color debugDisplayCol;

			public Vector3 Size
			{
				get
				{
					if (dimensions.x > 0 && dimensions.y > 0 && dimensions.z > 0)
					{
						return dimensions;
					}

					if (legacyUniformSize > 0)
					{
						return Vector3.one * legacyUniformSize;
					}

					return Vector3.one;
				}
			}

			public float Volume
			{
				get
				{
					Vector3 size = Size;
					return size.x * size.y * size.z;
				}
			}

			public int3 CalculateParticleCountPerAxis(int particleDensity)
			{
				Vector3 size = Size;
				float particlesPerUnit = (float)Math.Cbrt(Mathf.Max(0, particleDensity));
				return new int3(
					Mathf.Max(1, Mathf.RoundToInt(size.x * particlesPerUnit)),
					Mathf.Max(1, Mathf.RoundToInt(size.y * particlesPerUnit)),
					Mathf.Max(1, Mathf.RoundToInt(size.z * particlesPerUnit))
				);
			}

			public void Validate()
			{
				if (legacyUniformSize > 0 &&
					(dimensions.x <= 0 || dimensions.y <= 0 || dimensions.z <= 0))
				{
					dimensions = Vector3.one * legacyUniformSize;
				}
				else if (dimensions == Vector3.zero)
				{
					dimensions = Vector3.one;
				}

				dimensions.x = Mathf.Max(0.0001f, dimensions.x);
				dimensions.y = Mathf.Max(0.0001f, dimensions.y);
				dimensions.z = Mathf.Max(0.0001f, dimensions.z);
				legacyUniformSize = 0;
			}
		}

		public struct SpawnData
		{
			public float3[] points;
			public float3[] velocities;
		}
	}
}
