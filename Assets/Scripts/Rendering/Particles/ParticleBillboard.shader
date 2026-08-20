Shader "Fluid/ParticleBillboard" {
	Properties {
		
	}
	SubShader {

		Tags {"Queue"="Geometry" }

		Pass {

			CGPROGRAM

			#pragma vertex vert
			#pragma fragment frag
			#pragma target 4.5

			#include "UnityCG.cginc"
			
			StructuredBuffer<float3> Positions;
			StructuredBuffer<float3> Velocities;
			StructuredBuffer<float> Temperature;
			Texture2D<float4> ColourMap;
			SamplerState linear_clamp_sampler;
			float velocityMax;
			float temperatureMax;
			int useTemperatureMapping;

			float scale;
			float3 colour;

			float4x4 localToWorld;

			struct v2f
			{
				float4 pos : SV_POSITION;
				float2 uv : TEXCOORD0;
				float3 colour : TEXCOORD1;
				float3 normal : NORMAL;
			};

			v2f vert (appdata_full v, uint instanceID : SV_InstanceID)
			{
				v2f o;
				o.uv = v.texcoord;
				o.normal = v.normal;
				
				float3 centreWorld = Positions[instanceID];
				float3 objectVertPos = v.vertex * scale * 2;
				float4 viewPos = mul(UNITY_MATRIX_V, float4(centreWorld, 1)) + float4(objectVertPos, 0);
				o.pos = mul(UNITY_MATRIX_P, viewPos);

				if (useTemperatureMapping)
				{
					// Map temperature to color
					float temperature = Temperature[instanceID];
					float tempT = saturate(temperature / temperatureMax);
					o.colour = ColourMap.SampleLevel(linear_clamp_sampler, float2(tempT, 0.5), 0);
				}
				else
				{
					// Map velocity to color
					float speed = length(Velocities[instanceID]);
					float speedT = saturate(speed / velocityMax);
					o.colour = ColourMap.SampleLevel(linear_clamp_sampler, float2(speedT, 0.5), 0);
				}

				return o;
			}

			float4 frag (v2f i) : SV_Target
			{
				float shading = saturate(dot(_WorldSpaceLightPos0.xyz, i.normal));
				shading = (shading + 0.6) / 1.4;
				return float4(i.colour, 1);
			}

			ENDCG
		}
	}
}