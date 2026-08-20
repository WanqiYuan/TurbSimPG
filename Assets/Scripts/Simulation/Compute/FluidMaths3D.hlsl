static const float PI = 3.1415926;

const float K_SpikyPow2;
const float K_SpikyPow3;
const float K_SpikyPow2Grad;
const float K_SpikyPow3Grad;

float LinearKernel(float dst, float radius)
{
	if (dst < radius)
    {
        return 1 - dst / radius;
    }
    return 0;
}

float SmoothingKernelPoly6(float dst, float radius)
{
	if (dst < radius)
	{
		float scale = 315 / (64 * PI * pow(abs(radius), 9));
		float v = radius * radius - dst * dst;
		return v * v * v * scale;
	}
	return 0;
}

float SpikyKernelPow3(float dst, float radius)
{
	if (dst < radius)
	{
		float v = radius - dst;
		return v * v * v * K_SpikyPow3;
	}
	return 0;
}


//Integrate[(h-r)^2 r^2 Sin[θ], {r, 0, h}, {θ, 0, π}, {φ, 0, 2*π}]
float SpikyKernelPow2(float dst, float radius)
{
	if (dst < radius)
	{
		float v = radius - dst;
		return v * v * K_SpikyPow2;
	}
	return 0;
}

float DerivativeSpikyPow3(float dst, float radius)
{
	if (dst <= radius)
	{
		float v = radius - dst;
		return -v * v * K_SpikyPow3Grad;
	}
	return 0;
}

float DerivativeSpikyPow2(float dst, float radius)
{
	if (dst <= radius)
	{
		float v = radius - dst;
		return -v * K_SpikyPow2Grad;
	}
	return 0;
}

float DensityKernel(float dst, float radius)
{
	//return SmoothingKernelPoly6(dst, radius);
	return SpikyKernelPow2(dst, radius);
}

float NearDensityKernel(float dst, float radius)
{
	return SpikyKernelPow3(dst, radius);
}

float DensityDerivative(float dst, float radius)
{
	return DerivativeSpikyPow2(dst, radius);
}

float NearDensityDerivative(float dst, float radius)
{
	return DerivativeSpikyPow3(dst, radius);
}

// ------------------------------------------------------------
// Desbrun's Spiky Kernel (3D)
// W(r)      = 15/(π h^6) * (h - r)^3,   0 <= r < h;  else 0
// dW/dr     = -45/(π h^6) * (h - r)^2,  0 <= r < h;  else 0
// ∇W(r⃗)    = (dW/dr) * (r⃗ / r)
// Inputs:
//   dst    = |r⃗| (= distance between two particles)
//   radius = h   (= smoothing length)
// ------------------------------------------------------------
#ifndef SISSM_SPIKY_KERNEL_3D_INCLUDED
#define SISSM_SPIKY_KERNEL_3D_INCLUDED

static const float SISSM_PI = 3.14159265358979323846;
static const float SISSM_EPS = 1e-6;

// 核值：W(r)
inline float SpikyKernel3D(float dst, float radius)
{
	// branchless：x = max(h - r, 0)
	float x = max(radius - dst, 0.0);
	// 15/(π h^6)
	float h2 = radius * radius;
	float h4 = h2 * h2;
	float h6 = h4 * h2;
	float k = 15.0 / (SISSM_PI * h6);
	return k * x * x * x;           // (h - r)^3
}

// 径向导数：dW/dr（标量）
// 用于构建 SISSM 的 A_ij 系数（Eq.14 类似项）
inline float SpikyKernel3D_dWdr(float dst, float radius)
{
	float x = max(radius - dst, 0.0);
	// -45/(π h^6) * (h - r)^2
	float h2 = radius * radius;
	float h4 = h2 * h2;
	float h6 = h4 * h2;
	float k = -45.0 / (SISSM_PI * h6);
	return k * x * x;
}

// 向量梯度：∇W(r⃗) = (dW/dr) * (r⃗ / r)
inline float3 SpikyKernel3D_Grad(float3 rij, float radius)
{
	float r = length(rij);
	if (r <= SISSM_EPS) return 0.0.xxx;
	float dw = SpikyKernel3D_dWdr(r, radius);
	return (dw / r) * rij;
}

inline float dWdr(float r, float h) 
{ 
	return SpikyKernel3D_dWdr(r, h); 
}

#endif // SISSM_SPIKY_KERNEL_3D_INCLUDED


