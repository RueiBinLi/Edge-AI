#pragma once
// =============================================================================
//  custom.cuh  --  Your own optimized kernels (Part 4)
// =============================================================================
//  Row-major, FP32:  C (MxN) = alpha * A (MxK) @ B (KxN) + beta * C
//  YOU EDIT THIS FILE.
// =============================================================================
#include "../common.cuh"

// e.g. a vectorized / double-buffered / swizzled variant of the reg2d kernel,
// a shape-dispatching launcher, ... Register each one in src/kernels.cuh.
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
    sgemm_vec4(int M, int N, int K, float alpha, const float* __restrict__ A,
                const float* __restrict__ B, float beta,
                float* __restrict__ C) {
  constexpr int NT = (BM / TM) * (BN / TN);  // threads per block
  // As is stored TRANSPOSED: As[k][m] holds A[m][k] of the current tile.
  __shared__ float As[BK][BM];
  __shared__ __align__(16) float Bs[BK][BN];
  const int tid = threadIdx.x;
  // This thread computes rows tr*TM .. tr*TM+TM-1 and
  // columns tc*TN .. tc*TN+TN-1 of the block's BM x BN tile.
  const int tr = tid / (BN / TN), tc = tid % (BN / TN);
  float acc[TM][TN] = {};
  float ra[TM], rb[TN];
  for (int k0 = 0; k0 < K; k0 += BK) {
    for (int idx = tid; idx < BM*BK/4; idx += NT) {
      int m = idx / (BK/4);
      int k = (idx % (BK/4)) * 4;
      int global_row = blockIdx.y * BM + m;
      int global_col = k0 + k;
      float4 v = *reinterpret_cast<const float4 *>(&A[(size_t)global_row * K + global_col]);
      As[k][m] = v.x;
      As[k+1][m] = v.y;
      As[k+2][m] = v.z;
      As[k+3][m] = v.w;
    }

    for (int idx = tid; idx < BK*BN/4; idx += NT) {
      int k = idx / (BN/4);
      int n = (idx % (BN/4)) * 4;
      int global_row = k0 + k;
      int global_col = blockIdx.x * BN + n;
      float4 v = *reinterpret_cast<const float4 *>(&B[(size_t)global_row * N + global_col]);
      *reinterpret_cast<float4*>(&Bs[k][n]) = v;
    }

    __syncthreads();
    // GIVEN: outer product. Per k, each thread loads TM values of A and TN
    // values of B into registers and does TM*TN FMAs with them.
    for (int k = 0; k < BK; ++k) {
      for (int i = 0; i < TM; ++i) ra[i] = As[k][tr * TM + i];
      for (int j = 0; j < TN; ++j) rb[j] = Bs[k][tc * TN + j];
      for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j) acc[i][j] += ra[i] * rb[j];
    }
    __syncthreads();
  }
  for (int i = 0; i < TM; i++) {
    for (int j = 0; j < TN; j++) {
      int row = blockIdx.y * BM + tr * TM + i;
      int col = blockIdx.x * BN + tc * TN + j;
      C[(size_t)row * N +  col] = alpha * acc[i][j] + beta * C[(size_t)row * N + col];
    }
  }
}

template <int BM, int BN, int BK, int TM, int TN>
bool launch_vec4(int M, int N, int K, float alpha, const float* A,
                  const float* B, float beta, float* C, cudaStream_t s) {
  static_assert(BM % TM == 0 && BN % TN == 0, "tile must divide block tile");
  constexpr int kThreads = (BM / TM) * (BN / TN);
  static_assert(kThreads <= 1024, "too many threads per block");
  if (M % BM || N % BN || K % BK) return false;
  dim3 grid(N / BN, M / BM);
  sgemm_vec4<BM, BN, BK, TM, TN>
      <<<grid, kThreads, 0, s>>>(M, N, K, alpha, A, B, beta, C);
  return true;
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM / TM) * (BN / TN), 3)
    sgemm_vec4occ(int M, int N, int K, float alpha, const float* __restrict__ A,
                const float* __restrict__ B, float beta,
                float* __restrict__ C) {
  constexpr int NT = (BM / TM) * (BN / TN);  // threads per block
  // As is stored TRANSPOSED: As[k][m] holds A[m][k] of the current tile.
  __shared__ float As[BK][BM];
  __shared__ __align__(16) float Bs[BK][BN];
  const int tid = threadIdx.x;
  // This thread computes rows tr*TM .. tr*TM+TM-1 and
  // columns tc*TN .. tc*TN+TN-1 of the block's BM x BN tile.
  const int tr = tid / (BN / TN), tc = tid % (BN / TN);
  float acc[TM][TN] = {};
  float ra[TM], rb[TN];
  for (int k0 = 0; k0 < K; k0 += BK) {
    for (int idx = tid; idx < BM*BK/4; idx += NT) {
      int m = idx / (BK/4);
      int k = (idx % (BK/4)) * 4;
      int global_row = blockIdx.y * BM + m;
      int global_col = k0 + k;
      float4 v = *reinterpret_cast<const float4 *>(&A[(size_t)global_row * K + global_col]);
      As[k][m] = v.x;
      As[k+1][m] = v.y;
      As[k+2][m] = v.z;
      As[k+3][m] = v.w;
    }

    for (int idx = tid; idx < BK*BN/4; idx += NT) {
      int k = idx / (BN/4);
      int n = (idx % (BN/4)) * 4;
      int global_row = k0 + k;
      int global_col = blockIdx.x * BN + n;
      float4 v = *reinterpret_cast<const float4 *>(&B[(size_t)global_row * N + global_col]);
      *reinterpret_cast<float4*>(&Bs[k][n]) = v;
    }

    __syncthreads();
    // GIVEN: outer product. Per k, each thread loads TM values of A and TN
    // values of B into registers and does TM*TN FMAs with them.
    for (int k = 0; k < BK; ++k) {
      for (int i = 0; i < TM; ++i) ra[i] = As[k][tr * TM + i];
      for (int j = 0; j < TN; ++j) rb[j] = Bs[k][tc * TN + j];
      for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j) acc[i][j] += ra[i] * rb[j];
    }
    __syncthreads();
  }
  for (int i = 0; i < TM; i++) {
    for (int j = 0; j < TN; j++) {
      int row = blockIdx.y * BM + tr * TM + i;
      int col = blockIdx.x * BN + tc * TN + j;
      C[(size_t)row * N +  col] = alpha * acc[i][j] + beta * C[(size_t)row * N + col];
    }
  }
}

template <int BM, int BN, int BK, int TM, int TN>
bool launch_vec4occ(int M, int N, int K, float alpha, const float* A,
                  const float* B, float beta, float* C, cudaStream_t s) {
  static_assert(BM % TM == 0 && BN % TN == 0, "tile must divide block tile");
  constexpr int kThreads = (BM / TM) * (BN / TN);
  static_assert(kThreads <= 1024, "too many threads per block");
  if (M % BM || N % BN || K % BK) return false;
  dim3 grid(N / BN, M / BM);
  sgemm_vec4occ<BM, BN, BK, TM, TN>
      <<<grid, kThreads, 0, s>>>(M, N, K, alpha, A, B, beta, C);
  return true;
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
    sgemm_vec4db(int M, int N, int K, float alpha, const float* __restrict__ A,
                const float* __restrict__ B, float beta,
                float* __restrict__ C) {
  constexpr int NT = (BM / TM) * (BN / TN);  // threads per block
  constexpr int LA = BM * BK / 4 / NT; // float4s of A per thread per tile
  constexpr int LB = BK * BN / 4 / NT;
  static_assert((BM * BK / 4) % NT == 0 && (BK * BN / 4) % NT == 0,
                "every thread must load the same number of float4s");
  // As is stored TRANSPOSED: As[k][m] holds A[m][k] of the current tile.
  __shared__ float As[BK][BM];
  __shared__ __align__(16) float Bs[BK][BN];
  const int tid = threadIdx.x;
  // This thread computes rows tr*TM .. tr*TM+TM-1 and
  // columns tc*TN .. tc*TN+TN-1 of the block's BM x BN tile.
  const int tr = tid / (BN / TN), tc = tid % (BN / TN);
  float acc[TM][TN] = {};
  float ra[TM], rb[TN];
  float4 va[LA], vb[LB];

  auto load_tile = [&](int kt) {
#pragma unroll
    for (int p = 0; p < LA; ++p) {
        int idx = tid + p * NT;
        int m = idx / (BK / 4);
        int k = (idx % (BK / 4)) * 4;
        int global_row = blockIdx.y * BM + m;
        int global_col = kt + k;
        va[p] = *reinterpret_cast<const float4 *>(&A[(size_t)global_row * K + global_col]);
    }
#pragma unroll
    for (int p = 0; p < LB; ++p) {
        int idx = tid + p * NT;
        int k = idx / (BN / 4);
        int n = (idx % (BN / 4)) * 4;
        int global_row = kt + k;
        int global_col = blockIdx.x * BN + n;
        vb[p] = *reinterpret_cast<const float4 *>(&B[(size_t)global_row * N + global_col]);
    }
  };

  auto store_tile = [&]() {
#pragma unroll
    for (int p = 0; p < LA; ++p) {
        int idx = tid + p * NT;
        int m = idx / (BK / 4);
        int k = (idx % (BK / 4)) * 4;
        As[k][m] = va[p].x;
        As[k+1][m] = va[p].y;
        As[k+2][m] = va[p].z;
        As[k+3][m] = va[p].w;
    }
#pragma unroll
    for (int p = 0; p < LB; ++p) {
        int idx = tid + p * NT;
        int k = idx / (BN / 4);
        int n = (idx % (BN / 4)) * 4;
        *reinterpret_cast<float4*>(&Bs[k][n]) = vb[p];
    }
  };

  load_tile(0);
  store_tile();
  __syncthreads();

  for (int k0 = 0; k0 < K; k0 += BK) {
    const bool has_next = k0 + BK < K;
    if (has_next) load_tile(k0 + BK);
    // GIVEN: outer product. Per k, each thread loads TM values of A and TN
    // values of B into registers and does TM*TN FMAs with them.
    for (int k = 0; k < BK; ++k) {
      for (int i = 0; i < TM; ++i) ra[i] = As[k][tr * TM + i];
      for (int j = 0; j < TN; ++j) rb[j] = Bs[k][tc * TN + j];
      for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j) acc[i][j] += ra[i] * rb[j];
    }
    __syncthreads();
    if (has_next) store_tile();
    __syncthreads();
  }
  for (int i = 0; i < TM; i++) {
    for (int j = 0; j < TN; j++) {
      int row = blockIdx.y * BM + tr * TM + i;
      int col = blockIdx.x * BN + tc * TN + j;
      C[(size_t)row * N +  col] = alpha * acc[i][j] + beta * C[(size_t)row * N + col];
    }
  }
}

template <int BM, int BN, int BK, int TM, int TN>
bool launch_vec4db(int M, int N, int K, float alpha, const float* A,
                  const float* B, float beta, float* C, cudaStream_t s) {
  static_assert(BM % TM == 0 && BN % TN == 0, "tile must divide block tile");
  constexpr int kThreads = (BM / TM) * (BN / TN);
  static_assert(kThreads <= 1024, "too many threads per block");
  if (M % BM || N % BN || K % BK) return false;
  dim3 grid(N / BN, M / BM);
  sgemm_vec4db<BM, BN, BK, TM, TN>
      <<<grid, kThreads, 0, s>>>(M, N, K, alpha, A, B, beta, C);
  return true;
}