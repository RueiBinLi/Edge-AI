#pragma once
// =============================================================================
//  smem.cuh  --  Kernel 3: shared-memory tiling, one output per thread (Part 3a)
// =============================================================================
//  Row-major, FP32:  C (MxN) = alpha * A (MxK) @ B (KxN) + beta * C
//  YOU EDIT THIS FILE.
// =============================================================================
#include "../common.cuh"

// Each block computes a TILE x TILE tile of C. Loop over K in steps of TILE:
// cooperatively load a tile of A and a tile of B into __shared__ memory,
// __syncthreads(), accumulate, __syncthreads().
// Must handle ANY M, N, K (zero-pad out-of-range tile elements).
template <int TILE>
__global__ void sgemm_smem(int M, int N, int K, float alpha,
                           const float* __restrict__ A,
                           const float* __restrict__ B, float beta,
                           float* __restrict__ C) {
  // TODO(Part 3a)
  const int ty = threadIdx.y;
  const int tx = threadIdx.x;
  const int row = blockIdx.y * TILE + ty;
  const int col = blockIdx.x * TILE + tx;

  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];
  float acc = 0.f;

  for (int t = 0; t < ceil_div(K, TILE); t++) {
    if (row < M && t * TILE + tx < K) 
      As[ty][tx] = A[(size_t)row * K + t * TILE + tx];
    else
      As[ty][tx] = 0;

    if (col < N && t * TILE + ty < K)
      Bs[ty][tx] = B[(size_t)(t * TILE + ty) * N + col];
    else
      Bs[ty][tx] = 0;

    __syncthreads();

    for (int i = 0; i < TILE; i++) {
      acc += As[ty][i] * Bs[i][tx];
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    const size_t idx = (size_t)row * N + col;
    C[idx] = alpha * acc + beta * C[idx];
  }
}

template <int TILE>
bool launch_smem(int M, int N, int K, float alpha, const float* A,
                 const float* B, float beta, float* C, cudaStream_t s) {
  // TODO(Part 3a)
  dim3 block(TILE, TILE);
  dim3 grid(ceil_div(N, TILE), ceil_div(M, TILE));
  sgemm_smem<TILE><<<grid, block, 0, s>>>(M, N, K, alpha, A, B, beta, C);
  return true;
}
