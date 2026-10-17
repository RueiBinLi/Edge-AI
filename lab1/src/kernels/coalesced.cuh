#pragma once
// =============================================================================
//  coalesced.cuh  --  Kernel 2: coalesced global memory access (Part 2)
// =============================================================================
//  Row-major, FP32:  C (MxN) = alpha * A (MxK) @ B (KxN) + beta * C
//  YOU EDIT THIS FILE.
// =============================================================================
#include "../common.cuh"

// Same algorithm as the naive kernel. Change only how threads map to (row, col)
// so that consecutive threads in a warp touch consecutive addresses.
__global__ void sgemm_coalesced(int M, int N, int K, float alpha,
                                const float* __restrict__ A,
                                const float* __restrict__ B, float beta,
                                float* __restrict__ C) {
  // TODO(Part 2)
}

bool launch_coalesced(int M, int N, int K, float alpha, const float* A,
                      const float* B, float beta, float* C, cudaStream_t s) {
  // TODO(Part 2): choose grid/block, launch, and return true.
  return false;
}
