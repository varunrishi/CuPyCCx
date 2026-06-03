#pragma once

#include "types.hpp"

// ---------------------------------------------------------------------------
// Tensor contraction dispatcher
//
// All heavy O(N^6) work is routed through this interface so that CPU and GPU
// implementations can be swapped without touching the CC driver.
// ---------------------------------------------------------------------------

namespace cupyccx {
namespace tensor_ops {

// Initialize the backend (call once before any contraction).
// If use_gpu == true and CUDA is available, the GPU backend is activated.
void init(bool use_gpu);

// Shut down the backend and release device resources.
void finalize();

// Returns true if the GPU backend is currently active.
bool gpu_active();

// Pre-upload static W integral tensors to device before the iteration loop.
// Subsequent calls with the same cache_key (pass scf.eri_antisym.ptr()) are
// no-ops — the device buffers are reused as long as the molecule/basis is
// unchanged.  Benefits DCD/pCCD where vvvv/oooo are bare (static) integrals.
void gpu_upload_integrals(const Tensor4& vvvv,
                          const Tensor4& oooo,
                          const Tensor4& ovvo,
                          const void*    cache_key);

// ---------------------------------------------------------------------------
// Core contractions used by the CCD/LCCD residual equations
// All arrays are assumed to be contiguous, row-major (C order).
// Shapes are passed explicitly so that kernels can be compiled for any size.
// ---------------------------------------------------------------------------

// result[i,j,a,b] += alpha * sum_c  A[i,j,a,c] * B[b,c]
//   (Fock contribution  P(ab) f_bc t^ac_ij)
void contract_ijac_bc(const Tensor4& T2,
                      const Matrix&  F_vv,
                      Tensor4&       result,
                      real_t         alpha = 1.0);

// result[i,j,a,b] -= alpha * sum_k  A[i,k,a,b] * B[k,j]
//   (Fock contribution  P(ij) f_kj t^ab_ik)
void contract_ikab_kj(const Tensor4& T2,
                      const Matrix&  F_oo,
                      Tensor4&       result,
                      real_t         alpha = 1.0);

// result[i,j,a,b] += (alpha/2) * sum_{kl}  W_oooo[k,l,i,j] * T2[k,l,a,b]
//   (Ring/ladder diagram: W_klij t^ab_kl)
void contract_klij_klab(const Tensor4& W_oooo,
                        const Tensor4& T2,
                        Tensor4&       result,
                        real_t         alpha = 0.5);

// result[i,j,a,b] += (alpha/2) * sum_{cd}  W_vvvv[a,b,c,d] * T2[i,j,c,d]
//   (Ladder diagram: W_abcd t^cd_ij)
void contract_abcd_ijcd(const Tensor4& W_vvvv,
                        const Tensor4& T2,
                        Tensor4&       result,
                        real_t         alpha = 0.5);

// result[i,j,a,b] += alpha * sum_{kc}  W_ovvo[k,b,c,j] * T2[i,k,a,c]
//   (Mixed ring+ladder, one of the P(ij)P(ab) terms)
void contract_kbcj_ikac(const Tensor4& W_ovvo,
                        const Tensor4& T2,
                        Tensor4&       result,
                        real_t         alpha = 1.0);

// General batched DGEMM used internally by the GPU backend
// C = alpha * A * B + beta * C  (row-major)
void dgemm(int M, int N, int K,
           real_t alpha, const real_t* A,
           const real_t* B,
           real_t beta,  real_t* C);

}  // namespace tensor_ops
}  // namespace cupyccx
