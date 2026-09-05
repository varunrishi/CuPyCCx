#include "cupyccx/tensor_ops.hpp"
#include <stdexcept>
#include <vector>

// ---------------------------------------------------------------------------
// CPU backend — tensor contractions with CBLAS DGEMM fast paths.
//
// When CUPYCCX_HAVE_CBLAS is defined (set by CMake when BLAS is found), each
// O(N^6) contraction is implemented as one or two cblas_dgemm calls, which
// route to the vendor BLAS (OpenBLAS / MKL / Accelerate) and achieve near-peak
// FP64 throughput.  The explicit for-loop reference implementations are kept as
// the #else branch — correct, readable, and used when no BLAS is linked.
// ---------------------------------------------------------------------------

#ifdef CUPYCCX_HAVE_CBLAS
#  ifdef CUPYCCX_BLAS_ACCELERATE
#    include <Accelerate/Accelerate.h>
#  else
#    include <cblas.h>
#  endif
#endif

#ifdef CUPYCCX_CUDA
// Forward-declare GPU entry points with C linkage to match their definitions
// in tensor_ops_gpu.cu (which uses extern "C"). Declarations must be at file
// scope — linkage-specifications are not valid at block scope in C++.
extern "C" {
void gpu_backend_init();
void gpu_backend_finalize();
}
#endif

namespace cupyccx {
namespace tensor_ops {

static bool s_gpu = false;

void init(bool use_gpu) {
#ifdef CUPYCCX_CUDA
    if (use_gpu) {
        gpu_backend_init();
        s_gpu = true;
        return;
    }
#endif
    s_gpu = false;
}

void finalize() {
#ifdef CUPYCCX_CUDA
    if (s_gpu) {
        gpu_backend_finalize();
        s_gpu = false;
        return;
    }
#endif
}

bool gpu_active() { return s_gpu; }

// No-op on CPU-only builds; the CUDA backend provides the real implementation.
#ifndef CUPYCCX_CUDA
void gpu_upload_integrals(const Tensor4&, const Tensor4&, const Tensor4&, const void*) {}
void gpu_upload_oovv(const Tensor4&, const void*) {}
void gpu_begin_residual(const Tensor4&, const Tensor4&) {}
void gpu_end_residual(Tensor4&) {}
#endif

// Stub: public dgemm symbol required by the header; only the CUDA backend
// provides a real implementation via cuBLAS.
void dgemm(int, int, int, real_t, const real_t*, const real_t*, real_t, real_t*) {
    throw std::runtime_error(
        "tensor_ops::dgemm called on CPU backend — "
        "build with CUPYCCX_CUDA=ON to use cuBLAS DGEMM.");
}

void contract_oovv_t2_t2(const Tensor4& oovv, const Tensor4& t2,
                          Tensor4& R, real_t alpha) {
#ifdef CUPYCCX_CUDA
    if (s_gpu) {
        extern void gpu_contract_oovv_t2_t2(const Tensor4&, const Tensor4&,
                                             Tensor4&, real_t);
        gpu_contract_oovv_t2_t2(oovv, t2, R, alpha);
        return;
    }
#endif
    throw std::runtime_error(
        "tensor_ops::contract_oovv_t2_t2 is GPU-only; "
        "build with CUPYCCX_CUDA=ON.");
}

void contract_qD_Pijab(const Tensor4& oovv, const Tensor4& t2,
                        Tensor4& R, real_t alpha) {
#ifdef CUPYCCX_CUDA
    if (s_gpu) {
        extern void gpu_contract_qD_Pijab(const Tensor4&, const Tensor4&,
                                           Tensor4&, real_t);
        gpu_contract_qD_Pijab(oovv, t2, R, alpha);
        return;
    }
#endif
    throw std::runtime_error(
        "tensor_ops::contract_qD_Pijab is GPU-only; "
        "build with CUPYCCX_CUDA=ON.");
}

// ---------------------------------------------------------------------------
// R[i,j,a,b] += alpha * sum_c  T2[i,j,a,c] * F_vv[b,c]
//
// DGEMM: treat (i,j,a) as a flat row index → R[(o²v), v] += alpha * T2[(o²v), v] @ F_vv^T[v, v]
// ---------------------------------------------------------------------------
void contract_ijac_bc(const Tensor4& T2, const Matrix& F_vv,
                      Tensor4& R, real_t alpha) {
    const int o = static_cast<int>(T2.n0);
    const int v = static_cast<int>(T2.n2);
#ifdef CUPYCCX_HAVE_CBLAS
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                o * o * v, v, v,
                alpha, T2.ptr(), v,
                F_vv.data(), v,
                1.0, R.ptr(), v);
#else
    // Reference: explicit loop
    for (int i = 0; i < o; ++i)
    for (int j = 0; j < o; ++j)
    for (int a = 0; a < v; ++a)
    for (int b = 0; b < v; ++b) {
        real_t s = 0.0;
        for (int c = 0; c < v; ++c) s += T2(i, j, a, c) * F_vv(b, c);
        R(i, j, a, b) += alpha * s;
    }
#endif
}

// ---------------------------------------------------------------------------
// R[i,j,a,b] += alpha * sum_k  T2[i,k,a,b] * F_oo[k,j]
//
// DGEMM (per i): R_i[(o, v²)] += alpha * F_oo^T[(o, o)] @ T2_i[(o, v²)]
// ---------------------------------------------------------------------------
void contract_ikab_kj(const Tensor4& T2, const Matrix& F_oo,
                      Tensor4& R, real_t alpha) {
    const int o = static_cast<int>(T2.n0);
    const int v = static_cast<int>(T2.n2);
    const int vv = v * v;
#ifdef CUPYCCX_HAVE_CBLAS
    // Loop over i; each slice T2[i,k,a,b] is contiguous as (o, v²)
    for (int i = 0; i < o; ++i) {
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    o, vv, o,
                    alpha, F_oo.data(), o,
                    T2.ptr() + i * o * vv, vv,
                    1.0, R.ptr() + i * o * vv, vv);
    }
#else
    // Reference: explicit loop
    for (int i = 0; i < o; ++i)
    for (int j = 0; j < o; ++j)
    for (int a = 0; a < v; ++a)
    for (int b = 0; b < v; ++b) {
        real_t s = 0.0;
        for (int k = 0; k < o; ++k) s += T2(i, k, a, b) * F_oo(k, j);
        R(i, j, a, b) += alpha * s;
    }
#endif
}

// ---------------------------------------------------------------------------
// R[i,j,a,b] += alpha * sum_{kl}  W[k,l,i,j] * T2[k,l,a,b]
//
// DGEMM: R[(o², v²)] += alpha * W^T[(o², o²)] @ T2[(o², v²)]
//   Both W and T2 are stored with (kl) as the leading (o²) dimension.
// ---------------------------------------------------------------------------
void contract_klij_klab(const Tensor4& W, const Tensor4& T2,
                        Tensor4& R, real_t alpha) {
    const int o = static_cast<int>(T2.n0);
    const int v = static_cast<int>(T2.n2);
    const int oo = o * o, vv = v * v;
#ifdef CUPYCCX_CUDA
    if (s_gpu) {
        extern void gpu_contract_klij_klab(const Tensor4&, const Tensor4&,
                                           Tensor4&, real_t);
        gpu_contract_klij_klab(W, T2, R, alpha);
        return;
    }
#endif
#ifdef CUPYCCX_HAVE_CBLAS
    // W is stored as (kl, ij) = (oo, oo); transposing gives W^T[(ij),(kl)].
    // R[(ij),(ab)] += alpha * W^T[(ij),(kl)] @ T2[(kl),(ab)]
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                oo, vv, oo,
                alpha, W.ptr(), oo,
                T2.ptr(), vv,
                1.0, R.ptr(), vv);
#else
    // Reference: explicit loop
    for (int i = 0; i < o; ++i)
    for (int j = 0; j < o; ++j)
    for (int a = 0; a < v; ++a)
    for (int b = 0; b < v; ++b) {
        real_t s = 0.0;
        for (int k = 0; k < o; ++k)
        for (int l = 0; l < o; ++l)
            s += W(k, l, i, j) * T2(k, l, a, b);
        R(i, j, a, b) += alpha * s;
    }
#endif
}

// ---------------------------------------------------------------------------
// R[i,j,a,b] += alpha * sum_{cd}  W[a,b,c,d] * T2[i,j,c,d]
//
// DGEMM: R[(o², v²)] += alpha * T2[(o², v²)] @ W^T[(v², v²)]
//   Both T2 and W are laid out with the 2-index batch as the leading (v²) dim.
// ---------------------------------------------------------------------------
void contract_abcd_ijcd(const Tensor4& W, const Tensor4& T2,
                        Tensor4& R, real_t alpha) {
    const int o = static_cast<int>(T2.n0);
    const int v = static_cast<int>(T2.n2);
    const int oo = o * o, vv = v * v;
#ifdef CUPYCCX_CUDA
    if (s_gpu) {
        extern void gpu_contract_abcd_ijcd(const Tensor4&, const Tensor4&,
                                           Tensor4&, real_t);
        gpu_contract_abcd_ijcd(W, T2, R, alpha);
        return;
    }
#endif
#ifdef CUPYCCX_HAVE_CBLAS
    // T2 is stored as (ij, cd) = (oo, vv); W is stored as (ab, cd) = (vv, vv).
    // R[(ij),(ab)] += alpha * T2[(ij),(cd)] @ W^T[(ab),(cd)]
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                oo, vv, vv,
                alpha, T2.ptr(), vv,
                W.ptr(), vv,
                1.0, R.ptr(), vv);
#else
    // Reference: explicit loop
    for (int i = 0; i < o; ++i)
    for (int j = 0; j < o; ++j)
    for (int a = 0; a < v; ++a)
    for (int b = 0; b < v; ++b) {
        real_t s = 0.0;
        for (int c = 0; c < v; ++c)
        for (int d = 0; d < v; ++d)
            s += W(a, b, c, d) * T2(i, j, c, d);
        R(i, j, a, b) += alpha * s;
    }
#endif
}

// ---------------------------------------------------------------------------
// Shared permutation helper: build T2_perm[(ia),(kc)] = T2[i,k,a,c]
//                            and W_perm[(kc),(jb)]  = W[k,b,c,j]
// Used by both contract_kbcj_ikac_Pijab and contract_kbcj_ikac.
// ---------------------------------------------------------------------------
#ifdef CUPYCCX_HAVE_CBLAS
static void permute_ring_operands(const Tensor4& W, const Tensor4& T2,
                                  int o, int v,
                                  std::vector<real_t>& T2_perm,
                                  std::vector<real_t>& W_perm) {
    const int ov = o * v;
    T2_perm.resize(ov * ov);
    W_perm.resize(ov * ov);
    for (int i = 0; i < o; ++i)
    for (int k = 0; k < o; ++k)
    for (int a = 0; a < v; ++a)
    for (int c = 0; c < v; ++c)
        T2_perm[(i * v + a) * ov + (k * v + c)] = T2(i, k, a, c);

    for (int k = 0; k < o; ++k)
    for (int j = 0; j < o; ++j)
    for (int b = 0; b < v; ++b)
    for (int c = 0; c < v; ++c)
        W_perm[(k * v + c) * ov + (j * v + b)] = W(k, b, c, j);
}
#endif

// ---------------------------------------------------------------------------
// R[i,j,a,b] += alpha * P(ij)P(ab) sum_{kc} W[k,b,c,j] * T2[i,k,a,c]
//
// DGEMM strategy:
//   1. Permute: T2_perm[(ia),(kc)] = T2[i,k,a,c]
//               W_perm[(kc),(jb)]  = W[k,b,c,j]
//   2. R_ring[(ia),(jb)] = T2_perm @ W_perm   (DGEMM, (ov)×(ov) matrices)
//   3. Scatter with P(ij)P(ab):
//      R[i,j,a,b] += alpha * ( R_ring[ia,jb] − R_ring[ib,ja]
//                             − R_ring[ja,ib] + R_ring[jb,ia] )
// GPU: single permute+DGEMM+k_scatter_Pijab pass. CPU: explicit four-term loop.
// ---------------------------------------------------------------------------
void contract_kbcj_ikac_Pijab(const Tensor4& W, const Tensor4& T2,
                               Tensor4& R, real_t alpha) {
#ifdef CUPYCCX_CUDA
    if (s_gpu) {
        extern void gpu_contract_kbcj_ikac_Pijab(const Tensor4&, const Tensor4&,
                                                  Tensor4&, real_t);
        gpu_contract_kbcj_ikac_Pijab(W, T2, R, alpha);
        return;
    }
#endif
    const int o = static_cast<int>(T2.n0);
    const int v = static_cast<int>(T2.n2);
    const int ov = o * v;
#ifdef CUPYCCX_HAVE_CBLAS
    std::vector<real_t> T2_perm, W_perm, R_ring(ov * ov);
    permute_ring_operands(W, T2, o, v, T2_perm, W_perm);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                ov, ov, ov,
                1.0, T2_perm.data(), ov,
                W_perm.data(), ov,
                0.0, R_ring.data(), ov);
    // Scatter P(ij)P(ab) into R[i,j,a,b]
    for (int i = 0; i < o; ++i)
    for (int j = 0; j < o; ++j)
    for (int a = 0; a < v; ++a)
    for (int b = 0; b < v; ++b) {
        const int ia = i * v + a, jb = j * v + b;
        const int ib = i * v + b, ja = j * v + a;
        R(i, j, a, b) += alpha * (
             R_ring[ia * ov + jb]   // base
           - R_ring[ib * ov + ja]   // P(ab)
           - R_ring[ja * ov + ib]   // P(ij)
           + R_ring[jb * ov + ia]   // P(ij)P(ab)
        );
    }
#else
    // Reference: explicit four-term loop
    for (int i = 0; i < o; ++i)
    for (int j = 0; j < o; ++j)
    for (int a = 0; a < v; ++a)
    for (int b = 0; b < v; ++b) {
        real_t s = 0.0;
        for (int k = 0; k < o; ++k)
        for (int c = 0; c < v; ++c) {
            s += W(k, b, c, j) * T2(i, k, a, c)
               - W(k, a, c, j) * T2(i, k, b, c)
               - W(k, b, c, i) * T2(j, k, a, c)
               + W(k, a, c, i) * T2(j, k, b, c);
        }
        R(i, j, a, b) += alpha * s;
    }
#endif
}

// ---------------------------------------------------------------------------
// R[i,j,a,b] += alpha * sum_{kc}  W[k,b,c,j] * T2[i,k,a,c]
//
// DGEMM: same permute+DGEMM as Pijab variant; scatter is base term only.
// ---------------------------------------------------------------------------
void contract_kbcj_ikac(const Tensor4& W, const Tensor4& T2,
                        Tensor4& R, real_t alpha) {
    const int o = static_cast<int>(T2.n0);
    const int v = static_cast<int>(T2.n2);
    const int ov = o * v;
#ifdef CUPYCCX_CUDA
    if (s_gpu) {
        extern void gpu_contract_kbcj_ikac(const Tensor4&, const Tensor4&,
                                           Tensor4&, real_t);
        gpu_contract_kbcj_ikac(W, T2, R, alpha);
        return;
    }
#endif
#ifdef CUPYCCX_HAVE_CBLAS
    std::vector<real_t> T2_perm, W_perm, R_ring(ov * ov);
    permute_ring_operands(W, T2, o, v, T2_perm, W_perm);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                ov, ov, ov,
                1.0, T2_perm.data(), ov,
                W_perm.data(), ov,
                0.0, R_ring.data(), ov);
    for (int i = 0; i < o; ++i)
    for (int j = 0; j < o; ++j)
    for (int a = 0; a < v; ++a)
    for (int b = 0; b < v; ++b)
        R(i, j, a, b) += alpha * R_ring[(i * v + a) * ov + (j * v + b)];
#else
    // Reference: explicit loop
    for (int i = 0; i < o; ++i)
    for (int j = 0; j < o; ++j)
    for (int a = 0; a < v; ++a)
    for (int b = 0; b < v; ++b) {
        real_t s = 0.0;
        for (int k = 0; k < o; ++k)
        for (int c = 0; c < v; ++c)
            s += W(k, b, c, j) * T2(i, k, a, c);
        R(i, j, a, b) += alpha * s;
    }
#endif
}

}  // namespace tensor_ops
}  // namespace cupyccx
