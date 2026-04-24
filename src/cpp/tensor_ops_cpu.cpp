#include "cupyccx/tensor_ops.hpp"
#include <stdexcept>

// ---------------------------------------------------------------------------
// CPU backend — explicit loop implementations of all tensor contractions.
//
// For production-scale systems these loops are the bottleneck; they can be
// replaced with BLAS DGEMM calls (OpenBLAS / MKL / Accelerate) once the
// index reorderings for each contraction are worked out.  The loop versions
// here are the reference implementation: correct, readable, and verifiable.
// ---------------------------------------------------------------------------

namespace cupyccx {
namespace tensor_ops {

static bool s_gpu = false;

void init(bool use_gpu) {
#ifdef CUPYCCX_CUDA
    if (use_gpu) {
        extern void gpu_backend_init();
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
        extern void gpu_backend_finalize();
        gpu_backend_finalize();
        s_gpu = false;
        return;
    }
#endif
}

bool gpu_active() { return s_gpu; }

// Stub: public dgemm symbol required by the header; only the CUDA backend
// provides a real implementation via cuBLAS.
void dgemm(int, int, int, real_t, const real_t*, const real_t*, real_t, real_t*) {
    throw std::runtime_error(
        "tensor_ops::dgemm called on CPU backend — "
        "build with CUPYCCX_CUDA=ON to use cuBLAS DGEMM.");
}

// ---------------------------------------------------------------------------
// R[i,j,a,b] += alpha * sum_c  T2[i,j,a,c] * F_vv[b,c]
// ---------------------------------------------------------------------------
void contract_ijac_bc(const Tensor4& T2, const Matrix& F_vv,
                      Tensor4& R, real_t alpha) {
    const int o = static_cast<int>(T2.n0);
    const int v = static_cast<int>(T2.n2);
    for (int i = 0; i < o; ++i)
    for (int j = 0; j < o; ++j)
    for (int a = 0; a < v; ++a)
    for (int b = 0; b < v; ++b) {
        real_t s = 0.0;
        for (int c = 0; c < v; ++c) s += T2(i, j, a, c) * F_vv(b, c);
        R(i, j, a, b) += alpha * s;
    }
}

// ---------------------------------------------------------------------------
// R[i,j,a,b] += alpha * sum_k  T2[i,k,a,b] * F_oo[k,j]
// ---------------------------------------------------------------------------
void contract_ikab_kj(const Tensor4& T2, const Matrix& F_oo,
                      Tensor4& R, real_t alpha) {
    const int o = static_cast<int>(T2.n0);
    const int v = static_cast<int>(T2.n2);
    for (int i = 0; i < o; ++i)
    for (int j = 0; j < o; ++j)
    for (int a = 0; a < v; ++a)
    for (int b = 0; b < v; ++b) {
        real_t s = 0.0;
        for (int k = 0; k < o; ++k) s += T2(i, k, a, b) * F_oo(k, j);
        R(i, j, a, b) += alpha * s;
    }
}

// ---------------------------------------------------------------------------
// R[i,j,a,b] += alpha * sum_{kl}  W[k,l,i,j] * T2[k,l,a,b]
// ---------------------------------------------------------------------------
void contract_klij_klab(const Tensor4& W, const Tensor4& T2,
                        Tensor4& R, real_t alpha) {
    const int o = static_cast<int>(T2.n0);
    const int v = static_cast<int>(T2.n2);
#ifdef CUPYCCX_CUDA
    if (s_gpu) {
        extern void gpu_contract_klij_klab(const Tensor4&, const Tensor4&,
                                           Tensor4&, real_t);
        gpu_contract_klij_klab(W, T2, R, alpha);
        return;
    }
#endif
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
}

// ---------------------------------------------------------------------------
// R[i,j,a,b] += alpha * sum_{cd}  W[a,b,c,d] * T2[i,j,c,d]
// ---------------------------------------------------------------------------
void contract_abcd_ijcd(const Tensor4& W, const Tensor4& T2,
                        Tensor4& R, real_t alpha) {
    const int o = static_cast<int>(T2.n0);
    const int v = static_cast<int>(T2.n2);
#ifdef CUPYCCX_CUDA
    if (s_gpu) {
        extern void gpu_contract_abcd_ijcd(const Tensor4&, const Tensor4&,
                                           Tensor4&, real_t);
        gpu_contract_abcd_ijcd(W, T2, R, alpha);
        return;
    }
#endif
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
}

// ---------------------------------------------------------------------------
// R[i,j,a,b] += alpha * sum_{kc}  W[k,b,c,j] * T2[i,k,a,c]
// ---------------------------------------------------------------------------
void contract_kbcj_ikac(const Tensor4& W, const Tensor4& T2,
                        Tensor4& R, real_t alpha) {
    const int o = static_cast<int>(T2.n0);
    const int v = static_cast<int>(T2.n2);
#ifdef CUPYCCX_CUDA
    if (s_gpu) {
        extern void gpu_contract_kbcj_ikac(const Tensor4&, const Tensor4&,
                                           Tensor4&, real_t);
        gpu_contract_kbcj_ikac(W, T2, R, alpha);
        return;
    }
#endif
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
}

}  // namespace tensor_ops
}  // namespace cupyccx
