//
// GPU tensor contractions for cupyccx
//
// Persistent device buffers (s_d_W, s_d_T2, s_d_R) are allocated on first use
// and reused across all contraction calls, eliminating per-call cudaMalloc /
// cudaFree overhead identified as the primary bottleneck via Nsight Systems.
// They are freed in gpu_backend_finalize().
//

#include "cuda_utils.cuh"
#include "cupyccx/types.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace cupyccx {

// ---------------------------------------------------------------------------
// Module-level state
// ---------------------------------------------------------------------------
static cuda_utils::CublasHandle*       s_handle = nullptr;
static bool                            s_active  = false;

// Persistent device buffers — survive across contraction calls and iterations.
// s_d_W is shared between W_oooo (oo*oo) and W_vvvv (vv*vv); ensure() keeps
// it large enough for whichever is bigger.
static cuda_utils::DeviceBuffer<double> s_d_W;
static cuda_utils::DeviceBuffer<double> s_d_T2;
static cuda_utils::DeviceBuffer<double> s_d_R;

// Grow the buffer to at least n elements; never shrinks.
static void ensure(cuda_utils::DeviceBuffer<double>& buf, std::size_t n) {
    if (buf.size < n)
        buf = cuda_utils::DeviceBuffer<double>(n);
}

extern "C" {

void gpu_backend_init() {
    if (!cuda_utils::has_cuda_device())
        throw std::runtime_error("cupyccx GPU backend: no CUDA device found");
    s_handle = new cuda_utils::CublasHandle();
    s_active = true;
}

void gpu_backend_finalize() {
    // Free persistent buffers before destroying the cuBLAS handle.
    s_d_W  = cuda_utils::DeviceBuffer<double>();
    s_d_T2 = cuda_utils::DeviceBuffer<double>();
    s_d_R  = cuda_utils::DeviceBuffer<double>();
    delete s_handle;
    s_handle = nullptr;
    s_active  = false;
}

}  // extern "C"

namespace tensor_ops {

// ---------------------------------------------------------------------------
// (1/2) sum_{kl}  W_oooo[k,l,i,j] * T2[k,l,a,b]  →  R[i,j,a,b]
//
// R[ij,ab] += alpha * W^T[ij,kl] * T2[kl,ab]   (DGEMM: M=oo, N=vv, K=oo)
// ---------------------------------------------------------------------------
void gpu_contract_klij_klab(const Tensor4& W, const Tensor4& T2,
                             Tensor4& R, double alpha) {
    const int o  = static_cast<int>(T2.n0);
    const int v  = static_cast<int>(T2.n2);
    const int oo = o * o, vv = v * v;

    ensure(s_d_W,  oo * oo);
    ensure(s_d_T2, oo * vv);
    ensure(s_d_R,  oo * vv);

    s_d_W.upload(W.ptr(),   oo * oo);
    s_d_T2.upload(T2.ptr(), oo * vv);
    s_d_R.upload(R.ptr(),   oo * vv);

    const double one = 1.0;
    CUBLAS_CHECK(cublasDgemm(s_handle->handle,
                             CUBLAS_OP_N, CUBLAS_OP_T,
                             vv, oo, oo,
                             &alpha,
                             s_d_T2.ptr, vv,
                             s_d_W.ptr,  oo,
                             &one,
                             s_d_R.ptr,  vv));

    s_d_R.download(R.ptr(), oo * vv);
}

// ---------------------------------------------------------------------------
// (1/2) sum_{cd}  W_vvvv[a,b,c,d] * T2[i,j,c,d]  →  R[i,j,a,b]
//
// R[oo,vv_ab] += alpha * T2[oo,vv_cd] * W^T[vv_cd,vv_ab]  (M=oo, N=vv, K=vv)
// ---------------------------------------------------------------------------
void gpu_contract_abcd_ijcd(const Tensor4& W, const Tensor4& T2,
                             Tensor4& R, double alpha) {
    const int o  = static_cast<int>(T2.n0);
    const int v  = static_cast<int>(T2.n2);
    const int oo = o * o, vv = v * v;

    ensure(s_d_W,  vv * vv);
    ensure(s_d_T2, oo * vv);
    ensure(s_d_R,  oo * vv);

    s_d_W.upload(W.ptr(),   vv * vv);
    s_d_T2.upload(T2.ptr(), oo * vv);
    s_d_R.upload(R.ptr(),   oo * vv);

    const double one = 1.0;
    CUBLAS_CHECK(cublasDgemm(s_handle->handle,
                             CUBLAS_OP_T, CUBLAS_OP_N,
                             vv, oo, vv,
                             &alpha,
                             s_d_W.ptr,  vv,
                             s_d_T2.ptr, vv,
                             &one,
                             s_d_R.ptr,  vv));

    s_d_R.download(R.ptr(), oo * vv);
}

// ---------------------------------------------------------------------------
// sum_{kc}  W_ovvo[k,b,c,j] * T2[i,k,a,c]  →  R[i,j,a,b]
//
// Mixed index ordering — falls back to CPU loops until a tensor-transpose
// pass is added to reformulate as batched DGEMM.
// ---------------------------------------------------------------------------
void gpu_contract_kbcj_ikac(const Tensor4& W, const Tensor4& T2,
                             Tensor4& R, double alpha) {
    const int o = static_cast<int>(T2.n0);
    const int v = static_cast<int>(T2.n2);
    for (int i = 0; i < o; ++i)
    for (int j = 0; j < o; ++j)
    for (int a = 0; a < v; ++a)
    for (int b = 0; b < v; ++b) {
        double s = 0.0;
        for (int k = 0; k < o; ++k)
        for (int c = 0; c < v; ++c)
            s += W(k, b, c, j) * T2(i, k, a, c);
        R(i, j, a, b) += alpha * s;
    }
}

}  // namespace tensor_ops
}  // namespace cupyccx
