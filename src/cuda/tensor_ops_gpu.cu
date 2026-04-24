//
// GPU tensor contractions for cupyccx
//
// Each contraction is expressed as one or more cuBLAS DGEMM calls on device
// memory.  Host arrays are uploaded to device, the kernel fires, and results
// are accumulated back — all synchronously for now.  Async streams and
// persistent device allocations can be added as a follow-up optimisation.
//

#include "cuda_utils.cuh"
#include "cupyccx/types.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace cupyccx {

// ---------------------------------------------------------------------------
// Module-level state
// ---------------------------------------------------------------------------
static cuda_utils::CublasHandle* s_handle  = nullptr;
static bool                       s_active  = false;

extern "C" {

void gpu_backend_init() {
    if (!cuda_utils::has_cuda_device())
        throw std::runtime_error("cupyccx GPU backend: no CUDA device found");
    s_handle = new cuda_utils::CublasHandle();
    s_active = true;
}

void gpu_backend_finalize() {
    delete s_handle;
    s_handle = nullptr;
    s_active = false;
}

}  // extern "C"

// ---------------------------------------------------------------------------
// Helper: DGEMM wrapper (row-major, all NoTrans)
//
//   C[M,N] += alpha * A[M,K] * B[K,N]
//
// cuBLAS is column-major, so we pass transposed arguments:
//   C^T[N,M] += alpha * B^T[N,K] * A^T[K,M]
// which in column-major is:
//   cublasDgemm(handle, NoTrans, NoTrans, N, M, K, alpha, B, N, A, K, 1, C, N)
// ---------------------------------------------------------------------------
static void gpu_dgemm(int M, int N, int K,
                      double alpha,
                      const double* d_A,
                      const double* d_B,
                      double        beta,
                      double*       d_C) {
    CUBLAS_CHECK(cublasDgemm(s_handle->handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K,
                             &alpha,
                             d_B, N,
                             d_A, K,
                             &beta,
                             d_C, N));
}

// ---------------------------------------------------------------------------
// (1/2) sum_{kl}  W_oooo[k,l,i,j] * T2[k,l,a,b]  →  R[i,j,a,b]
//
// Reshape:  W  as [oo, oo]   (first two = kl, last two = ij)
//           T2 as [oo, vv]
//           R  as [oo, vv]
//
// W is stored [k,l,i,j] so the (kl) stride is correct only if we see W as
// W^T[ij, kl].  We want R[ij,ab] += alpha * W^T[ij,kl] * T2[kl,ab]
// which is a standard DGEMM(M=oo, N=vv, K=oo).
// ---------------------------------------------------------------------------
void gpu_contract_klij_klab(const Tensor4& W, const Tensor4& T2,
                             Tensor4& R, double alpha) {
    const int o  = static_cast<int>(T2.n0);
    const int v  = static_cast<int>(T2.n2);
    const int oo = o * o, vv = v * v;

    cuda_utils::DeviceBuffer<double> d_W(oo * oo);
    cuda_utils::DeviceBuffer<double> d_T2(oo * vv);
    cuda_utils::DeviceBuffer<double> d_R(oo * vv);

    d_W.upload(W.ptr(),  oo * oo);
    d_T2.upload(T2.ptr(), oo * vv);
    d_R.upload(R.ptr(),  oo * vv);

    // W stored as [kl, ij]; need W^T[ij, kl] * T2[kl, vv]
    // Use cuBLAS: C[oo_ij, vv] = alpha * W^T[oo_ij, oo_kl] * T2[oo_kl, vv]
    CUBLAS_CHECK(cublasDgemm(s_handle->handle,
                             CUBLAS_OP_N,  CUBLAS_OP_T,  // B^T then A^T in col-major
                             vv, oo, oo,
                             &alpha,
                             d_T2.ptr, vv,
                             d_W.ptr,  oo,
                             &alpha,   // beta=alpha here since we accumulate
                             d_R.ptr,  vv));

    d_R.download(R.ptr(), oo * vv);
}

// ---------------------------------------------------------------------------
// (1/2) sum_{cd}  W_vvvv[a,b,c,d] * T2[i,j,c,d]  →  R[i,j,a,b]
//
// Reshape:  W  as [vv_ab, vv_cd]
//           T2 as [oo,    vv_cd]
//           R  as [oo,    vv_ab]
//
// R[oo, vv_ab] += alpha * T2[oo, vv_cd] * W^T[vv_cd, vv_ab]
// ---------------------------------------------------------------------------
void gpu_contract_abcd_ijcd(const Tensor4& W, const Tensor4& T2,
                             Tensor4& R, double alpha) {
    const int o  = static_cast<int>(T2.n0);
    const int v  = static_cast<int>(T2.n2);
    const int oo = o * o, vv = v * v;

    cuda_utils::DeviceBuffer<double> d_W(vv * vv);
    cuda_utils::DeviceBuffer<double> d_T2(oo * vv);
    cuda_utils::DeviceBuffer<double> d_R(oo * vv);

    d_W.upload(W.ptr(),   vv * vv);
    d_T2.upload(T2.ptr(), oo * vv);
    d_R.upload(R.ptr(),   oo * vv);

    // R[oo, vv_ab] += alpha * T2[oo, vv_cd] * W^T[vv_cd, vv_ab]
    CUBLAS_CHECK(cublasDgemm(s_handle->handle,
                             CUBLAS_OP_T,  CUBLAS_OP_N,
                             vv, oo, vv,
                             &alpha,
                             d_W.ptr,  vv,
                             d_T2.ptr, vv,
                             &alpha,
                             d_R.ptr,  vv));

    d_R.download(R.ptr(), oo * vv);
}

// ---------------------------------------------------------------------------
// sum_{kc}  W_ovvo[k,b,c,j] * T2[i,k,a,c]  →  R[i,j,a,b]
//
// This contraction has mixed index ordering that does not map to a single
// DGEMM.  We reformulate by looping over (a,b) pairs and calling DGEMM on
// the (i,j) x (k,c) inner block.  For production code, a tensor transpose
// can make this a single batched GEMM.
// ---------------------------------------------------------------------------
void gpu_contract_kbcj_ikac(const Tensor4& W, const Tensor4& T2,
                             Tensor4& R, double alpha) {
    // Fall back to CPU reference for now; the index permutation needed
    // to express this as a DGEMM requires a prior tensor transpose pass.
    // TODO: add cuTENSOR / batched DGEMM implementation.
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

}  // namespace cupyccx
