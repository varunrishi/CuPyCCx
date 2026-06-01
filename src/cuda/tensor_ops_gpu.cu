//
// GPU tensor contractions for cupyccx
//
// Persistent device buffers are allocated on first use and reused across all
// contraction calls and iterations; freed in gpu_backend_finalize().
//
// Ring contraction (kbcj_ikac) is implemented via two index permutations and
// a single ov×ov DGEMM rather than a CPU fallback loop.
//

#include "cuda_utils.cuh"
#include "cupyccx/types.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace cupyccx {

// ---------------------------------------------------------------------------
// Module-level state
// ---------------------------------------------------------------------------
static cuda_utils::CublasHandle*        s_handle  = nullptr;
static bool                             s_active  = false;

// Persistent buffers for ladder contractions (klij_klab, abcd_ijcd)
static cuda_utils::DeviceBuffer<double> s_d_W;
static cuda_utils::DeviceBuffer<double> s_d_T2;
static cuda_utils::DeviceBuffer<double> s_d_R;

// Persistent buffers for the ring contraction (kbcj_ikac)
static cuda_utils::DeviceBuffer<double> s_d_W_ovvo;  // W[k,b,c,j]  uploaded
static cuda_utils::DeviceBuffer<double> s_d_W_T;     // W_T[k,c,j,b] after permute(0,2,3,1)
static cuda_utils::DeviceBuffer<double> s_d_T2_ov;   // T2[i,k,a,c]  uploaded
static cuda_utils::DeviceBuffer<double> s_d_T2_T;    // T2_T[i,a,k,c] after permute(0,2,1,3)
static cuda_utils::DeviceBuffer<double> s_d_R_ring;  // R_tilde[i,a,j,b] DGEMM result

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
    s_d_W      = cuda_utils::DeviceBuffer<double>();
    s_d_T2     = cuda_utils::DeviceBuffer<double>();
    s_d_R      = cuda_utils::DeviceBuffer<double>();
    s_d_W_ovvo = cuda_utils::DeviceBuffer<double>();
    s_d_W_T    = cuda_utils::DeviceBuffer<double>();
    s_d_T2_ov  = cuda_utils::DeviceBuffer<double>();
    s_d_T2_T   = cuda_utils::DeviceBuffer<double>();
    s_d_R_ring = cuda_utils::DeviceBuffer<double>();
    delete s_handle;
    s_handle = nullptr;
    s_active  = false;
}

}  // extern "C"

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------

// Generic 4D permutation kernel.
// Reads src[d0,d1,d2,d3] (row-major) and writes to dst permuted as perm[].
// perm[i] = which source axis feeds destination axis i.
__global__ void k_permute4d(const double* __restrict__ src,
                             double*       __restrict__ dst,
                             int d0, int d1, int d2, int d3,
                             int p0, int p1, int p2, int p3)
{
    int n   = d0 * d1 * d2 * d3;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Unpack linear index in src layout [d0,d1,d2,d3]
    int src_dims[4] = {d0, d1, d2, d3};
    int coords[4];
    int tmp = idx;
    for (int ax = 3; ax >= 0; --ax) {
        coords[ax] = tmp % src_dims[ax];
        tmp /= src_dims[ax];
    }

    // Permuted destination dimensions
    int dst_dims[4]   = {src_dims[p0], src_dims[p1], src_dims[p2], src_dims[p3]};
    int dst_coords[4] = {coords[p0],   coords[p1],   coords[p2],   coords[p3]};

    int dst_idx = ((dst_coords[0] * dst_dims[1] + dst_coords[1])
                               * dst_dims[2] + dst_coords[2])
                               * dst_dims[3] + dst_coords[3];
    dst[dst_idx] = src[idx];
}

// Scatter R_tilde[i,a,j,b] (ov×ov layout) into R[i,j,a,b] (oo×vv layout).
// R[i,j,a,b] += R_tilde[i,a,j,b]
__global__ void k_scatter_0213(const double* __restrict__ R_tilde,
                                double*       __restrict__ R,
                                int o, int v)
{
    int n   = o * o * v * v;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Unpack idx as R[i,j,a,b] in row-major [o,o,v,v]
    int b = idx % v;           int tmp = idx / v;
    int a = tmp % v;               tmp /= v;
    int j = tmp % o;
    int i = tmp / o;

    // R_tilde stored as [i,a,j,b] in row-major [o,v,o,v]
    int tilde_idx = ((i * v + a) * o + j) * v + b;
    R[idx] += R_tilde[tilde_idx];
}

namespace tensor_ops {

// ---------------------------------------------------------------------------
// sum_{kl}  W_oooo[k,l,i,j] * T2[k,l,a,b]  →  R[i,j,a,b]
//
// R[ij,ab] += alpha * W^T[ij,kl] * T2[kl,ab]   (DGEMM M=oo, N=vv, K=oo)
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
// sum_{cd}  W_vvvv[a,b,c,d] * T2[i,j,c,d]  →  R[i,j,a,b]
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
// Reformulated as a single ov×ov DGEMM via two index permutations:
//
//   T2_T[i,a,k,c]  = permute(T2[i,k,a,c], 0,2,1,3)   → [(ia),(kc)]  ov×ov
//   W_T [k,c,j,b]  = permute(W [k,b,c,j], 0,2,3,1)   → [(kc),(jb)]  ov×ov
//   R_tilde[(ia),(jb)] = alpha * T2_T * W_T            → ov×ov DGEMM
//   R[i,j,a,b] += R_tilde[i,a,j,b]                    → scatter kernel
// ---------------------------------------------------------------------------
void gpu_contract_kbcj_ikac(const Tensor4& W, const Tensor4& T2,
                             Tensor4& R, double alpha) {
    const int o  = static_cast<int>(T2.n0);
    const int v  = static_cast<int>(T2.n2);
    const int ov = o * v;
    const int oo = o * o, vv = v * v, oovv = oo * vv;

    ensure(s_d_W_ovvo, oovv);
    ensure(s_d_W_T,    oovv);
    ensure(s_d_T2_ov,  oovv);
    ensure(s_d_T2_T,   oovv);
    ensure(s_d_R_ring, oovv);
    ensure(s_d_R,      oovv);

    s_d_W_ovvo.upload(W.ptr(),   oovv);
    s_d_T2_ov.upload(T2.ptr(),   oovv);
    s_d_R.upload(R.ptr(),        oovv);

    const int threads = 256;
    const int blocks  = (oovv + threads - 1) / threads;

    // Permute T2[i,k,a,c] → T2_T[i,a,k,c]  (perm 0,2,1,3)
    k_permute4d<<<blocks, threads>>>(s_d_T2_ov.ptr, s_d_T2_T.ptr,
                                     o, o, v, v,
                                     0, 2, 1, 3);

    // Permute W[k,b,c,j] → W_T[k,c,j,b]  (perm 0,2,3,1)
    k_permute4d<<<blocks, threads>>>(s_d_W_ovvo.ptr, s_d_W_T.ptr,
                                     o, v, v, o,
                                     0, 2, 3, 1);

    // DGEMM: R_tilde[ov,ov] = alpha * T2_T[ov,ov] * W_T[ov,ov]
    const double zero = 0.0, one = 1.0;
    CUBLAS_CHECK(cublasDgemm(s_handle->handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             ov, ov, ov,
                             &alpha,
                             s_d_W_T.ptr,  ov,
                             s_d_T2_T.ptr, ov,
                             &zero,
                             s_d_R_ring.ptr, ov));

    // Scatter: R[i,j,a,b] += R_tilde[i,a,j,b]
    k_scatter_0213<<<blocks, threads>>>(s_d_R_ring.ptr, s_d_R.ptr, o, v);

    s_d_R.download(R.ptr(), oovv);
}

}  // namespace tensor_ops
}  // namespace cupyccx
