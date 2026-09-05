//
// GPU tensor contractions for cupyccx
//
// Static W integral tensors (vvvv, oooo) are pre-uploaded once per molecule
// via gpu_upload_integrals() and reused across all iterations.  Benefits
// DCD/pCCD where vvvv/oooo are bare integrals that never change between
// iterations.  CCD passes dressed W tensors (rebuilt each iteration from T2)
// which are detected by host pointer mismatch and re-uploaded as before.
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

// Dedicated per-integral device buffers
static cuda_utils::DeviceBuffer<double> s_d_W_vvvv;   // W[a,b,c,d]
static cuda_utils::DeviceBuffer<double> s_d_W_oooo;   // W[k,l,i,j]
static cuda_utils::DeviceBuffer<double> s_d_T2;
static cuda_utils::DeviceBuffer<double> s_d_R;

// Buffers for contract_oovv_t2_t2 (CCD Q_B GPU path)
static cuda_utils::DeviceBuffer<double> s_d_oovv;     // <ij||ab> occ-occ-vir-vir
static cuda_utils::DeviceBuffer<double> s_d_M_vvvv;   // M[ab,cd] intermediate

// Persistent buffers for ring contraction (kbcj_ikac)
static cuda_utils::DeviceBuffer<double> s_d_W_ovvo;
static cuda_utils::DeviceBuffer<double> s_d_W_T;
static cuda_utils::DeviceBuffer<double> s_d_T2_ov;
static cuda_utils::DeviceBuffer<double> s_d_T2_T;
static cuda_utils::DeviceBuffer<double> s_d_R_ring;

// Cache invalidation: host pointer of the last uploaded bare W_vvvv/W_oooo/ovvo.
// If the W passed to a contraction matches, the upload is skipped.
static const double* s_cached_vvvv_host = nullptr;
static const double* s_cached_oooo_host = nullptr;
static const double* s_cached_ovvo_host = nullptr;
// ERI pointer used as the molecule/basis cache key.
static const void*   s_eri_cache_key    = nullptr;
// Separate cache key for the CCD-only bare oovv upload (gpu_upload_oovv).
static const void*   s_oovv_cache_key   = nullptr;

// Device-resident T2/residual buffers for gpu_begin_residual/gpu_end_residual.
// While s_session_active is true, the gpu_contract_* functions accumulate
// directly into these buffers instead of round-tripping T2/R through the
// host on every call.
static cuda_utils::DeviceBuffer<double> s_d_T2_persist;
static cuda_utils::DeviceBuffer<double> s_d_R_persist;
static bool                             s_session_active = false;

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
    s_d_W_vvvv = cuda_utils::DeviceBuffer<double>();
    s_d_W_oooo = cuda_utils::DeviceBuffer<double>();
    s_d_T2     = cuda_utils::DeviceBuffer<double>();
    s_d_R      = cuda_utils::DeviceBuffer<double>();
    s_d_oovv   = cuda_utils::DeviceBuffer<double>();
    s_d_M_vvvv = cuda_utils::DeviceBuffer<double>();
    s_d_W_ovvo = cuda_utils::DeviceBuffer<double>();
    s_d_W_T    = cuda_utils::DeviceBuffer<double>();
    s_d_T2_ov  = cuda_utils::DeviceBuffer<double>();
    s_d_T2_T   = cuda_utils::DeviceBuffer<double>();
    s_d_R_ring = cuda_utils::DeviceBuffer<double>();
    s_d_T2_persist = cuda_utils::DeviceBuffer<double>();
    s_d_R_persist  = cuda_utils::DeviceBuffer<double>();
    s_session_active   = false;
    s_cached_vvvv_host = nullptr;
    s_cached_oooo_host = nullptr;
    s_cached_ovvo_host = nullptr;
    s_eri_cache_key    = nullptr;
    s_oovv_cache_key   = nullptr;
    delete s_handle;
    s_handle = nullptr;
    s_active  = false;
}

}  // extern "C"

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------

__global__ void k_permute4d(const double* __restrict__ src,
                             double*       __restrict__ dst,
                             int d0, int d1, int d2, int d3,
                             int p0, int p1, int p2, int p3)
{
    int n   = d0 * d1 * d2 * d3;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int src_dims[4] = {d0, d1, d2, d3};
    int coords[4];
    int tmp = idx;
    for (int ax = 3; ax >= 0; --ax) {
        coords[ax] = tmp % src_dims[ax];
        tmp /= src_dims[ax];
    }

    int dst_dims[4]   = {src_dims[p0], src_dims[p1], src_dims[p2], src_dims[p3]};
    int dst_coords[4] = {coords[p0],   coords[p1],   coords[p2],   coords[p3]};

    int dst_idx = ((dst_coords[0] * dst_dims[1] + dst_coords[1])
                               * dst_dims[2] + dst_coords[2])
                               * dst_dims[3] + dst_coords[3];
    dst[dst_idx] = src[idx];
}

__global__ void k_scatter_0213(const double* __restrict__ R_tilde,
                                double*       __restrict__ R,
                                int o, int v)
{
    int n   = o * o * v * v;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int b = idx % v;           int tmp = idx / v;
    int a = tmp % v;               tmp /= v;
    int j = tmp % o;
    int i = tmp / o;

    int tilde_idx = ((i * v + a) * o + j) * v + b;
    R[idx] += R_tilde[tilde_idx];
}

// Full P(ij)P(ab) antisymmetrization of R_tilde[i,a,j,b].
// R[i,j,a,b] += R_tilde[i,a,j,b] - R_tilde[i,b,j,a]
//             - R_tilde[j,a,i,b] + R_tilde[j,b,i,a]
__global__ void k_scatter_Pijab(const double* __restrict__ R_tilde,
                                 double*       __restrict__ R,
                                 int o, int v)
{
    int n   = o * o * v * v;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int b = idx % v;           int tmp = idx / v;
    int a = tmp % v;               tmp /= v;
    int j = tmp % o;
    int i = tmp / o;

    R[idx] += R_tilde[((i * v + a) * o + j) * v + b]
            - R_tilde[((i * v + b) * o + j) * v + a]
            - R_tilde[((j * v + a) * o + i) * v + b]
            + R_tilde[((j * v + b) * o + i) * v + a];
}

namespace tensor_ops {

// ---------------------------------------------------------------------------
// Pre-upload static bare W tensors before the iteration loop.
// cache_key should be scf.eri_antisym.ptr() — unique per molecule/basis.
// ---------------------------------------------------------------------------
void gpu_upload_integrals(const Tensor4& vvvv,
                          const Tensor4& oooo,
                          const Tensor4& ovvo,
                          const void*    cache_key) {
    // Always record the current host pointers so within-iteration checks work
    // correctly even when compute() allocates fresh Tensor4 objects each call.
    s_cached_vvvv_host = vvvv.ptr();
    s_cached_oooo_host = oooo.ptr();
    s_cached_ovvo_host = ovvo.ptr();

    if (cache_key == s_eri_cache_key) return;  // same molecule — data unchanged
    s_eri_cache_key = cache_key;

    const std::size_t vvvv_n = vvvv.n0 * vvvv.n1 * vvvv.n2 * vvvv.n3;
    ensure(s_d_W_vvvv, vvvv_n);
    s_d_W_vvvv.upload(vvvv.ptr(), vvvv_n);

    const std::size_t oooo_n = oooo.n0 * oooo.n1 * oooo.n2 * oooo.n3;
    ensure(s_d_W_oooo, oooo_n);
    s_d_W_oooo.upload(oooo.ptr(), oooo_n);

    const std::size_t ovvo_n = ovvo.n0 * ovvo.n1 * ovvo.n2 * ovvo.n3;
    ensure(s_d_W_ovvo, ovvo_n);
    s_d_W_ovvo.upload(ovvo.ptr(), ovvo_n);
}

// ---------------------------------------------------------------------------
// Pre-upload the bare oovv integral (CCD-only Q_B/Q_D GPU path). Same
// cache_key convention as gpu_upload_integrals — call once before the
// iteration loop so contract_oovv_t2_t2/contract_qD_Pijab never re-upload it.
// ---------------------------------------------------------------------------
void gpu_upload_oovv(const Tensor4& oovv, const void* cache_key) {
    if (cache_key == s_oovv_cache_key) return;
    s_oovv_cache_key = cache_key;

    const std::size_t n = oovv.n0 * oovv.n1 * oovv.n2 * oovv.n3;
    ensure(s_d_oovv, n);
    s_d_oovv.upload(oovv.ptr(), n);
}

// ---------------------------------------------------------------------------
// Begin/end a device-resident accumulation session for one residual build.
// See declaration in tensor_ops.hpp for the full contract.
// ---------------------------------------------------------------------------
void gpu_begin_residual(const Tensor4& t2, const Tensor4& R) {
    const std::size_t n = t2.n0 * t2.n1 * t2.n2 * t2.n3;
    ensure(s_d_T2_persist, n);
    ensure(s_d_R_persist,  n);
    s_d_T2_persist.upload(t2.ptr(), n);
    s_d_R_persist.upload(R.ptr(),   n);
    s_session_active = true;
}

void gpu_end_residual(Tensor4& R) {
    const std::size_t n = R.n0 * R.n1 * R.n2 * R.n3;
    s_d_R_persist.download(R.ptr(), n);
    s_session_active = false;
}

// ---------------------------------------------------------------------------
// sum_{kl}  W_oooo[k,l,i,j] * T2[k,l,a,b]  →  R[i,j,a,b]
// ---------------------------------------------------------------------------
void gpu_contract_klij_klab(const Tensor4& W, const Tensor4& T2,
                             Tensor4& R, double alpha) {
    const int o  = static_cast<int>(T2.n0);
    const int v  = static_cast<int>(T2.n2);
    const int oo = o * o, vv = v * v;

    // Use pre-uploaded W_oooo if available; otherwise upload now.
    if (W.ptr() != s_cached_oooo_host) {
        ensure(s_d_W_oooo, oo * oo);
        s_d_W_oooo.upload(W.ptr(), oo * oo);
    }

    // Within a gpu_begin_residual/gpu_end_residual session, T2/R already live
    // on the device — reuse them instead of re-uploading/downloading here.
    double* t2_ptr;
    double* r_ptr;
    if (s_session_active) {
        t2_ptr = s_d_T2_persist.ptr;
        r_ptr  = s_d_R_persist.ptr;
    } else {
        ensure(s_d_T2, oo * vv);
        ensure(s_d_R,  oo * vv);
        s_d_T2.upload(T2.ptr(), oo * vv);
        s_d_R.upload(R.ptr(),   oo * vv);
        t2_ptr = s_d_T2.ptr;
        r_ptr  = s_d_R.ptr;
    }

    const double one = 1.0;
    CUBLAS_CHECK(cublasDgemm(s_handle->handle,
                             CUBLAS_OP_N, CUBLAS_OP_T,
                             vv, oo, oo,
                             &alpha,
                             t2_ptr,        vv,
                             s_d_W_oooo.ptr, oo,
                             &one,
                             r_ptr,         vv));

    if (!s_session_active)
        s_d_R.download(R.ptr(), oo * vv);
}

// ---------------------------------------------------------------------------
// sum_{cd}  W_vvvv[a,b,c,d] * T2[i,j,c,d]  →  R[i,j,a,b]
// ---------------------------------------------------------------------------
void gpu_contract_abcd_ijcd(const Tensor4& W, const Tensor4& T2,
                             Tensor4& R, double alpha) {
    const int o  = static_cast<int>(T2.n0);
    const int v  = static_cast<int>(T2.n2);
    const int oo = o * o, vv = v * v;

    // Use pre-uploaded W_vvvv if available; otherwise upload now.
    if (W.ptr() != s_cached_vvvv_host) {
        ensure(s_d_W_vvvv, vv * vv);
        s_d_W_vvvv.upload(W.ptr(), vv * vv);
    }

    // Within a gpu_begin_residual/gpu_end_residual session, T2/R already live
    // on the device — reuse them instead of re-uploading/downloading here.
    double* t2_ptr;
    double* r_ptr;
    if (s_session_active) {
        t2_ptr = s_d_T2_persist.ptr;
        r_ptr  = s_d_R_persist.ptr;
    } else {
        ensure(s_d_T2, oo * vv);
        ensure(s_d_R,  oo * vv);
        s_d_T2.upload(T2.ptr(), oo * vv);
        s_d_R.upload(R.ptr(),   oo * vv);
        t2_ptr = s_d_T2.ptr;
        r_ptr  = s_d_R.ptr;
    }

    const double one = 1.0;
    CUBLAS_CHECK(cublasDgemm(s_handle->handle,
                             CUBLAS_OP_T, CUBLAS_OP_N,
                             vv, oo, vv,
                             &alpha,
                             s_d_W_vvvv.ptr, vv,
                             t2_ptr,         vv,
                             &one,
                             r_ptr,          vv));

    if (!s_session_active)
        s_d_R.download(R.ptr(), oo * vv);
}

// ---------------------------------------------------------------------------
// sum_{kc}  W_ovvo[k,b,c,j] * T2[i,k,a,c]  →  R[i,j,a,b]
//
// Implemented via two index permutations and a single ov×ov DGEMM:
//   T2_T[i,a,k,c]  = permute(T2[i,k,a,c], 0,2,1,3)
//   W_T [k,c,j,b]  = permute(W [k,b,c,j], 0,2,3,1)
//   R_tilde[(ia),(jb)] = alpha * T2_T * W_T
//   R[i,j,a,b] += R_tilde[i,a,j,b]
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

    k_permute4d<<<blocks, threads>>>(s_d_T2_ov.ptr, s_d_T2_T.ptr,
                                     o, o, v, v,
                                     0, 2, 1, 3);
    k_permute4d<<<blocks, threads>>>(s_d_W_ovvo.ptr, s_d_W_T.ptr,
                                     o, v, v, o,
                                     0, 2, 3, 1);

    const double zero = 0.0, one = 1.0;
    CUBLAS_CHECK(cublasDgemm(s_handle->handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             ov, ov, ov,
                             &alpha,
                             s_d_W_T.ptr,  ov,
                             s_d_T2_T.ptr, ov,
                             &zero,
                             s_d_R_ring.ptr, ov));

    k_scatter_0213<<<blocks, threads>>>(s_d_R_ring.ptr, s_d_R.ptr, o, v);

    s_d_R.download(R.ptr(), oovv);
}

// ---------------------------------------------------------------------------
// Full P(ij)P(ab) ring: R[i,j,a,b] += alpha * P(ij)P(ab) sum_{kc} W[k,b,c,j]*T2[i,k,a,c]
//
// Same permute+DGEMM as gpu_contract_kbcj_ikac, but the scatter applies all four
// antisymmetry terms at once (k_scatter_Pijab) instead of just the base term.
// ---------------------------------------------------------------------------
void gpu_contract_kbcj_ikac_Pijab(const Tensor4& W, const Tensor4& T2,
                                    Tensor4& R, double alpha) {
    const int o  = static_cast<int>(T2.n0);
    const int v  = static_cast<int>(T2.n2);
    const int ov = o * v;
    const int oo = o * o, vv = v * v, oovv = oo * vv;

    // Use pre-uploaded W_ovvo if available; otherwise upload now.
    if (W.ptr() != s_cached_ovvo_host) {
        ensure(s_d_W_ovvo, oovv);
        s_d_W_ovvo.upload(W.ptr(), oovv);
    }
    ensure(s_d_W_T,    oovv);
    ensure(s_d_T2_T,   oovv);
    ensure(s_d_R_ring, oovv);

    // Within a gpu_begin_residual/gpu_end_residual session, T2/R already live
    // on the device — reuse them instead of re-uploading/downloading here.
    double* t2_ptr;
    double* r_ptr;
    if (s_session_active) {
        t2_ptr = s_d_T2_persist.ptr;
        r_ptr  = s_d_R_persist.ptr;
    } else {
        ensure(s_d_T2_ov, oovv);
        ensure(s_d_R,     oovv);
        s_d_T2_ov.upload(T2.ptr(), oovv);
        s_d_R.upload(R.ptr(),      oovv);
        t2_ptr = s_d_T2_ov.ptr;
        r_ptr  = s_d_R.ptr;
    }

    const int threads = 256;
    const int blocks  = (oovv + threads - 1) / threads;

    k_permute4d<<<blocks, threads>>>(t2_ptr, s_d_T2_T.ptr,
                                     o, o, v, v,
                                     0, 2, 1, 3);
    k_permute4d<<<blocks, threads>>>(s_d_W_ovvo.ptr, s_d_W_T.ptr,
                                     o, v, v, o,
                                     0, 2, 3, 1);

    const double zero = 0.0;
    CUBLAS_CHECK(cublasDgemm(s_handle->handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             ov, ov, ov,
                             &alpha,
                             s_d_W_T.ptr,  ov,
                             s_d_T2_T.ptr, ov,
                             &zero,
                             s_d_R_ring.ptr, ov));

    k_scatter_Pijab<<<blocks, threads>>>(s_d_R_ring.ptr, r_ptr, o, v);

    if (!s_session_active)
        s_d_R.download(R.ptr(), oovv);
}

// ---------------------------------------------------------------------------
// CCD Q_B correction: R[i,j,a,b] += alpha * sum_{kl,cd} oovv[k,l,c,d] * t2[k,l,a,b] * t2[i,j,c,d]
//
// Computed as two DGEMMs:
//   Step 1  M[ab,cd]  = sum_{kl} t2[kl,ab] * oovv[kl,cd]  (oovv^T × T2^T in col-major)
//   Step 2  R[ij,ab] += alpha * sum_{cd} M[ab,cd] * t2[ij,cd]  (reuses contract_abcd_ijcd DGEMM)
//
// Replaces the CPU build_W_vvvv + build_W_oooo T2 corrections on the GPU path.
// Call with alpha=0.25 to account for both the vvvv and oooo W corrections (each 1/8).
// ---------------------------------------------------------------------------
void gpu_contract_oovv_t2_t2(const Tensor4& oovv, const Tensor4& T2,
                               Tensor4& R, double alpha) {
    // oovv is pre-uploaded once per molecule via gpu_upload_oovv() — always
    // called before the CCD GPU iteration loop, so s_d_oovv is already
    // populated here; no per-call upload needed.
    (void)oovv;
    const int o  = static_cast<int>(T2.n0);
    const int v  = static_cast<int>(T2.n2);
    const int oo = o * o, vv = v * v, oovv_n = oo * vv;

    ensure(s_d_M_vvvv, (std::size_t)vv * vv);

    // Within a gpu_begin_residual/gpu_end_residual session, T2/R already live
    // on the device — reuse them instead of re-uploading/downloading here.
    double* t2_ptr;
    double* r_ptr;
    if (s_session_active) {
        t2_ptr = s_d_T2_persist.ptr;
        r_ptr  = s_d_R_persist.ptr;
    } else {
        ensure(s_d_T2, oovv_n);
        ensure(s_d_R,  oovv_n);
        s_d_T2.upload(T2.ptr(), oovv_n);
        s_d_R.upload(R.ptr(),   oovv_n);
        t2_ptr = s_d_T2.ptr;
        r_ptr  = s_d_R.ptr;
    }

    const double zero = 0.0, one = 1.0;

    // Step 1: M^T[ab,cd] = oovv_col(vv×oo) × T2_col^T(oo×vv)
    // Result in M_ptr (row-major vv×vv) = M[ab,cd] = sum_{kl} t2[kl,ab]*oovv[kl,cd] ✓
    CUBLAS_CHECK(cublasDgemm(s_handle->handle,
                             CUBLAS_OP_N, CUBLAS_OP_T,
                             vv, vv, oo,
                             &one,
                             s_d_oovv.ptr, vv,
                             t2_ptr,       vv,
                             &zero,
                             s_d_M_vvvv.ptr, vv));

    // Step 2: R[ij,ab] += alpha * sum_{cd} M[ab,cd] * T2[ij,cd]  (same DGEMM as contract_abcd_ijcd)
    CUBLAS_CHECK(cublasDgemm(s_handle->handle,
                             CUBLAS_OP_T, CUBLAS_OP_N,
                             vv, oo, vv,
                             &alpha,
                             s_d_M_vvvv.ptr, vv,
                             t2_ptr,         vv,
                             &one,
                             r_ptr,          vv));

    if (!s_session_active)
        s_d_R.download(R.ptr(), oovv_n);
}

// ---------------------------------------------------------------------------
// Q_D quadratic ring-ring:
//   R[i,j,a,b] += alpha * P(ij)P(ab) * sum_{k,l,c,d} oovv[k,l,c,d] * t2[i,k,a,c] * t2[j,l,b,d]
//
// Step 1  permute T2[i,k,a,c] → T2_perm[i,a,k,c]  and  oovv[k,l,c,d] → oovv_perm[k,c,l,d]
//         both as (ov, ov) matrices.
//
// Step 2  X[(ia),(ld)] = T2_perm[(ia),(kc)] @ oovv_perm[(kc),(ld)]   DGEMM ov×ov×ov
//
// Step 3  Z[(ia),(jb)] = X[(ia),(ld)] @ T2_perm^T[(ld),(jb)]          DGEMM ov×ov×ov
//         (the alpha factor is folded into this DGEMM)
//
// Step 4  R[i,j,a,b] += P(ij)P(ab) scatter of Z  (k_scatter_Pijab)
// ---------------------------------------------------------------------------
void gpu_contract_qD_Pijab(const Tensor4& oovv, const Tensor4& t2,
                             Tensor4& R, double alpha) {
    // oovv is pre-uploaded once per molecule via gpu_upload_oovv() — always
    // called before the CCD GPU iteration loop, so s_d_oovv is already
    // populated here; no per-call upload needed.
    (void)oovv;
    const int o      = static_cast<int>(t2.n0);
    const int v      = static_cast<int>(t2.n2);
    const int ov     = o * v;
    const int oovv_n = ov * ov;

    ensure(s_d_T2_T,   oovv_n);   // T2_perm [i,a,k,c]
    ensure(s_d_W_T,    oovv_n);   // oovv_perm [k,c,l,d]
    ensure(s_d_R_ring, oovv_n);   // X intermediate [(ia),(ld)]
    ensure(s_d_W_ovvo, oovv_n);   // Z intermediate [(ia),(jb)]

    // Within a gpu_begin_residual/gpu_end_residual session, T2/R already live
    // on the device — reuse them instead of re-uploading/downloading here.
    double* t2_ptr;
    double* r_ptr;
    if (s_session_active) {
        t2_ptr = s_d_T2_persist.ptr;
        r_ptr  = s_d_R_persist.ptr;
    } else {
        ensure(s_d_T2_ov, oovv_n);
        ensure(s_d_R,     oovv_n);
        s_d_T2_ov.upload(t2.ptr(), oovv_n);
        s_d_R.upload(R.ptr(),      oovv_n);
        t2_ptr = s_d_T2_ov.ptr;
        r_ptr  = s_d_R.ptr;
    }

    const int threads = 256;
    const int blocks  = (oovv_n + threads - 1) / threads;

    // T2_perm[i,a,k,c] ← t2[i,k,a,c]: permute (o,o,v,v) with perm 0,2,1,3
    k_permute4d<<<blocks, threads>>>(t2_ptr, s_d_T2_T.ptr,
                                     o, o, v, v, 0, 2, 1, 3);

    // oovv_perm[k,c,l,d] ← oovv[k,l,c,d]: permute (o,o,v,v) with perm 0,2,1,3
    k_permute4d<<<blocks, threads>>>(s_d_oovv.ptr, s_d_W_T.ptr,
                                     o, o, v, v, 0, 2, 1, 3);

    const double zero = 0.0, one = 1.0;

    // Step 2: X[(ia),(ld)] = T2_perm[(ia),(kc)] @ oovv_perm[(kc),(ld)]
    // Row-major A @ B pattern: cublasDgemm(N,N, M, N, K, alpha, B, K, A, K, beta, C, N)
    CUBLAS_CHECK(cublasDgemm(s_handle->handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             ov, ov, ov,
                             &one,
                             s_d_W_T.ptr,   ov,   // B = oovv_perm
                             s_d_T2_T.ptr,  ov,   // A = T2_perm
                             &zero,
                             s_d_R_ring.ptr, ov)); // C = X

    // Step 3: Z[(ia),(jb)] = X[(ia),(ld)] @ T2_perm^T[(ld),(jb)]   (alpha applied here)
    // Row-major A @ B^T pattern: cublasDgemm(T,N, M, N, K, alpha, B, K, A, K, beta, C, N)
    // where the "B^T" matrix (T2_perm) appears first in the call with op=T.
    CUBLAS_CHECK(cublasDgemm(s_handle->handle,
                             CUBLAS_OP_T, CUBLAS_OP_N,
                             ov, ov, ov,
                             &alpha,
                             s_d_T2_T.ptr,  ov,    // B = T2_perm (op=T → transpose it)
                             s_d_R_ring.ptr, ov,   // A = X
                             &zero,
                             s_d_W_ovvo.ptr, ov)); // C = Z

    // Step 4: R[i,j,a,b] += P(ij)P(ab) Z[i,a,j,b]
    k_scatter_Pijab<<<blocks, threads>>>(s_d_W_ovvo.ptr, r_ptr, o, v);

    if (!s_session_active)
        s_d_R.download(R.ptr(), oovv_n);
}

}  // namespace tensor_ops
}  // namespace cupyccx
