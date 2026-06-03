#include "cupyccx/ccd.hpp"
#include "cupyccx/integrals.hpp"
#include "cupyccx/tensor_ops.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace cupyccx {

// ============================================================
//  CCD / LCCD
// ============================================================

void CCD::build_W_oooo(const Tensor4& oooo,
                        const Tensor4& oovv,
                        const Tensor4& t2,
                        Tensor4&       W_oooo) const {
    W_oooo = oooo;
    if (variant_ == "LCCD") return;  // LCCD: no T2 contribution
    const int o = scf_.n_occ, v = scf_.n_vir;
    // W_{klij} += (1/4) sum_{cd} <kl||cd> t_{ij}^{cd}
    for (int k = 0; k < o; ++k)
    for (int l = 0; l < o; ++l)
    for (int i = 0; i < o; ++i)
    for (int j = 0; j < o; ++j) {
        real_t s = 0.0;
        for (int c = 0; c < v; ++c)
        for (int d = 0; d < v; ++d)
            s += oovv(k, l, c, d) * t2(i, j, c, d);
        W_oooo(k, l, i, j) += 0.25 * s;
    }
}

void CCD::build_W_vvvv(const Tensor4& vvvv,
                        const Tensor4& oovv,
                        const Tensor4& t2,
                        Tensor4&       W_vvvv) const {
    W_vvvv = vvvv;
    if (variant_ == "LCCD") return;  // LCCD: no T2 contribution
    const int o = scf_.n_occ, v = scf_.n_vir;
    // W_{abcd} += (1/4) sum_{kl} <kl||cd> t_{kl}^{ab}
    for (int a = 0; a < v; ++a)
    for (int b = 0; b < v; ++b)
    for (int c = 0; c < v; ++c)
    for (int d = 0; d < v; ++d) {
        real_t s = 0.0;
        for (int k = 0; k < o; ++k)
        for (int l = 0; l < o; ++l)
            s += oovv(k, l, c, d) * t2(k, l, a, b);
        W_vvvv(a, b, c, d) += 0.25 * s;
    }
}

real_t CCD::build_residual(const Tensor4& t2,
                            const Tensor4& oovv,
                            const Tensor4& W_vvvv,
                            const Tensor4& W_oooo,
                            const Tensor4& ovvo,
                            const Matrix&  F_vv,
                            const Matrix&  F_oo,
                            Tensor4&       R) const {
    const int o = scf_.n_occ, v = scf_.n_vir;

    // Start from <ij||ab>
    R = oovv;

    // P(ab) f_bc t_{ij}^{ac}
    for (int i = 0; i < o; ++i)
    for (int j = 0; j < o; ++j)
    for (int a = 0; a < v; ++a)
    for (int b = 0; b < v; ++b) {
        real_t sab = 0.0, sba = 0.0;
        for (int c = 0; c < v; ++c) {
            sab += F_vv(b, c) * t2(i, j, a, c);
            sba += F_vv(a, c) * t2(i, j, b, c);
        }
        R(i, j, a, b) += sab - sba;
    }

    // -P(ij) f_kj t_{ik}^{ab}
    for (int i = 0; i < o; ++i)
    for (int j = 0; j < o; ++j)
    for (int a = 0; a < v; ++a)
    for (int b = 0; b < v; ++b) {
        real_t sij = 0.0, sji = 0.0;
        for (int k = 0; k < o; ++k) {
            sij += F_oo(k, j) * t2(i, k, a, b);
            sji += F_oo(k, i) * t2(j, k, a, b);
        }
        R(i, j, a, b) -= sij - sji;
    }

    // Q_B: These two below terms constitute B
    // (1/2) W_{klij} t_{kl}^{ab}  — W = bare <kl||ij> for LCCD
    tensor_ops::contract_klij_klab(W_oooo, t2, R, 0.5);

    // (1/2) W_{abcd} t_{ij}^{cd}  — W = bare <ab||cd> for LCCD
    tensor_ops::contract_abcd_ijcd(W_vvvv, t2, R, 0.5);

    // P(ij)P(ab) <kb||cj> t_{ik}^{ac}  (four permutation terms)
    for (int i = 0; i < o; ++i)
    for (int j = 0; j < o; ++j)
    for (int a = 0; a < v; ++a)
    for (int b = 0; b < v; ++b) {
        real_t s = 0.0;
        for (int k = 0; k < o; ++k)
        for (int c = 0; c < v; ++c) {
            s += ovvo(k, b, c, j) * t2(i, k, a, c)   // +1
               - ovvo(k, a, c, j) * t2(i, k, b, c)   // P(ab) -> -1
               - ovvo(k, b, c, i) * t2(j, k, a, c)   // P(ij) -> -1
               + ovvo(k, a, c, i) * t2(j, k, b, c);  // P(ij)P(ab) -> +1
        }
        R(i, j, a, b) += s;
    }

    // Q_D: (1/2) P(ij)P(ab) sum_{klcd} <kl||cd> t_{ik}^{ac} t_{jl}^{bd}
    // (quadratic ring-ring from dressed W_{mbej} intermediate; CCD only)
    if (variant_ != "LCCD") {
        // X(i,a,l,d) = sum_{k,c} oovv(k,l,c,d) * t2(i,k,a,c)
        Tensor4 X(o, v, o, v);
        for (int i = 0; i < o; ++i)
        for (int a = 0; a < v; ++a)
        for (int l = 0; l < o; ++l)
        for (int d = 0; d < v; ++d) {
            real_t s = 0.0;
            for (int k = 0; k < o; ++k)
            for (int c = 0; c < v; ++c)
                s += oovv(k, l, c, d) * t2(i, k, a, c);
            X(i, a, l, d) = s;
        }
        for (int i = 0; i < o; ++i)
        for (int j = 0; j < o; ++j)
        for (int a = 0; a < v; ++a)
        for (int b = 0; b < v; ++b) {
            real_t s = 0.0;
            for (int l = 0; l < o; ++l)
            for (int d = 0; d < v; ++d) {
                s += X(i, a, l, d) * t2(j, l, b, d)   // base
                   - X(j, a, l, d) * t2(i, l, b, d)   // P(ij)
                   - X(i, b, l, d) * t2(j, l, a, d)   // P(ab)
                   + X(j, b, l, d) * t2(i, l, a, d);  // P(ij)P(ab)
            }
            R(i, j, a, b) += 0.5 * s;
        }

        // Q_C: Dressed F_vv: ΔF_vv(b,e) = -(1/2) Σ_{k,l,d} <kl||ed> t_{kl}^{bd}
        // Contribution to residual: P(ab) Σ_e ΔF_vv(b,e) t_{ij}^{ae}
        {
            Matrix dF_vv(v, v);
            dF_vv.setZero();
            for (int b = 0; b < v; ++b)
            for (int e = 0; e < v; ++e) {
                real_t s = 0.0;
                for (int k = 0; k < o; ++k)
                for (int l = 0; l < o; ++l)
                for (int d = 0; d < v; ++d)
                    s += oovv(k, l, e, d) * t2(k, l, b, d);
                dF_vv(b, e) = -0.5 * s;
            }
            for (int i = 0; i < o; ++i)
            for (int j = 0; j < o; ++j)
            for (int a = 0; a < v; ++a)
            for (int b = 0; b < v; ++b) {
                real_t s = 0.0;
                for (int e = 0; e < v; ++e)
                    s += dF_vv(a, e) * t2(i, j, b, e)
                       - dF_vv(b, e) * t2(i, j, a, e);
                R(i, j, a, b) -= s;
            }
        }

        // Q_A: Dressed F_oo: ΔF_oo(k,j) = +(1/2) Σ_{l,c,d} <kl||cd> t_{jl}^{cd}
        // Contribution to residual: -P(ij) Σ_k ΔF_oo(k,j) t_{ik}^{ab}
        {
            Matrix dF_oo(o, o);
            dF_oo.setZero();
            for (int k = 0; k < o; ++k)
            for (int jj = 0; jj < o; ++jj) {
                real_t s = 0.0;
                for (int l = 0; l < o; ++l)
                for (int c = 0; c < v; ++c)
                for (int d = 0; d < v; ++d)
                    s += oovv(k, l, c, d) * t2(jj, l, c, d);
                dF_oo(k, jj) = 0.5 * s;
            }
            for (int i = 0; i < o; ++i)
            for (int j = 0; j < o; ++j)
            for (int a = 0; a < v; ++a)
            for (int b = 0; b < v; ++b) {
                real_t s = 0.0;
                for (int k = 0; k < o; ++k)
                    s += dF_oo(k, j) * t2(i, k, a, b)
                       - dF_oo(k, i) * t2(j, k, a, b);
                R(i, j, a, b) -= s;
            }
        }
    }

    real_t rms = 0.0;
    for (auto x : R.data) rms += x * x;
    return std::sqrt(rms / R.data.size());
}

CCResult CCD::compute(real_t e_scf) {
    validate_scf(scf_);
    tensor_ops::init(opts_.use_gpu);

    const int o = scf_.n_occ, v = scf_.n_vir;

    auto oovv = slice_oovv(scf_);
    auto vvvv = slice_vvvv(scf_);
    auto oooo = slice_oooo(scf_);
    auto ovvo = slice_ovvo(scf_);
    auto D2   = make_D2(scf_);

    if (tensor_ops::gpu_active())
        tensor_ops::gpu_upload_integrals(vvvv, oooo, ovvo, scf_.eri_antisym.ptr());

    Matrix F_vv = scf_.fock.block(o, o, v, v);
    Matrix F_oo = scf_.fock.block(0, 0, o, o);

    // MP2 initial guess
    Tensor4 t2(o, o, v, v);
    for (std::size_t n = 0; n < t2.data.size(); ++n)
        t2.data[n] = oovv.data[n] / D2.data[n];

    DIISState diis(opts_.diis_size);
    Tensor4 W_oooo(o, o, o, o), W_vvvv(v, v, v, v), residual(o, o, v, v);

    real_t e_prev = 0.0;
    CCResult result{};
    result.converged = false;

    for (int iter = 1; iter <= opts_.max_iter; ++iter) {
        build_W_oooo(oooo, oovv, t2, W_oooo);
        build_W_vvvv(vvvv, oovv, t2, W_vvvv);

        real_t rms = build_residual(t2, oovv, W_vvvv, W_oooo, ovvo, F_vv, F_oo, residual);

        Tensor4 new_t2(o, o, v, v);
        for (std::size_t n = 0; n < t2.data.size(); ++n)
            new_t2.data[n] = t2.data[n] - residual.data[n] / D2.data[n];

        if (opts_.use_diis) {
            Tensor4 err(o, o, v, v);
            for (std::size_t n = 0; n < t2.data.size(); ++n)
                err.data[n] = new_t2.data[n] - t2.data[n];
            diis.push(err, new_t2);
            if (iter >= 2) {
                Tensor4 extrap = diis.extrapolate();
                extrap.n0 = o; extrap.n1 = o; extrap.n2 = v; extrap.n3 = v;
                new_t2 = extrap;
            }
        }

        t2 = new_t2;
        real_t e_corr = compute_energy(t2, oovv);
        real_t de = std::abs(e_corr - e_prev);
        e_prev = e_corr;

        if (callback_) callback_(iter, e_corr, de, rms);

        if (de < opts_.conv_energy && rms < opts_.conv_amp) {
            result.converged = true;
            result.n_iter    = iter;
            result.e_corr    = e_corr;
            result.e_total   = e_scf + e_corr;
            result.t2        = t2;
            break;
        }
    }

    if (!result.converged) {
        const char* name = variant_.c_str();
        std::cerr << name << ": WARNING — did not converge in "
                  << opts_.max_iter << " iterations\n";
    }

    tensor_ops::finalize();
    return result;
}


// ============================================================
//  Shared helpers for DCD/pCCD quadratic terms
// ============================================================

// ---------------------------------------------------------------------------
// Compute Q[i,j,a,b] = sum of quadratic T2*T2 diagrams (pre-symmetrisation).
//
// After calling this, the caller adds Q[i,j,a,b] + Q[j,i,b,a] to R.
//
// Parameter → diagram mapping (P-term computes exactly the einsum shown):
//   cDc (P1): D_c = sum_{k,l,c,d} oovv_plain(k,l,c,d) * t2(i,l,a,d) * t2(k,j,c,b)
//   cC  (P2): C   = sum_{k,l,c,d} oovv(k,l,c,d)       * t2(k,l,a,d) * t2(i,j,c,b)
//   cA  (P3): A   = sum_{k,l,c,d} oovv(k,l,c,d)       * t2(i,l,c,d) * t2(k,j,a,b)
//   cDx (P4): D_x = sum_{k,l,c,d} oovv_plain(k,l,c,d) * t2(i,l,d,b) * t2(k,j,a,c)
//   cB  (P5): B   = sum_{k,l,c,d} oovv(k,l,c,d)       * t2(i,j,c,d) * t2(k,l,a,b)
//
// Verified relationship to CCD quadratics (at CCD fixed point, diff ~1e-17):
//   Q_D = (1/2)*(sym(P1_antisym) - sym(P4_antisym))
//   Q_C = -(1/2)*sym(P2)
//   Q_A = -(1/2)*sym(P3)
//   Q_B = (1/8)*sym(P5)
// where sym(X)[i,j,a,b] = X[i,j,a,b] + X[j,i,b,a].
//
// Symmetrisation note: B is self-conjugate (P5[j,i,b,a]=P5[i,j,a,b]), so
// Q+Q' doubles its contribution. All other diagrams are not self-conjugate.
//
// For DCD:   Coulomb ring (plain ERIs) + mixed ring-ladder (×1/2), no D_x, no B
//            cDc=0.5(plain), cC=-0.25, cA=-0.25, cDx=0, cB=0
// For pCCD:  pass oovv as oovv_plain so D uses antisym; at α=β=1 recovers CCD
//            cDc=0.5β, cC=-0.5β, cA=-0.25(1+α), cDx=-0.5β, cB=α/8
// ---------------------------------------------------------------------------
static void build_quad_Q(
        const Tensor4& oovv,        // <kl||cd> antisym — used for C, A, B
        const Tensor4& oovv_plain,  // <kl|cd>  plain   — used for D_c, D_x
        const Tensor4& t2, int o, int v,
        real_t cDc,  // coeff of D_c (P1)
        real_t cC,   // coeff of C   (P2)
        real_t cA,   // coeff of A   (P3)
        real_t cDx,  // coeff of D_x (P4)
        real_t cB,   // coeff of B   (P5); pass alpha/2 — B doubles under symmetrisation
        Tensor4& Q)
{
    // --- D_c via intermediate A[l,d,j,b] = sum_{k,c} oovv_plain(k,l,c,d)*t2(k,j,c,b) ---
    if (cDc != 0.0) {
        Tensor4 A(o, v, o, v);   // A[l,d,j,b]
        for (int l = 0; l < o; ++l)
        for (int d = 0; d < v; ++d)
        for (int jj = 0; jj < o; ++jj)
        for (int b = 0; b < v; ++b) {
            real_t s = 0.0;
            for (int k = 0; k < o; ++k)
            for (int c = 0; c < v; ++c)
                s += oovv_plain(k, l, c, d) * t2(k, jj, c, b);
            A(l, d, jj, b) = s;
        }
        for (int i = 0; i < o; ++i)
        for (int jj = 0; jj < o; ++jj)
        for (int a = 0; a < v; ++a)
        for (int b = 0; b < v; ++b) {
            real_t s = 0.0;
            for (int l = 0; l < o; ++l)
            for (int d = 0; d < v; ++d)
                s += t2(i, l, a, d) * A(l, d, jj, b);
            Q(i, jj, a, b) += cDc * s;
        }
    }

    // --- C via intermediate B[a,c] = sum_{k,l,d} oovv(k,l,c,d)*t2(k,l,a,d) ---
    if (cC != 0.0) {
        std::vector<real_t> B(v * v, 0.0);
        for (int a = 0; a < v; ++a)
        for (int c = 0; c < v; ++c) {
            real_t s = 0.0;
            for (int k = 0; k < o; ++k)
            for (int l = 0; l < o; ++l)
            for (int d = 0; d < v; ++d)
                s += oovv(k, l, c, d) * t2(k, l, a, d);
            B[a * v + c] = s;
        }
        for (int i = 0; i < o; ++i)
        for (int jj = 0; jj < o; ++jj)
        for (int a = 0; a < v; ++a)
        for (int b = 0; b < v; ++b) {
            real_t s = 0.0;
            for (int c = 0; c < v; ++c)
                s += B[a * v + c] * t2(i, jj, c, b);
            Q(i, jj, a, b) += cC * s;
        }
    }

    // --- A via intermediate C[k,i] = sum_{l,c,d} oovv(k,l,c,d)*t2(i,l,c,d) ---
    if (cA != 0.0) {
        std::vector<real_t> C(o * o, 0.0);
        for (int k = 0; k < o; ++k)
        for (int i = 0; i < o; ++i) {
            real_t s = 0.0;
            for (int l = 0; l < o; ++l)
            for (int c = 0; c < v; ++c)
            for (int d = 0; d < v; ++d)
                s += oovv(k, l, c, d) * t2(i, l, c, d);
            C[k * o + i] = s;
        }
        for (int i = 0; i < o; ++i)
        for (int jj = 0; jj < o; ++jj)
        for (int a = 0; a < v; ++a)
        for (int b = 0; b < v; ++b) {
            real_t s = 0.0;
            for (int k = 0; k < o; ++k)
                s += C[k * o + i] * t2(k, jj, a, b);
            Q(i, jj, a, b) += cA * s;
        }
    }

    // --- D_x via intermediate H[k,c,i,b] = sum_{l,d} oovv_plain(k,l,c,d)*t2(i,l,d,b) ---
    if (cDx != 0.0) {
        Tensor4 H(o, v, o, v);   // H[k,c,i,b]
        for (int k = 0; k < o; ++k)
        for (int c = 0; c < v; ++c)
        for (int i = 0; i < o; ++i)
        for (int b = 0; b < v; ++b) {
            real_t s = 0.0;
            for (int l = 0; l < o; ++l)
            for (int d = 0; d < v; ++d)
                s += oovv_plain(k, l, c, d) * t2(i, l, d, b);
            H(k, c, i, b) = s;
        }
        for (int i = 0; i < o; ++i)
        for (int jj = 0; jj < o; ++jj)
        for (int a = 0; a < v; ++a)
        for (int b = 0; b < v; ++b) {
            real_t s = 0.0;
            for (int k = 0; k < o; ++k)
            for (int c = 0; c < v; ++c)
                s += H(k, c, i, b) * t2(k, jj, a, c);
            Q(i, jj, a, b) += cDx * s;
        }
    }

    // --- B via intermediate M[c,d,a,b] = sum_{k,l} oovv(k,l,c,d)*t2(k,l,a,b) ---
    if (cB != 0.0) {
        Tensor4 M(v, v, v, v);   // M[c,d,a,b]
        for (int c = 0; c < v; ++c)
        for (int d = 0; d < v; ++d)
        for (int a = 0; a < v; ++a)
        for (int b = 0; b < v; ++b) {
            real_t s = 0.0;
            for (int k = 0; k < o; ++k)
            for (int l = 0; l < o; ++l)
                s += oovv(k, l, c, d) * t2(k, l, a, b);
            M(c, d, a, b) = s;
        }
        for (int i = 0; i < o; ++i)
        for (int jj = 0; jj < o; ++jj)
        for (int a = 0; a < v; ++a)
        for (int b = 0; b < v; ++b) {
            real_t s = 0.0;
            for (int c = 0; c < v; ++c)
            for (int d = 0; d < v; ++d)
                s += t2(i, jj, c, d) * M(c, d, a, b);
            Q(i, jj, a, b) += cB * s;
        }
    }
}

// ---------------------------------------------------------------------------
// Shared linear residual (Fock + bare ladder + bare ring) used by DCD & pCCD.
// Writes to R and returns RMS(R) for convergence testing.
// ---------------------------------------------------------------------------
static real_t build_linear_residual(
        const Tensor4& t2, const Tensor4& oovv,
        const Tensor4& vvvv, const Tensor4& oooo,
        const Tensor4& ovvo,
        const Matrix& F_vv, const Matrix& F_oo,
        int o, int v, Tensor4& R)
{
    R = oovv;

    // P(ab) f_bc t_{ij}^{ac}
    for (int i = 0; i < o; ++i)
    for (int j = 0; j < o; ++j)
    for (int a = 0; a < v; ++a)
    for (int b = 0; b < v; ++b) {
        real_t sab = 0.0, sba = 0.0;
        for (int c = 0; c < v; ++c) {
            sab += F_vv(b, c) * t2(i, j, a, c);
            sba += F_vv(a, c) * t2(i, j, b, c);
        }
        R(i, j, a, b) += sab - sba;
    }

    // -P(ij) f_kj t_{ik}^{ab}
    for (int i = 0; i < o; ++i)
    for (int j = 0; j < o; ++j)
    for (int a = 0; a < v; ++a)
    for (int b = 0; b < v; ++b) {
        real_t sij = 0.0, sji = 0.0;
        for (int k = 0; k < o; ++k) {
            sij += F_oo(k, j) * t2(i, k, a, b);
            sji += F_oo(k, i) * t2(j, k, a, b);
        }
        R(i, j, a, b) -= sij - sji;
    }

    // (1/2) <kl||ij> t_{kl}^{ab}  (bare hole-hole ladder)
    tensor_ops::contract_klij_klab(oooo, t2, R, 0.5);

    // (1/2) <ab||cd> t_{ij}^{cd}  (bare particle-particle ladder)
    tensor_ops::contract_abcd_ijcd(vvvv, t2, R, 0.5);

    // P(ij)P(ab) <kb||cj> t_{ik}^{ac}  (bare ring)
    for (int i = 0; i < o; ++i)
    for (int j = 0; j < o; ++j)
    for (int a = 0; a < v; ++a)
    for (int b = 0; b < v; ++b) {
        real_t s = 0.0;
        for (int k = 0; k < o; ++k)
        for (int c = 0; c < v; ++c) {
            s += ovvo(k, b, c, j) * t2(i, k, a, c)
               - ovvo(k, a, c, j) * t2(i, k, b, c)
               - ovvo(k, b, c, i) * t2(j, k, a, c)
               + ovvo(k, a, c, i) * t2(j, k, b, c);
        }
        R(i, j, a, b) += s;
    }

    real_t rms = 0.0;
    for (auto x : R.data) rms += x * x;
    return std::sqrt(rms / R.data.size());
}

// ============================================================
//  DCD
// ============================================================

real_t DCD::build_residual(const Tensor4& t2,
                            const Tensor4& oovv,
                            const Tensor4& oovv_plain,
                            const Tensor4& vvvv,
                            const Tensor4& oooo,
                            const Tensor4& ovvo,
                            const Matrix&  F_vv,
                            const Matrix&  F_oo,
                            Tensor4&       R) const {
    const int o = scf_.n_occ, v = scf_.n_vir;

    // Linear terms (Fock + bare ladders + bare ring)
    build_linear_residual(t2, oovv, vvvv, oooo, ovvo, F_vv, F_oo, o, v, R);

    // DCD quadratic terms (Kats & Manby 2013):
    // Q1: +(1/4) P(ij) sum_{klcd} <kl||cd> t_{ik}^{cd} t_{lj}^{ab}
    // Q2: +(1/4) P(ab) sum_{klcd} <kl||cd> t_{kl}^{ac} t_{ij}^{db}
    // Q3: +(1/2) P(ij)P(ab) sum_{klcd} <kl|cd> t_{ik}^{ac} t_{jl}^{bd}

    // Q1: Y1[i,l] = sum_{k,c,d} oovv[k,l,c,d]*t2[i,k,c,d]
    {
        std::vector<real_t> Y1(o * o, 0.0);
        for (int i = 0; i < o; ++i)
        for (int l = 0; l < o; ++l) {
            real_t s = 0.0;
            for (int k = 0; k < o; ++k)
            for (int c = 0; c < v; ++c)
            for (int d = 0; d < v; ++d)
                s += oovv(k, l, c, d) * t2(i, k, c, d);
            Y1[i * o + l] = s;
        }
        for (int i = 0; i < o; ++i)
        for (int j = 0; j < o; ++j)
        for (int a = 0; a < v; ++a)
        for (int b = 0; b < v; ++b) {
            real_t x1ij = 0.0, x1ji = 0.0;
            for (int l = 0; l < o; ++l) {
                x1ij += Y1[i * o + l] * t2(l, j, a, b);
                x1ji += Y1[j * o + l] * t2(l, i, a, b);
            }
            R(i, j, a, b) += 0.25 * (x1ij - x1ji);
        }
    }

    // Q2: Y2[a,d] = sum_{k,l,c} oovv[k,l,c,d]*t2[k,l,a,c]
    {
        std::vector<real_t> Y2(v * v, 0.0);
        for (int a = 0; a < v; ++a)
        for (int d = 0; d < v; ++d) {
            real_t s = 0.0;
            for (int k = 0; k < o; ++k)
            for (int l = 0; l < o; ++l)
            for (int c = 0; c < v; ++c)
                s += oovv(k, l, c, d) * t2(k, l, a, c);
            Y2[a * v + d] = s;
        }
        for (int i = 0; i < o; ++i)
        for (int j = 0; j < o; ++j)
        for (int a = 0; a < v; ++a)
        for (int b = 0; b < v; ++b) {
            real_t x2ab = 0.0, x2ba = 0.0;
            for (int d = 0; d < v; ++d) {
                x2ab += Y2[a * v + d] * t2(i, j, d, b);
                x2ba += Y2[b * v + d] * t2(i, j, d, a);
            }
            R(i, j, a, b) += 0.25 * (x2ab - x2ba);
        }
    }

    // Q3: Z3[i,a,l,d] = sum_{k,c} oovv_plain[k,l,c,d]*t2[i,k,a,c]
    {
        Tensor4 Z3(o, v, o, v);
        for (int i = 0; i < o; ++i)
        for (int a = 0; a < v; ++a)
        for (int l = 0; l < o; ++l)
        for (int d = 0; d < v; ++d) {
            real_t s = 0.0;
            for (int k = 0; k < o; ++k)
            for (int c = 0; c < v; ++c)
                s += oovv_plain(k, l, c, d) * t2(i, k, a, c);
            Z3(i, a, l, d) = s;
        }
        for (int i = 0; i < o; ++i)
        for (int j = 0; j < o; ++j)
        for (int a = 0; a < v; ++a)
        for (int b = 0; b < v; ++b) {
            real_t x3_ijab = 0.0, x3_jiab = 0.0, x3_ijba = 0.0, x3_jiba = 0.0;
            for (int l = 0; l < o; ++l)
            for (int d = 0; d < v; ++d) {
                x3_ijab += Z3(i, a, l, d) * t2(j, l, b, d);
                x3_jiab += Z3(j, a, l, d) * t2(i, l, b, d);
                x3_ijba += Z3(i, b, l, d) * t2(j, l, a, d);
                x3_jiba += Z3(j, b, l, d) * t2(i, l, a, d);
            }
            R(i, j, a, b) += 0.5 * (x3_ijab - x3_jiab - x3_ijba + x3_jiba);
        }
    }

    real_t rms = 0.0;
    for (auto x : R.data) rms += x * x;
    return std::sqrt(rms / R.data.size());
}

CCResult DCD::compute(real_t e_scf) {
    validate_scf(scf_);
    tensor_ops::init(opts_.use_gpu);

    const int o = scf_.n_occ, v = scf_.n_vir;

    auto oovv       = slice_oovv(scf_);
    auto oovv_plain = slice_oovv_plain(scf_);
    auto vvvv = slice_vvvv(scf_);
    auto oooo = slice_oooo(scf_);
    auto ovvo = slice_ovvo(scf_);
    auto D2   = make_D2(scf_);

    if (tensor_ops::gpu_active())
        tensor_ops::gpu_upload_integrals(vvvv, oooo, ovvo, scf_.eri_antisym.ptr());

    Matrix F_vv = scf_.fock.block(o, o, v, v);
    Matrix F_oo = scf_.fock.block(0, 0, o, o);

    Tensor4 t2(o, o, v, v);
    for (std::size_t n = 0; n < t2.data.size(); ++n)
        t2.data[n] = oovv.data[n] / D2.data[n];

    DIISState diis(opts_.diis_size);
    Tensor4 residual(o, o, v, v);

    real_t e_prev = 0.0;
    CCResult result{};
    result.converged = false;

    for (int iter = 1; iter <= opts_.max_iter; ++iter) {
        real_t rms = build_residual(t2, oovv, oovv_plain, vvvv, oooo, ovvo, F_vv, F_oo, residual);

        Tensor4 new_t2(o, o, v, v);
        for (std::size_t n = 0; n < t2.data.size(); ++n)
            new_t2.data[n] = t2.data[n] - residual.data[n] / D2.data[n];

        if (opts_.use_diis) {
            Tensor4 err(o, o, v, v);
            for (std::size_t n = 0; n < t2.data.size(); ++n)
                err.data[n] = new_t2.data[n] - t2.data[n];
            diis.push(err, new_t2);
            if (iter >= 2) {
                Tensor4 extrap = diis.extrapolate();
                extrap.n0 = o; extrap.n1 = o; extrap.n2 = v; extrap.n3 = v;
                new_t2 = extrap;
            }
        }

        t2 = new_t2;
        real_t e_corr = compute_energy(t2, oovv);
        real_t de = std::abs(e_corr - e_prev);
        e_prev = e_corr;

        if (callback_) callback_(iter, e_corr, de, rms);

        if (de < opts_.conv_energy && rms < opts_.conv_amp) {
            result.converged = true;
            result.n_iter    = iter;
            result.e_corr    = e_corr;
            result.e_total   = e_scf + e_corr;
            result.t2        = t2;
            break;
        }
    }

    if (!result.converged)
        std::cerr << "DCD: WARNING — did not converge in " << opts_.max_iter << " iterations\n";

    tensor_ops::finalize();
    return result;
}


// ============================================================
//  pCCD
// ============================================================

real_t pCCD::build_residual(const Tensor4& t2,
                              const Tensor4& oovv,
                              const Tensor4& oovv_plain,
                              const Tensor4& vvvv,
                              const Tensor4& oooo,
                              const Tensor4& ovvo,
                              const Matrix&  F_vv,
                              const Matrix&  F_oo,
                              Tensor4&       R) const {
    const int o = scf_.n_occ, v = scf_.n_vir;

    // Linear terms (Fock + bare ladders + bare ring)
    build_linear_residual(t2, oovv, vvvv, oooo, ovvo, F_vv, F_oo, o, v, R);

    // pCCD quadratic: A/2 + alpha*(A/2+B) + beta*(C+D),  D uses antisym ERIs
    // Pass oovv for oovv_plain so P1/P4 use antisym → pCCD(1,1) == CCD exactly
    Tensor4 Q(o, o, v, v);
    build_quad_Q(oovv, oovv, t2, o, v,
                 /*cDc=*/  0.5 * beta_,
                 /*cC=*/  -0.5 * beta_,
                 /*cA=*/  -0.25 * (1.0 + alpha_),
                 /*cDx=*/ -0.5 * beta_,
                 /*cB=*/   alpha_ / 8.0,
                 Q);

    for (int i = 0; i < o; ++i)
    for (int j = 0; j < o; ++j)
    for (int a = 0; a < v; ++a)
    for (int b = 0; b < v; ++b)
        R(i, j, a, b) += Q(i, j, a, b) + Q(j, i, b, a);

    real_t rms = 0.0;
    for (auto x : R.data) rms += x * x;
    return std::sqrt(rms / R.data.size());
}

CCResult pCCD::compute(real_t e_scf) {
    validate_scf(scf_);
    tensor_ops::init(opts_.use_gpu);

    const int o = scf_.n_occ, v = scf_.n_vir;

    auto oovv       = slice_oovv(scf_);
    auto oovv_plain = slice_oovv_plain(scf_);
    auto vvvv = slice_vvvv(scf_);
    auto oooo = slice_oooo(scf_);
    auto ovvo = slice_ovvo(scf_);
    auto D2   = make_D2(scf_);

    if (tensor_ops::gpu_active())
        tensor_ops::gpu_upload_integrals(vvvv, oooo, ovvo, scf_.eri_antisym.ptr());

    Matrix F_vv = scf_.fock.block(o, o, v, v);
    Matrix F_oo = scf_.fock.block(0, 0, o, o);

    Tensor4 t2(o, o, v, v);
    for (std::size_t n = 0; n < t2.data.size(); ++n)
        t2.data[n] = oovv.data[n] / D2.data[n];

    DIISState diis(opts_.diis_size);
    Tensor4 residual(o, o, v, v);

    real_t e_prev = 0.0;
    CCResult result{};
    result.converged = false;

    for (int iter = 1; iter <= opts_.max_iter; ++iter) {
        real_t rms = build_residual(t2, oovv, oovv_plain, vvvv, oooo, ovvo, F_vv, F_oo, residual);

        Tensor4 new_t2(o, o, v, v);
        for (std::size_t n = 0; n < t2.data.size(); ++n)
            new_t2.data[n] = t2.data[n] - residual.data[n] / D2.data[n];

        if (opts_.use_diis) {
            Tensor4 err(o, o, v, v);
            for (std::size_t n = 0; n < t2.data.size(); ++n)
                err.data[n] = new_t2.data[n] - t2.data[n];
            diis.push(err, new_t2);
            if (iter >= 2) {
                Tensor4 extrap = diis.extrapolate();
                extrap.n0 = o; extrap.n1 = o; extrap.n2 = v; extrap.n3 = v;
                new_t2 = extrap;
            }
        }

        t2 = new_t2;
        real_t e_corr = compute_energy(t2, oovv);
        real_t de = std::abs(e_corr - e_prev);
        e_prev = e_corr;

        if (callback_) callback_(iter, e_corr, de, rms);

        if (de < opts_.conv_energy && rms < opts_.conv_amp) {
            result.converged = true;
            result.n_iter    = iter;
            result.e_corr    = e_corr;
            result.e_total   = e_scf + e_corr;
            result.t2        = t2;
            break;
        }
    }

    if (!result.converged)
        std::cerr << "pCCD: WARNING — did not converge in " << opts_.max_iter << " iterations\n";

    tensor_ops::finalize();
    return result;
}

}  // namespace cupyccx
