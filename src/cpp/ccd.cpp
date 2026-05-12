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
    // W_{klij} += (1/2) sum_{cd} <kl||cd> t_{ij}^{cd}
    for (int k = 0; k < o; ++k)
    for (int l = 0; l < o; ++l)
    for (int i = 0; i < o; ++i)
    for (int j = 0; j < o; ++j) {
        real_t s = 0.0;
        for (int c = 0; c < v; ++c)
        for (int d = 0; d < v; ++d)
            s += oovv(k, l, c, d) * t2(i, j, c, d);
        W_oooo(k, l, i, j) += 0.5 * s;
    }
}

void CCD::build_W_vvvv(const Tensor4& vvvv,
                        const Tensor4& oovv,
                        const Tensor4& t2,
                        Tensor4&       W_vvvv) const {
    W_vvvv = vvvv;
    if (variant_ == "LCCD") return;  // LCCD: no T2 contribution
    const int o = scf_.n_occ, v = scf_.n_vir;
    // W_{abcd} += (1/2) sum_{kl} <kl||cd> t_{kl}^{ab}
    for (int a = 0; a < v; ++a)
    for (int b = 0; b < v; ++b)
    for (int c = 0; c < v; ++c)
    for (int d = 0; d < v; ++d) {
        real_t s = 0.0;
        for (int k = 0; k < o; ++k)
        for (int l = 0; l < o; ++l)
            s += oovv(k, l, c, d) * t2(k, l, a, b);
        W_vvvv(a, b, c, d) += 0.5 * s;
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
// DCD uses:   DCD_1C + 0.5*(DCD_3 + DCD_4)
// pCCD uses:  beta*(DCD_1C+DCD_1X+DCD_3) + alpha*(0.5*DCD_4+DCD_5) + 0.5*DCD_4
//
// After calling this, the caller should add Q[i,j,a,b] + Q[j,i,b,a] to R.
//
// All quadratic terms reduce to five index patterns (using antisymmetry of
// oovv and t2 to simplify the einsum expressions):
//
//   P1[i,j,a,b] = sum_{k,l,c,d} oovv(k,l,c,d) * t2(i,l,a,d) * t2(k,j,c,b)
//   P2[i,j,a,b] = sum_{k,l,c,d} oovv(k,l,c,d) * t2(k,l,a,d) * t2(i,j,c,b)
//   P3[i,j,a,b] = sum_{k,l,c,d} oovv(k,l,c,d) * t2(i,l,c,d) * t2(k,j,a,b)
//   P4[i,j,a,b] = sum_{k,l,c,d} oovv(k,l,c,d) * t2(i,l,d,b) * t2(k,j,a,c)
//   P5[i,j,a,b] = sum_{k,l,c,d} oovv(k,l,c,d) * t2(i,j,c,d) * t2(k,l,a,b)
//
// Simplified coefficients (after applying antisymmetry to the Python einsum):
//   DCD_1C         =  4.5 * P1
//   DCD_1X         =  2.0 * P1 - 0.5 * P4
//   DCD_3          = -3.0 * P2
//   DCD_4          = -3.0 * P3
//   DCD_5          =  0.5 * P5
//
// For DCD:     Q = 4.5*P1 - 1.5*P2 - 1.5*P3
// For pCCD:    Q = (6.5*beta)*P1 - (0.5*beta)*P4 - (3*beta)*P2
//                  - 1.5*(1+alpha)*P3 + (0.5*alpha)*P5
// ---------------------------------------------------------------------------
static void build_quad_Q(
        const Tensor4& oovv, const Tensor4& t2, int o, int v,
        real_t c1,   // coeff of P1
        real_t c2,   // coeff of P2
        real_t c3,   // coeff of P3
        real_t c4,   // coeff of P4
        real_t c5,   // coeff of P5
        Tensor4& Q)
{
    // --- P1 via intermediate A[l,d,j,b] = sum_{k,c} oovv(k,l,c,d)*t2(k,j,c,b) ---
    if (c1 != 0.0) {
        Tensor4 A(o, v, o, v);   // A[l,d,j,b]
        for (int l = 0; l < o; ++l)
        for (int d = 0; d < v; ++d)
        for (int jj = 0; jj < o; ++jj)
        for (int b = 0; b < v; ++b) {
            real_t s = 0.0;
            for (int k = 0; k < o; ++k)
            for (int c = 0; c < v; ++c)
                s += oovv(k, l, c, d) * t2(k, jj, c, b);
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
            Q(i, jj, a, b) += c1 * s;
        }
    }

    // --- P2 via intermediate B[a,c] = sum_{k,l,d} oovv(k,l,c,d)*t2(k,l,a,d) ---
    if (c2 != 0.0) {
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
            Q(i, jj, a, b) += c2 * s;
        }
    }

    // --- P3 via intermediate C[k,i] = sum_{l,c,d} oovv(k,l,c,d)*t2(i,l,c,d) ---
    if (c3 != 0.0) {
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
            Q(i, jj, a, b) += c3 * s;
        }
    }

    // --- P4 via intermediate H[k,c,i,b] = sum_{l,d} oovv(k,l,c,d)*t2(i,l,d,b) ---
    if (c4 != 0.0) {
        Tensor4 H(o, v, o, v);   // H[k,c,i,b]
        for (int k = 0; k < o; ++k)
        for (int c = 0; c < v; ++c)
        for (int i = 0; i < o; ++i)
        for (int b = 0; b < v; ++b) {
            real_t s = 0.0;
            for (int l = 0; l < o; ++l)
            for (int d = 0; d < v; ++d)
                s += oovv(k, l, c, d) * t2(i, l, d, b);
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
            Q(i, jj, a, b) += c4 * s;
        }
    }

    // --- P5 via intermediate M[c,d,a,b] = sum_{k,l} oovv(k,l,c,d)*t2(k,l,a,b) ---
    if (c5 != 0.0) {
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
            Q(i, jj, a, b) += c5 * s;
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
                            const Tensor4& vvvv,
                            const Tensor4& oooo,
                            const Tensor4& ovvo,
                            const Matrix&  F_vv,
                            const Matrix&  F_oo,
                            Tensor4&       R) const {
    const int o = scf_.n_occ, v = scf_.n_vir;

    // Linear terms (Fock + bare ladders + bare ring)
    build_linear_residual(t2, oovv, vvvv, oooo, ovvo, F_vv, F_oo, o, v, R);

    // Quadratic DCD terms:  Q = 4.5*P1 - 1.5*P2 - 1.5*P3
    Tensor4 Q(o, o, v, v);
    build_quad_Q(oovv, t2, o, v,
                 /*c1=*/ 4.5,  /*c2=*/-1.5,  /*c3=*/-1.5,
                 /*c4=*/ 0.0,  /*c5=*/ 0.0,
                 Q);

    // Add symmetrised quadratic: Q[i,j,a,b] + Q[j,i,b,a]
    for (int i = 0; i < o; ++i)
    for (int j = 0; j < o; ++j)
    for (int a = 0; a < v; ++a)
    for (int b = 0; b < v; ++b)
        R(i, j, a, b) += Q(i, j, a, b) + Q(j, i, b, a);

    real_t rms = 0.0;
    for (auto x : R.data) rms += x * x;
    return std::sqrt(rms / R.data.size());
}

CCResult DCD::compute(real_t e_scf) {
    validate_scf(scf_);
    tensor_ops::init(opts_.use_gpu);

    const int o = scf_.n_occ, v = scf_.n_vir;

    auto oovv = slice_oovv(scf_);
    auto vvvv = slice_vvvv(scf_);
    auto oooo = slice_oooo(scf_);
    auto ovvo = slice_ovvo(scf_);
    auto D2   = make_D2(scf_);

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
        real_t rms = build_residual(t2, oovv, vvvv, oooo, ovvo, F_vv, F_oo, residual);

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
                              const Tensor4& vvvv,
                              const Tensor4& oooo,
                              const Tensor4& ovvo,
                              const Matrix&  F_vv,
                              const Matrix&  F_oo,
                              Tensor4&       R) const {
    const int o = scf_.n_occ, v = scf_.n_vir;

    // Linear terms (Fock + bare ladders + bare ring)
    build_linear_residual(t2, oovv, vvvv, oooo, ovvo, F_vv, F_oo, o, v, R);

    // pCCD quadratic:
    //   Q = (6.5*beta)*P1 - (0.5*beta)*P4 - (3*beta)*P2
    //       - 1.5*(1+alpha)*P3 + (0.5*alpha)*P5
    Tensor4 Q(o, o, v, v);
    build_quad_Q(oovv, t2, o, v,
                 /*c1=*/  6.5 * beta_,
                 /*c2=*/ -3.0 * beta_,
                 /*c3=*/ -1.5 * (1.0 + alpha_),
                 /*c4=*/ -0.5 * beta_,
                 /*c5=*/  0.5 * alpha_,
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

    auto oovv = slice_oovv(scf_);
    auto vvvv = slice_vvvv(scf_);
    auto oooo = slice_oooo(scf_);
    auto ovvo = slice_ovvo(scf_);
    auto D2   = make_D2(scf_);

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
        real_t rms = build_residual(t2, oovv, vvvv, oooo, ovvo, F_vv, F_oo, residual);

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
