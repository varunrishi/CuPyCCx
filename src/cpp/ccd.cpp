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
//  pCCD
// ============================================================

void pCCD::build_W_oooo(const Tensor4& oooo,
                         const Tensor4& oovv,
                         const Tensor4& t2,
                         Tensor4&       W_oooo) const {
    W_oooo = oooo;
    const int o = scf_.n_occ, v = scf_.n_vir;
    // Coefficient (1+alpha)/4 gives A*(1+alpha)/2 in the residual (= A/2 + alpha*A/2).
    // At alpha=1 this recovers the CCD coefficient of 0.5.
    const real_t coeff = (1.0 + alpha_) * 0.25;
    for (int k = 0; k < o; ++k)
    for (int l = 0; l < o; ++l)
    for (int i = 0; i < o; ++i)
    for (int j = 0; j < o; ++j) {
        real_t s = 0.0;
        for (int c = 0; c < v; ++c)
        for (int d = 0; d < v; ++d)
            s += oovv(k, l, c, d) * t2(i, j, c, d);
        W_oooo(k, l, i, j) += coeff * s;
    }
}

void pCCD::build_W_vvvv(const Tensor4& vvvv,
                         const Tensor4& oovv,
                         const Tensor4& t2,
                         Tensor4&       W_vvvv) const {
    W_vvvv = vvvv;
    const int o = scf_.n_occ, v = scf_.n_vir;
    // Coefficient alpha/2 gives alpha*B in the residual.
    // At alpha=1 this recovers the CCD coefficient of 0.5.
    const real_t coeff = alpha_ * 0.5;
    for (int a = 0; a < v; ++a)
    for (int b = 0; b < v; ++b)
    for (int c = 0; c < v; ++c)
    for (int d = 0; d < v; ++d) {
        real_t s = 0.0;
        for (int k = 0; k < o; ++k)
        for (int l = 0; l < o; ++l)
            s += oovv(k, l, c, d) * t2(k, l, a, b);
        W_vvvv(a, b, c, d) += coeff * s;
    }
}

real_t pCCD::build_residual(const Tensor4& t2,
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

    // (1/2) W_{klij} t_{kl}^{ab}  — quadratic contribution is A*(1+alpha)/2
    tensor_ops::contract_klij_klab(W_oooo, t2, R, 0.5);

    // (1/2) W_{abcd} t_{ij}^{cd}  — quadratic contribution is alpha*B
    tensor_ops::contract_abcd_ijcd(W_vvvv, t2, R, 0.5);

    // P(ij)P(ab) <kb||cj> t_{ik}^{ac}  (linear, same as CCD)
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

    // TODO: add beta_*(C + D) quadratic box/cross terms here.
    // These are the T2-dependent corrections to the ovvo block (W_ovvo).
    // Provide the explicit equations from the reference to implement this.
    (void)beta_;

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

    if (!result.converged)
        std::cerr << "pCCD: WARNING — did not converge in " << opts_.max_iter << " iterations\n";

    tensor_ops::finalize();
    return result;
}

}  // namespace cupyccx
