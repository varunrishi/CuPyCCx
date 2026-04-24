#include "cupyccx/ccd.hpp"
#include "cupyccx/integrals.hpp"
#include "cupyccx/tensor_ops.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace cupyccx {

// ============================================================
//  LCCD
// ============================================================

real_t LCCD::build_residual(const Tensor4& t2,
                             const Tensor4& oovv,
                             const Tensor4& vvvv,
                             const Tensor4& oooo,
                             const Tensor4& ovvo,
                             const Matrix&  F_vv,
                             const Matrix&  F_oo,
                             Tensor4&       R) const {
    const int o = scf_.n_occ, v = scf_.n_vir;

    // Initialise with <ij||ab>
    R = oovv;

    // P(ab) f_bc t_{ij}^{ac}  — virtual Fock contribution
    tensor_ops::contract_ijac_bc(t2, F_vv, R, +1.0);
    // Antisymmetry P(ab): subtract transposed (a<->b)
    {
        Tensor4 tmp(o, o, v, v);
        for (int i = 0; i < o; ++i)
        for (int j = 0; j < o; ++j)
        for (int a = 0; a < v; ++a)
        for (int c = 0; c < v; ++c) {
            real_t s = 0.0;
            for (int b = 0; b < v; ++b) s += F_vv(b, c) * t2(i, j, b, a);
            tmp(i, j, a, c) = s;
        }
        for (int i = 0; i < o; ++i)
        for (int j = 0; j < o; ++j)
        for (int a = 0; a < v; ++a)
        for (int b = 0; b < v; ++b)
            R(i, j, a, b) -= tmp(i, j, a, b);
    }

    // -P(ij) f_kj t_{ik}^{ab}  — occupied Fock contribution
    tensor_ops::contract_ikab_kj(t2, F_oo, R, -1.0);
    {
        Tensor4 tmp(o, o, v, v);
        for (int i = 0; i < o; ++i)
        for (int k = 0; k < o; ++k)
        for (int a = 0; a < v; ++a)
        for (int b = 0; b < v; ++b) {
            real_t s = 0.0;
            for (int j = 0; j < o; ++j) s += F_oo(k, j) * t2(j, k, a, b);
            tmp(i, k, a, b) = s;
        }
        for (int i = 0; i < o; ++i)
        for (int j = 0; j < o; ++j)
        for (int a = 0; a < v; ++a)
        for (int b = 0; b < v; ++b)
            R(i, j, a, b) += tmp(i, j, a, b);
    }

    // (1/2) <kl||ij> t_{kl}^{ab}
    tensor_ops::contract_klij_klab(oooo, t2, R, 0.5);

    // (1/2) <ab||cd> t_{ij}^{cd}
    tensor_ops::contract_abcd_ijcd(vvvv, t2, R, 0.5);

    // P(ij)P(ab) <kb||cj> t_{ik}^{ac}
    tensor_ops::contract_kbcj_ikac(ovvo, t2, R, +1.0);
    // Antisymmetry terms (i<->j) and (a<->b)
    {
        Tensor4 tmp(o, o, v, v);
        // -<ka||cj> t_{ik}^{bc}
        for (int i = 0; i < o; ++i)
        for (int j = 0; j < o; ++j)
        for (int a = 0; a < v; ++a)
        for (int b = 0; b < v; ++b) {
            real_t s = 0.0;
            for (int k = 0; k < o; ++k)
            for (int c = 0; c < v; ++c)
                s += ovvo(k, a, c, j) * t2(i, k, b, c);
            tmp(i, j, a, b) = s;
        }
        for (auto& x : tmp.data) x = -x;
        for (std::size_t n = 0; n < R.data.size(); ++n)
            R.data[n] += tmp.data[n];

        // -<kb||ci> t_{jk}^{ac}
        for (int i = 0; i < o; ++i)
        for (int j = 0; j < o; ++j)
        for (int a = 0; a < v; ++a)
        for (int b = 0; b < v; ++b) {
            real_t s = 0.0;
            for (int k = 0; k < o; ++k)
            for (int c = 0; c < v; ++c)
                s += ovvo(k, b, c, i) * t2(j, k, a, c);
            tmp(i, j, a, b) = -s;
        }
        for (std::size_t n = 0; n < R.data.size(); ++n)
            R.data[n] += tmp.data[n];

        // +<ka||ci> t_{jk}^{bc}
        for (int i = 0; i < o; ++i)
        for (int j = 0; j < o; ++j)
        for (int a = 0; a < v; ++a)
        for (int b = 0; b < v; ++b) {
            real_t s = 0.0;
            for (int k = 0; k < o; ++k)
            for (int c = 0; c < v; ++c)
                s += ovvo(k, a, c, i) * t2(j, k, b, c);
            tmp(i, j, a, b) = s;
        }
        for (std::size_t n = 0; n < R.data.size(); ++n)
            R.data[n] += tmp.data[n];
    }

    // Compute RMS residual
    real_t rms = 0.0;
    for (auto x : R.data) rms += x * x;
    return std::sqrt(rms / R.data.size());
}

CCResult LCCD::compute(real_t e_scf) {
    validate_scf(scf_);
    tensor_ops::init(opts_.use_gpu);

    const int o = scf_.n_occ, v = scf_.n_vir;

    // Pre-slice integrals
    auto oovv = slice_oovv(scf_);
    auto vvvv = slice_vvvv(scf_);
    auto oooo = slice_oooo(scf_);
    auto ovvo = slice_ovvo(scf_);
    auto D2   = make_D2(scf_);

    // Virtual and occupied Fock blocks (off-diagonal elements)
    Matrix F_vv = scf_.fock.block(o, o, v, v);
    Matrix F_oo = scf_.fock.block(0, 0, o, o);

    // Initial T2 = <ij||ab> / D_{ij}^{ab}  (MP2 amplitudes)
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

        // Update amplitudes: t2 -= R / D2
        Tensor4 new_t2(o, o, v, v);
        for (std::size_t n = 0; n < t2.data.size(); ++n)
            new_t2.data[n] = t2.data[n] - residual.data[n] / D2.data[n];

        if (opts_.use_diis) {
            // Error vector = new_t2 - t2
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
        std::cerr << "LCCD: WARNING — did not converge in " << opts_.max_iter << " iterations\n";

    tensor_ops::finalize();
    return result;
}


// ============================================================
//  CCD
// ============================================================

void CCD::build_W_oooo(const Tensor4& oooo,
                        const Tensor4& oovv,
                        const Tensor4& t2,
                        Tensor4&       W_oooo) const {
    W_oooo = oooo;
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

    // Fock contributions (same as LCCD)
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

    // (1/2) W_{klij} t_{kl}^{ab}
    tensor_ops::contract_klij_klab(W_oooo, t2, R, 0.5);

    // (1/2) W_{abcd} t_{ij}^{cd}
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

    if (!result.converged)
        std::cerr << "CCD: WARNING — did not converge in " << opts_.max_iter << " iterations\n";

    tensor_ops::finalize();
    return result;
}

}  // namespace cupyccx
