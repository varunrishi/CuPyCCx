#pragma once

#include "ccd_base.hpp"

namespace cupyccx {

// ---------------------------------------------------------------------------
// Linearized CCD (LCCD)
//
// Amplitude equations with quadratic T2*T2 terms dropped:
//
//   r_{ij}^{ab} = <ij||ab>
//               + P(ab) f_bc t_{ij}^{ac}  -  P(ij) f_kj t_{ik}^{ab}
//               + (1/2) <kl||ij> t_{kl}^{ab}
//               + (1/2) <ab||cd> t_{ij}^{cd}
//               + P(ij)P(ab) <kb||cj> t_{ik}^{ac}
//
// Reference: Bartlett & Purvis, Int. J. Quantum Chem. 14, 561 (1978)
// ---------------------------------------------------------------------------
class LCCD : public CCDBase {
public:
    using CCDBase::CCDBase;
    CCResult compute(real_t e_scf = 0.0) override;

private:
    // Build the residual (return rms norm)
    real_t build_residual(const Tensor4& t2,
                          const Tensor4& oovv,
                          const Tensor4& vvvv,
                          const Tensor4& oooo,
                          const Tensor4& ovvo,
                          const Matrix&  F_vv,
                          const Matrix&  F_oo,
                          Tensor4&       residual) const;
};

// ---------------------------------------------------------------------------
// Full CCD
//
// Extends LCCD with quadratic T2*T2 contributions that renormalize the
// effective two-particle interaction:
//
//   W_{klij} = <kl||ij> + (1/2) sum_{cd} <kl||cd> t_{ij}^{cd}
//   W_{abcd} = <ab||cd> + (1/2) sum_{kl} <kl||cd> t_{kl}^{ab}   (ring)
//
// The W intermediates replace the bare integrals in the residual expression.
//
// Reference: Purvis & Bartlett, J. Chem. Phys. 76, 1910 (1982)
// ---------------------------------------------------------------------------
class CCD : public CCDBase {
public:
    using CCDBase::CCDBase;
    CCResult compute(real_t e_scf = 0.0) override;

private:
    // Build effective W intermediates
    void build_W_oooo(const Tensor4& oooo,
                      const Tensor4& oovv,
                      const Tensor4& t2,
                      Tensor4&       W_oooo) const;

    void build_W_vvvv(const Tensor4& vvvv,
                      const Tensor4& oovv,
                      const Tensor4& t2,
                      Tensor4&       W_vvvv) const;

    real_t build_residual(const Tensor4& t2,
                          const Tensor4& oovv,
                          const Tensor4& W_vvvv,
                          const Tensor4& W_oooo,
                          const Tensor4& ovvo,
                          const Matrix&  F_vv,
                          const Matrix&  F_oo,
                          Tensor4&       residual) const;
};

}  // namespace cupyccx
