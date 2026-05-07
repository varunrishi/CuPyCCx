#pragma once

#include "ccd_base.hpp"
#include <string>

namespace cupyccx {

// ---------------------------------------------------------------------------
// CCD family (variant selected by name)
//
// variant="CCD"  — full CCD; W intermediates include T2*T2 contributions:
//   W_{klij} = <kl||ij> + (1/2) sum_{cd} <kl||cd> t_{ij}^{cd}
//   W_{abcd} = <ab||cd> + (1/2) sum_{kl} <kl||cd> t_{kl}^{ab}   (ring)
//
// variant="LCCD" — linearized CCD; T2-dependent parts of W are omitted so
//   W reduces to the bare integrals.  All other terms are identical to CCD.
//
// References:
//   LCCD: Bartlett & Purvis, Int. J. Quantum Chem. 14, 561 (1978)
//   CCD:  Purvis & Bartlett, J. Chem. Phys. 76, 1910 (1982)
// ---------------------------------------------------------------------------
class CCD : public CCDBase {
public:
    explicit CCD(const SCFData& scf, const CCOptions& opts,
                 std::string variant = "CCD")
        : CCDBase(scf, opts), variant_(std::move(variant)) {}

    CCResult compute(real_t e_scf = 0.0) override;

private:
    std::string variant_;  // "CCD", "LCCD", ...

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


// ---------------------------------------------------------------------------
// Parameterized CCD (pCCD)
//
// Scales the four classes of quadratic T2*T2 diagrams independently:
//
//   Quadratic in CCD  = A + B + C + D
//   Quadratic in pCCD = A/2 + alpha*(A/2 + B) + beta*(C + D)
//
// where A is the ring-ring term (W_oooo), B is the ladder-ladder term
// (W_vvvv), and C+D are the box/cross quadratic terms.
//
// At alpha=1, beta=1: recovers full CCD.
//
// Reference: Huntington and Nooijen, J. Chem. Phys. 133, 184109 (2010)
// ---------------------------------------------------------------------------
class pCCD : public CCDBase {
public:
    pCCD(const SCFData& scf, const CCOptions& opts, real_t alpha, real_t beta)
        : CCDBase(scf, opts), alpha_(alpha), beta_(beta) {}
    CCResult compute(real_t e_scf = 0.0) override;

private:
    real_t alpha_;
    real_t beta_;

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
