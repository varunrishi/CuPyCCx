#pragma once

#include "types.hpp"
#include <functional>

namespace cupyccx {

// ---------------------------------------------------------------------------
// Abstract base for all coupled-cluster doubles variants
// ---------------------------------------------------------------------------
class CCDBase {
public:
    explicit CCDBase(const SCFData& scf, const CCOptions& opts)
        : scf_(scf), opts_(opts) {}

    virtual ~CCDBase() = default;

    // Run the CC iterations; returns converged result
    virtual CCResult compute(real_t e_scf = 0.0) = 0;

    // Optional callback called after each macro-iteration:
    //   (iteration, energy, delta_energy, rms_residual)
    using IterCallback = std::function<void(int, real_t, real_t, real_t)>;
    void set_callback(IterCallback cb) { callback_ = cb; }

protected:
    const SCFData&   scf_;
    const CCOptions& opts_;
    IterCallback     callback_;

    // Compute correlation energy from T2 amplitudes and <ij||ab>
    real_t compute_energy(const Tensor4& t2, const Tensor4& oovv) const;

    // DIIS extrapolation helper (stores error vectors and amplitudes)
    struct DIISState {
        std::vector<std::vector<real_t>> errors;
        std::vector<std::vector<real_t>> amps;
        int max_vecs;
        DIISState(int max_vecs) : max_vecs(max_vecs) {}
        void push(const Tensor4& error, const Tensor4& amp);
        Tensor4 extrapolate() const;
    };
};

}  // namespace cupyccx
