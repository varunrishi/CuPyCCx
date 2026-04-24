#include "cupyccx/ccd_base.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>

namespace cupyccx {

// ---------------------------------------------------------------------------
// Correlation energy:  E_corr = (1/4) sum_{ijab} <ij||ab> t_{ij}^{ab}
// ---------------------------------------------------------------------------
real_t CCDBase::compute_energy(const Tensor4& t2, const Tensor4& oovv) const {
    const int o = scf_.n_occ, v = scf_.n_vir;
    real_t e = 0.0;
    for (int i = 0; i < o; ++i)
    for (int j = 0; j < o; ++j)
    for (int a = 0; a < v; ++a)
    for (int b = 0; b < v; ++b)
        e += oovv(i, j, a, b) * t2(i, j, a, b);
    return 0.25 * e;
}

// ---------------------------------------------------------------------------
// DIIS implementation
// ---------------------------------------------------------------------------
void CCDBase::DIISState::push(const Tensor4& error, const Tensor4& amp) {
    auto to_vec = [](const Tensor4& t) {
        return std::vector<real_t>(t.data.begin(), t.data.end());
    };
    if (static_cast<int>(errors.size()) >= max_vecs) {
        errors.erase(errors.begin());
        amps.erase(amps.begin());
    }
    errors.push_back(to_vec(error));
    amps.push_back(to_vec(amp));
}

Tensor4 CCDBase::DIISState::extrapolate() const {
    const int m = static_cast<int>(errors.size());
    if (m == 0) throw std::runtime_error("DIIS: no vectors stored");

    // Build B matrix (m+1 x m+1)
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(m + 1, m + 1);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j <= i; ++j) {
            double dot = 0.0;
            for (std::size_t k = 0; k < errors[i].size(); ++k)
                dot += errors[i][k] * errors[j][k];
            B(i, j) = B(j, i) = dot;
        }
        B(i, m) = B(m, i) = -1.0;
    }
    B(m, m) = 0.0;

    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(m + 1);
    rhs(m) = -1.0;

    Eigen::VectorXd c = B.colPivHouseholderQr().solve(rhs);

    // Extrapolated amplitudes
    const std::size_t sz = amps[0].size();
    std::vector<real_t> new_amp(sz, real_t(0));
    for (int i = 0; i < m; ++i)
        for (std::size_t k = 0; k < sz; ++k)
            new_amp[k] += static_cast<real_t>(c(i)) * amps[i][k];

    // Reconstruct a Tensor4 with the same shape as the stored amps
    // (We infer shape from the amplitude count; caller must match)
    Tensor4 out;
    out.data = new_amp;
    // Shape fields are filled by the caller after this returns
    return out;
}

}  // namespace cupyccx
