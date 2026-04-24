#pragma once

#include <Eigen/Dense>
#include <cstddef>
#include <string>
#include <vector>

namespace cupyccx {

// Floating-point precision policy
#ifdef CUPYCCX_DOUBLE_PRECISION
using real_t = double;
#else
using real_t = double;  // default to double for quantum chemistry accuracy
#endif

// Eigen matrix/vector aliases
using Matrix = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Vector = Eigen::Matrix<real_t, Eigen::Dynamic, 1>;

// Rank-4 tensor stored as flat array (row-major, p*q*r*s indexing)
// We wrap raw data arrays rather than Eigen::Tensor to keep CUDA interop simple
struct Tensor4 {
    std::vector<real_t> data;
    std::size_t n0, n1, n2, n3;

    Tensor4() = default;
    Tensor4(std::size_t n0, std::size_t n1, std::size_t n2, std::size_t n3)
        : data(n0 * n1 * n2 * n3, real_t(0)), n0(n0), n1(n1), n2(n2), n3(n3) {}

    inline real_t& operator()(std::size_t i, std::size_t j, std::size_t k, std::size_t l) {
        return data[i * n1 * n2 * n3 + j * n2 * n3 + k * n3 + l];
    }
    inline real_t operator()(std::size_t i, std::size_t j, std::size_t k, std::size_t l) const {
        return data[i * n1 * n2 * n3 + j * n2 * n3 + k * n3 + l];
    }

    std::size_t size() const { return data.size(); }
    real_t*       ptr()       { return data.data(); }
    const real_t* ptr() const { return data.data(); }
};

// SCF reference data passed to all correlated methods
struct SCFData {
    int    n_occ;          // number of occupied spin-orbitals
    int    n_vir;          // number of virtual  spin-orbitals
    int    n_mo;           // n_occ + n_vir
    Vector eps;            // orbital energies (length n_mo), occ first
    Matrix fock;           // Fock matrix in MO basis (n_mo x n_mo)
    // Two-electron integrals in physicist notation <pq||rs> (antisymmetrized)
    // Stored as flat Tensor4(n_mo, n_mo, n_mo, n_mo)
    Tensor4 eri_antisym;
};

// Options controlling a coupled-cluster calculation
struct CCOptions {
    int    max_iter    = 100;
    real_t conv_energy = 1e-9;
    real_t conv_amp    = 1e-7;
    bool   use_diis    = true;
    int    diis_size   = 6;
    bool   use_gpu     = false;   // enable CUDA tensor contractions if available
    std::string method = "CCD";   // "LCCD", "CCD"
};

// Convergence information returned after a CC calculation
struct CCResult {
    real_t e_corr;           // correlation energy
    real_t e_total;          // SCF energy + correlation energy
    int    n_iter;           // iterations to convergence
    bool   converged;
    Tensor4 t2;              // converged T2 amplitudes t_{ij}^{ab}
};

}  // namespace cupyccx
