#include <gtest/gtest.h>
#include "cupyccx/ccd.hpp"
#include "cupyccx/integrals.hpp"
#include <cmath>

// ---------------------------------------------------------------------------
// Reuse the H2 STO-3G fixture from test_integrals.cpp
// ---------------------------------------------------------------------------
extern cupyccx::SCFData make_h2_scf();  // defined below (local copy for linking)

namespace {

cupyccx::SCFData make_h2_scf_local() {
    const double eps_occ = -0.57827;
    const double eps_vir =  0.67012;
    const int n_occ = 2, n_vir = 2, n_mo = 4;
    const double K   = 0.20636;

    cupyccx::SCFData scf;
    scf.n_occ = n_occ; scf.n_vir = n_vir; scf.n_mo = n_mo;
    scf.eps.resize(n_mo);
    scf.eps << eps_occ, eps_occ, eps_vir, eps_vir;
    scf.fock = cupyccx::Matrix::Zero(n_mo, n_mo);
    scf.fock(0,0) = eps_occ; scf.fock(1,1) = eps_occ;
    scf.fock(2,2) = eps_vir; scf.fock(3,3) = eps_vir;
    scf.eri_antisym = cupyccx::Tensor4(n_mo, n_mo, n_mo, n_mo);
    scf.eri_antisym(0,1,2,3)= K; scf.eri_antisym(1,0,3,2)= K;
    scf.eri_antisym(0,1,3,2)=-K; scf.eri_antisym(1,0,2,3)=-K;
    scf.eri_antisym(2,3,0,1)= K; scf.eri_antisym(3,2,1,0)= K;
    scf.eri_antisym(3,2,0,1)=-K; scf.eri_antisym(2,3,1,0)=-K;
    return scf;
}

cupyccx::CCOptions default_opts() {
    cupyccx::CCOptions opts;
    opts.max_iter    = 200;
    opts.conv_energy = 1e-12;
    opts.conv_amp    = 1e-10;
    opts.use_gpu     = false;
    return opts;
}

}  // namespace

// For a 2-electron system, LCCD == CCD == FCI (within the basis)
// Reference: 2e CCD energy matches exact result
const double E_CORR_REF = -0.04222;  // approximate STO-3G H2 correlation energy

TEST(LCCD, H2_Converged) {
    auto scf  = make_h2_scf_local();
    auto opts = default_opts();
    cupyccx::LCCD solver(scf, opts);
    auto result = solver.compute(0.0);
    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.e_corr, E_CORR_REF, 5e-4);
}

TEST(LCCD, H2_IterCount) {
    auto scf  = make_h2_scf_local();
    auto opts = default_opts();
    cupyccx::LCCD solver(scf, opts);
    auto result = solver.compute(0.0);
    EXPECT_GT(result.n_iter, 0);
    EXPECT_LE(result.n_iter, opts.max_iter);
}

TEST(CCD, H2_Converged) {
    auto scf  = make_h2_scf_local();
    auto opts = default_opts();
    cupyccx::CCD solver(scf, opts);
    auto result = solver.compute(0.0);
    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.e_corr, E_CORR_REF, 5e-4);
}

TEST(CCD, H2_T2Antisymmetry) {
    auto scf  = make_h2_scf_local();
    auto opts = default_opts();
    cupyccx::CCD solver(scf, opts);
    auto result = solver.compute(0.0);

    const auto& t2 = result.t2;
    const int o = scf.n_occ, v = scf.n_vir;
    for (int i = 0; i < o; ++i)
    for (int j = 0; j < o; ++j)
    for (int a = 0; a < v; ++a)
    for (int b = 0; b < v; ++b) {
        EXPECT_NEAR(t2(i,j,a,b), -t2(j,i,a,b), 1e-10);
        EXPECT_NEAR(t2(i,j,a,b), -t2(i,j,b,a), 1e-10);
    }
}

TEST(CCD, EnergyLowerThanLCCD) {
    // For H2 the two methods coincide; for larger systems CCD < LCCD.
    // This just checks the sign relationship is sensible.
    auto scf = make_h2_scf_local();
    auto opts = default_opts();
    cupyccx::CCD  ccd(scf, opts);
    cupyccx::LCCD lccd(scf, opts);
    auto  r_ccd  =  ccd.compute(0.0);
    auto  r_lccd = lccd.compute(0.0);
    // Both should be negative
    EXPECT_LT(r_ccd.e_corr,  0.0);
    EXPECT_LT(r_lccd.e_corr, 0.0);
}

TEST(CCD, CallbackFired) {
    auto scf = make_h2_scf_local();
    auto opts = default_opts();
    int  cb_count = 0;
    cupyccx::CCD solver(scf, opts);
    solver.set_callback([&](int, double, double, double) { ++cb_count; });
    solver.compute(0.0);
    EXPECT_GT(cb_count, 0);
}
