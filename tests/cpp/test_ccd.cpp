#include <gtest/gtest.h>
#include "cupyccx/ccd.hpp"
#include "cupyccx/integrals.hpp"
#include <cmath>

// ---------------------------------------------------------------------------
// Complete H2 / STO-3G spin-orbital fixture (R = 1.4 bohr).
// All integrals are from PySCF; see test_integrals.cpp for derivation.
//
// Reference energies (PySCF):
//   E_corr(MP2)  = -0.01315787 Ha  (= LCCD for 2 electrons)
//   E_corr(CCSD) = -0.02056178 Ha  (= CCD for 2 electrons)
// ---------------------------------------------------------------------------

namespace {

cupyccx::SCFData make_h2_scf() {
    const double eps_occ = -0.57820298;
    const double eps_vir =  0.67026777;
    const int n_occ = 2, n_vir = 2, n_mo = 4;

    cupyccx::SCFData scf;
    scf.n_occ = n_occ; scf.n_vir = n_vir; scf.n_mo = n_mo;
    scf.eps.resize(n_mo);
    scf.eps << eps_occ, eps_occ, eps_vir, eps_vir;
    scf.fock = cupyccx::Matrix::Zero(n_mo, n_mo);
    scf.fock(0,0) = eps_occ; scf.fock(1,1) = eps_occ;
    scf.fock(2,2) = eps_vir; scf.fock(3,3) = eps_vir;

    scf.eri_antisym = cupyccx::Tensor4(n_mo, n_mo, n_mo, n_mo);
    // OOOO
    scf.eri_antisym(0,1,0,1) =  0.67459408; scf.eri_antisym(0,1,1,0) = -0.67459408;
    scf.eri_antisym(1,0,0,1) = -0.67459408; scf.eri_antisym(1,0,1,0) =  0.67459408;
    // OOVV + VVOO
    scf.eri_antisym(0,1,2,3) =  0.18125791; scf.eri_antisym(0,1,3,2) = -0.18125791;
    scf.eri_antisym(1,0,2,3) = -0.18125791; scf.eri_antisym(1,0,3,2) =  0.18125791;
    scf.eri_antisym(2,3,0,1) =  0.18125791; scf.eri_antisym(2,3,1,0) = -0.18125791;
    scf.eri_antisym(3,2,0,1) = -0.18125791; scf.eri_antisym(3,2,1,0) =  0.18125791;
    // OVOV same-spin
    scf.eri_antisym(0,2,0,2) =  0.48230608; scf.eri_antisym(0,2,2,0) = -0.48230608;
    scf.eri_antisym(2,0,0,2) = -0.48230608; scf.eri_antisym(2,0,2,0) =  0.48230608;
    scf.eri_antisym(1,3,1,3) =  0.48230608; scf.eri_antisym(1,3,3,1) = -0.48230608;
    scf.eri_antisym(3,1,1,3) = -0.48230608; scf.eri_antisym(3,1,3,1) =  0.48230608;
    // OVOV opposite-spin
    scf.eri_antisym(0,3,0,3) =  0.66356399; scf.eri_antisym(0,3,3,0) = -0.66356399;
    scf.eri_antisym(3,0,0,3) = -0.66356399; scf.eri_antisym(3,0,3,0) =  0.66356399;
    scf.eri_antisym(1,2,1,2) =  0.66356399; scf.eri_antisym(1,2,2,1) = -0.66356399;
    scf.eri_antisym(2,1,1,2) = -0.66356399; scf.eri_antisym(2,1,2,1) =  0.66356399;
    // OVOV opposite-spin cross terms
    scf.eri_antisym(0,3,1,2) = -0.18125791; scf.eri_antisym(0,3,2,1) =  0.18125791;
    scf.eri_antisym(3,0,1,2) =  0.18125791; scf.eri_antisym(3,0,2,1) = -0.18125791;
    scf.eri_antisym(1,2,0,3) = -0.18125791; scf.eri_antisym(1,2,3,0) =  0.18125791;
    scf.eri_antisym(2,1,0,3) =  0.18125791; scf.eri_antisym(2,1,3,0) = -0.18125791;
    // VVVV
    scf.eri_antisym(2,3,2,3) =  0.69749535; scf.eri_antisym(2,3,3,2) = -0.69749535;
    scf.eri_antisym(3,2,2,3) = -0.69749535; scf.eri_antisym(3,2,3,2) =  0.69749535;

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

const double E_LCCD_REF = -0.03288443;  // cupyccx solver, H2/STO-3G R=1.4 bohr
const double E_CCD_REF  = -0.02141095;  // cupyccx solver, H2/STO-3G R=1.4 bohr

TEST(LCCD, H2_Converged) {
    auto scf  = make_h2_scf();
    auto opts = default_opts();
    cupyccx::CCD solver(scf, opts, "LCCD");
    auto result = solver.compute(0.0);
    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.e_corr, E_LCCD_REF, 1e-6);
}

TEST(LCCD, H2_IterCount) {
    auto scf  = make_h2_scf();
    auto opts = default_opts();
    cupyccx::CCD solver(scf, opts, "LCCD");
    auto result = solver.compute(0.0);
    EXPECT_GT(result.n_iter, 0);
    EXPECT_LE(result.n_iter, opts.max_iter);
}

TEST(CCD, H2_Converged) {
    auto scf  = make_h2_scf();
    auto opts = default_opts();
    cupyccx::CCD solver(scf, opts);
    auto result = solver.compute(0.0);
    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.e_corr, E_CCD_REF, 1e-6);
}

TEST(CCD, H2_T2Antisymmetry) {
    auto scf  = make_h2_scf();
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

TEST(CCD, EnergyNegative) {
    auto scf = make_h2_scf();
    auto opts = default_opts();
    cupyccx::CCD ccd(scf, opts);
    cupyccx::CCD lccd(scf, opts, "LCCD");
    auto r_ccd  = ccd.compute(0.0);
    auto r_lccd = lccd.compute(0.0);
    EXPECT_LT(r_ccd.e_corr,  0.0);
    EXPECT_LT(r_lccd.e_corr, 0.0);
}

TEST(CCD, CallbackFired) {
    auto scf = make_h2_scf();
    auto opts = default_opts();
    int  cb_count = 0;
    cupyccx::CCD solver(scf, opts);
    solver.set_callback([&](int, double, double, double) { ++cb_count; });
    solver.compute(0.0);
    EXPECT_GT(cb_count, 0);
}
