#include <gtest/gtest.h>
#include "cupyccx/types.hpp"
#include "cupyccx/integrals.hpp"
#include <cmath>

// ---------------------------------------------------------------------------
// Build a tiny H2 / minimal-basis SCFData by hand for deterministic tests.
// Numbers match the analytic STO-3G H2 result at R=1.4 bohr.
//
// STO-3G H2:  n_occ=2 (alpha+beta), n_vir=2, n_mo=4
// Orbital energies: eps = {-0.5780, -0.5780, +0.6702, +0.6702}   (spin-orb)
// Reference CCD correlation energy: -0.04222... Ha  (MP2 limit for 2e)
// ---------------------------------------------------------------------------

namespace {

cupyccx::SCFData make_h2_scf() {
    // Minimal model: 1 occupied spatial orbital, 1 virtual => 2 spin-orbs each
    // eps (spatial): occ=-0.5780, vir=+0.6702
    // For spin-orbitals: alpha/beta are degenerate
    const double eps_occ = -0.57827;
    const double eps_vir =  0.67012;
    const int n_occ = 2, n_vir = 2, n_mo = 4;

    cupyccx::SCFData scf;
    scf.n_occ = n_occ;
    scf.n_vir = n_vir;
    scf.n_mo  = n_mo;
    scf.eps.resize(n_mo);
    scf.eps << eps_occ, eps_occ, eps_vir, eps_vir;

    scf.fock = cupyccx::Matrix::Zero(n_mo, n_mo);
    scf.fock(0,0) = eps_occ; scf.fock(1,1) = eps_occ;
    scf.fock(2,2) = eps_vir; scf.fock(3,3) = eps_vir;

    // Two-electron integrals — use the unique value for H2 STO-3G
    // <alpha_occ, beta_occ | alpha_vir, beta_vir> (chemist) = 0.20636
    // Antisymmetrized <pq||rs> = <pq|rs> - <pq|sr>
    const double K = 0.20636;  // spatial exchange integral
    scf.eri_antisym = cupyccx::Tensor4(n_mo, n_mo, n_mo, n_mo);

    // Fill the occ-occ-vir-vir block (p=0,1 occ; r,s=2,3 vir)
    // In spin-orbital physicist notation:
    //   <0 1 || 2 3> = +K  (alpha_occ beta_occ || alpha_vir beta_vir)
    //   <1 0 || 3 2> = +K
    //   <0 1 || 3 2> = -K  (exchanged last two)
    //   <1 0 || 2 3> = -K
    // and hermitian conjugates
    scf.eri_antisym(0,1,2,3) =  K;
    scf.eri_antisym(1,0,3,2) =  K;
    scf.eri_antisym(0,1,3,2) = -K;
    scf.eri_antisym(1,0,2,3) = -K;
    // conjugates
    scf.eri_antisym(2,3,0,1) =  K;
    scf.eri_antisym(3,2,1,0) =  K;
    scf.eri_antisym(3,2,0,1) = -K;
    scf.eri_antisym(2,3,1,0) = -K;

    return scf;
}

}  // namespace

TEST(Integrals, ValidateSCF) {
    auto scf = make_h2_scf();
    EXPECT_NO_THROW(cupyccx::validate_scf(scf));
}

TEST(Integrals, SliceOOVV_Antisymmetry) {
    auto scf  = make_h2_scf();
    auto oovv = cupyccx::slice_oovv(scf);

    // Antisymmetry: <ij||ab> = -<ji||ab> = -<ij||ba>
    for (int i = 0; i < scf.n_occ; ++i)
    for (int j = 0; j < scf.n_occ; ++j)
    for (int a = 0; a < scf.n_vir; ++a)
    for (int b = 0; b < scf.n_vir; ++b) {
        EXPECT_NEAR( oovv(i,j,a,b), -oovv(j,i,a,b), 1e-12);
        EXPECT_NEAR( oovv(i,j,a,b), -oovv(i,j,b,a), 1e-12);
    }
}

TEST(Integrals, DenominatorSign) {
    auto scf = make_h2_scf();
    auto D2  = cupyccx::make_D2(scf);
    // For any occ-occ-vir-vir element, eps_i + eps_j < eps_a + eps_b
    // => D_{ij}^{ab} < 0
    for (std::size_t n = 0; n < D2.data.size(); ++n)
        EXPECT_LT(D2.data[n], 0.0);
}

TEST(Integrals, MP2Energy) {
    // E_MP2 = (1/4) sum_{ijab} |<ij||ab>|^2 / D^{ab}_{ij}
    auto scf  = make_h2_scf();
    auto oovv = cupyccx::slice_oovv(scf);
    auto D2   = cupyccx::make_D2(scf);

    double e_mp2 = 0.0;
    for (std::size_t n = 0; n < oovv.data.size(); ++n)
        e_mp2 += oovv.data[n] * oovv.data[n] / D2.data[n];
    e_mp2 *= 0.25;

    // For 2-electron systems, MP2 == FCI correlation energy in this basis
    EXPECT_NEAR(e_mp2, -0.04222, 5e-4);
}
