#include <gtest/gtest.h>
#include "cupyccx/types.hpp"
#include "cupyccx/integrals.hpp"
#include <cmath>

// ---------------------------------------------------------------------------
// Build the exact H2 / STO-3G spin-orbital SCFData from PySCF.
// Geometry: H-H at R = 1.4 bohr.
//
// Spin-orbital ordering: 0 = occ_α, 1 = occ_β, 2 = vir_α, 3 = vir_β
//
// Reference energies (PySCF):
//   E_corr(MP2)  = -0.01315787 Ha
//   E_corr(CCSD) = -0.02056178 Ha   (= CCD for a 2-electron system)
// ---------------------------------------------------------------------------

namespace {

cupyccx::SCFData make_h2_scf() {
    const double eps_occ = -0.57820298;
    const double eps_vir =  0.67026777;
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

    // Complete antisymmetrized spin-orbital ERIs <pq||rs> from PySCF.
    // Spin-orbital index: 0=(occ,α) 1=(occ,β) 2=(vir,α) 3=(vir,β)
    scf.eri_antisym = cupyccx::Tensor4(n_mo, n_mo, n_mo, n_mo);

    // OOOO block
    scf.eri_antisym(0,1,0,1) =  0.67459408; scf.eri_antisym(0,1,1,0) = -0.67459408;
    scf.eri_antisym(1,0,0,1) = -0.67459408; scf.eri_antisym(1,0,1,0) =  0.67459408;

    // OOVV block
    scf.eri_antisym(0,1,2,3) =  0.18125791; scf.eri_antisym(0,1,3,2) = -0.18125791;
    scf.eri_antisym(1,0,2,3) = -0.18125791; scf.eri_antisym(1,0,3,2) =  0.18125791;
    // VVOO (hermitian conjugates)
    scf.eri_antisym(2,3,0,1) =  0.18125791; scf.eri_antisym(2,3,1,0) = -0.18125791;
    scf.eri_antisym(3,2,0,1) = -0.18125791; scf.eri_antisym(3,2,1,0) =  0.18125791;

    // OVOV block (same-spin pairs)
    scf.eri_antisym(0,2,0,2) =  0.48230608; scf.eri_antisym(0,2,2,0) = -0.48230608;
    scf.eri_antisym(2,0,0,2) = -0.48230608; scf.eri_antisym(2,0,2,0) =  0.48230608;
    scf.eri_antisym(1,3,1,3) =  0.48230608; scf.eri_antisym(1,3,3,1) = -0.48230608;
    scf.eri_antisym(3,1,1,3) = -0.48230608; scf.eri_antisym(3,1,3,1) =  0.48230608;

    // OVOV block (opposite-spin pairs)
    scf.eri_antisym(0,3,0,3) =  0.66356399; scf.eri_antisym(0,3,3,0) = -0.66356399;
    scf.eri_antisym(3,0,0,3) = -0.66356399; scf.eri_antisym(3,0,3,0) =  0.66356399;
    scf.eri_antisym(1,2,1,2) =  0.66356399; scf.eri_antisym(1,2,2,1) = -0.66356399;
    scf.eri_antisym(2,1,1,2) = -0.66356399; scf.eri_antisym(2,1,2,1) =  0.66356399;

    // OVOV opposite-spin cross terms (mix occ/vir α and β)
    scf.eri_antisym(0,3,1,2) = -0.18125791; scf.eri_antisym(0,3,2,1) =  0.18125791;
    scf.eri_antisym(3,0,1,2) =  0.18125791; scf.eri_antisym(3,0,2,1) = -0.18125791;
    scf.eri_antisym(1,2,0,3) = -0.18125791; scf.eri_antisym(1,2,3,0) =  0.18125791;
    scf.eri_antisym(2,1,0,3) =  0.18125791; scf.eri_antisym(2,1,3,0) = -0.18125791;

    // VVVV block
    scf.eri_antisym(2,3,2,3) =  0.69749535; scf.eri_antisym(2,3,3,2) = -0.69749535;
    scf.eri_antisym(3,2,2,3) = -0.69749535; scf.eri_antisym(3,2,3,2) =  0.69749535;

    return scf;
}

}  // namespace

// Reference energies (PySCF, H2/STO-3G, R=1.4 bohr)
const double E_MP2_REF  = -0.01315787;  // PySCF MP2, H2/STO-3G R=1.4 bohr

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
    // For any occ-occ-vir-vir element, eps_i + eps_j < eps_a + eps_b => D < 0
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

    EXPECT_NEAR(e_mp2, E_MP2_REF, 1e-6);
}
