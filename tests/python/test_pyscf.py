"""
Integration tests that require PySCF and the compiled extension.
Run with:  pytest tests/python/test_pyscf.py -v
"""

import pytest
import numpy as np

try:
    import pyscf  # noqa: F401
    HAS_PYSCF = True
except ImportError:
    HAS_PYSCF = False

try:
    import cupyccx._cupyccx  # noqa: F401
    HAS_EXT = True
except ImportError:
    HAS_EXT = False

skip_pyscf = pytest.mark.skipif(not HAS_PYSCF, reason="PySCF not installed")
skip_both  = pytest.mark.skipif(
    not (HAS_PYSCF and HAS_EXT),
    reason="Requires PySCF and the compiled _cupyccx extension",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def h2_rhf():
    """Converged RHF on H2 / STO-3G — fast, used for end-to-end checks."""
    from pyscf import gto, scf
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    mf  = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    return mf


@pytest.fixture(scope="module")
def h2o_rhf():
    """Converged RHF on H2O / STO-3G — used for frozen-core and UHF tests."""
    from pyscf import gto, scf
    mol = gto.M(
        atom="O 0 0 0; H 0 0.757 0.586; H 0 -0.757 0.586",
        basis="sto-3g", unit="Angstrom", verbose=0,
    )
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    return mf


# ---------------------------------------------------------------------------
# SCFInputData extraction (PySCF required; extension optional)
# ---------------------------------------------------------------------------

@skip_pyscf
def test_prepare_returns_scfinputdata(h2_rhf):
    from cupyccx.scf_data import prepare_from_pyscf, SCFInputData
    data = prepare_from_pyscf(h2_rhf, verbose=False)
    assert isinstance(data, SCFInputData)


@skip_pyscf
def test_scf_energy_preserved(h2_rhf):
    from cupyccx.scf_data import prepare_from_pyscf
    data = prepare_from_pyscf(h2_rhf, verbose=False)
    assert abs(data.e_scf - h2_rhf.e_tot) < 1e-12


@skip_pyscf
def test_spin_orbital_counts_rhf(h2_rhf):
    """RHF H2 STO-3G: 1 spatial occ, 1 spatial vir → 2 spin-orb occ, 2 spin-orb vir."""
    from cupyccx.scf_data import prepare_from_pyscf
    data = prepare_from_pyscf(h2_rhf, verbose=False)
    assert data.n_occ == 2
    assert data.n_vir == 2
    assert data.reference == "RHF"


@skip_pyscf
def test_eri_shape(h2_rhf):
    from cupyccx.scf_data import prepare_from_pyscf
    data = prepare_from_pyscf(h2_rhf, verbose=False)
    n = data.n_mo
    assert data.eri_antisym.shape == (n, n, n, n)


@skip_pyscf
def test_eri_antisymmetry(h2_rhf):
    """<pq||rs> = -<qp||rs> = -<pq||sr>."""
    from cupyccx.scf_data import prepare_from_pyscf
    data = prepare_from_pyscf(h2_rhf, verbose=False)
    g = data.eri_antisym
    np.testing.assert_allclose(g, -g.transpose(1, 0, 2, 3), atol=1e-12)
    np.testing.assert_allclose(g, -g.transpose(0, 1, 3, 2), atol=1e-12)
    np.testing.assert_allclose(g,  g.transpose(2, 3, 0, 1), atol=1e-12)


@skip_pyscf
def test_fock_diagonal_canonical(h2_rhf):
    """For canonical RHF, the MO Fock matrix should be diagonal."""
    from cupyccx.scf_data import prepare_from_pyscf
    data = prepare_from_pyscf(h2_rhf, verbose=False)
    off_diag = data.fock - np.diag(np.diag(data.fock))
    assert np.max(np.abs(off_diag)) < 1e-10


@skip_pyscf
def test_eps_matches_fock_diagonal(h2_rhf):
    from cupyccx.scf_data import prepare_from_pyscf
    data = prepare_from_pyscf(h2_rhf, verbose=False)
    np.testing.assert_allclose(data.eps, np.diag(data.fock), atol=1e-10)


@skip_pyscf
def test_occupied_orbitals_first(h2_rhf):
    """eps[0:n_occ] should all be less than eps[n_occ:]."""
    from cupyccx.scf_data import prepare_from_pyscf
    data = prepare_from_pyscf(h2_rhf, verbose=False)
    assert np.all(data.eps[:data.n_occ] < data.eps[data.n_occ:])


@skip_pyscf
def test_frozen_reduces_active_space(h2o_rhf):
    """Freezing 1 core orbital reduces n_occ by 2 (alpha+beta)."""
    from cupyccx.scf_data import prepare_from_pyscf
    full   = prepare_from_pyscf(h2o_rhf, verbose=False)
    frozen = prepare_from_pyscf(h2o_rhf, frozen=1, verbose=False)
    assert frozen.n_occ == full.n_occ - 2
    assert frozen.n_vir == full.n_vir


@skip_pyscf
def test_frozen_invalid_raises(h2o_rhf):
    """Freezing a virtual orbital should raise ValueError."""
    from cupyccx.scf_data import prepare_from_pyscf
    import re
    with pytest.raises(ValueError, match="virtual"):
        prepare_from_pyscf(h2o_rhf, frozen=[999], verbose=False)


@skip_pyscf
def test_save_load_roundtrip(tmp_path, h2_rhf):
    from cupyccx.scf_data import prepare_from_pyscf, SCFInputData
    data = prepare_from_pyscf(h2_rhf, verbose=False)
    path = str(tmp_path / "h2.npz")
    data.save(path)
    data2 = SCFInputData.load(path)
    assert data2.n_occ == data.n_occ
    assert data2.n_vir == data.n_vir
    assert abs(data2.e_scf - data.e_scf) < 1e-15
    np.testing.assert_array_equal(data2.eps, data.eps)
    np.testing.assert_array_equal(data2.eri_antisym, data.eri_antisym)


@skip_pyscf
def test_unconverged_raises(h2_rhf):
    """prepare_from_pyscf should raise if SCF is not converged."""
    from pyscf import gto, scf
    from cupyccx.scf_data import prepare_from_pyscf
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    mf_bad = scf.RHF(mol)
    mf_bad.verbose = 0
    # Do NOT call mf_bad.kernel() — converged defaults to False
    with pytest.raises(RuntimeError, match="not converged"):
        prepare_from_pyscf(mf_bad, verbose=False)


# ---------------------------------------------------------------------------
# End-to-end CC calculations (both PySCF and extension required)
# ---------------------------------------------------------------------------

@skip_both
def test_ccd_from_scf_data(h2_rhf):
    """CCD via SCFInputData.from_scf_data() should converge."""
    from cupyccx.scf_data import prepare_from_pyscf
    from cupyccx import CCD
    data   = prepare_from_pyscf(h2_rhf, verbose=False)
    result = CCD.from_scf_data(data).compute(e_scf=data.e_scf)
    assert result.converged
    assert result.e_corr < 0


@skip_both
def test_run_from_pyscf_ccd(h2_rhf):
    from cupyccx.pyscf_interface import run_from_pyscf
    result = run_from_pyscf(h2_rhf, method="CCD", verbose=False)
    assert result.converged
    assert result.e_corr < 0


@skip_both
def test_run_from_pyscf_lccd(h2_rhf):
    from cupyccx.pyscf_interface import run_from_pyscf
    result = run_from_pyscf(h2_rhf, method="LCCD", verbose=False)
    assert result.converged
    assert result.e_corr < 0


@skip_both
def test_energy_independent_of_path(h2_rhf):
    """run_from_pyscf and CCD.from_scf_data should give the same energy."""
    from cupyccx.scf_data import prepare_from_pyscf
    from cupyccx import CCD
    from cupyccx.pyscf_interface import run_from_pyscf

    data     = prepare_from_pyscf(h2_rhf, verbose=False)
    r_direct = CCD.from_scf_data(data).compute(e_scf=data.e_scf)
    r_wrap   = run_from_pyscf(h2_rhf, method="CCD", verbose=False)

    assert abs(r_direct.e_corr - r_wrap.e_corr) < 1e-12


@skip_both
def test_h2o_ccd_frozen(h2o_rhf):
    """CCD on H2O/STO-3G with frozen oxygen 1s should converge."""
    from cupyccx.pyscf_interface import run_from_pyscf
    result = run_from_pyscf(h2o_rhf, method="CCD", frozen=1, verbose=False)
    assert result.converged
    assert result.e_corr < 0


# ---------------------------------------------------------------------------
# N2 / STO-3G reference energies
# Reference CCD:  -0.15939390011736 Ha  (PySCF CCSD with t1=0)
# Reference LCCD: -0.16290979       Ha  (PSI4)
# Reference HF:  -107.5000635015    Ha
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def n2_sto3g_rhf():
    """Converged RHF on N2 / STO-3G at experimental bond length (2.118 Bohr)."""
    from pyscf import gto, scf
    mol = gto.M(atom="N 0 0 0; N 0 0 2.118", basis="sto-3g", unit="Bohr", verbose=0)
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    return mf


@skip_pyscf
def test_n2_hf_energy(n2_sto3g_rhf):
    assert abs(n2_sto3g_rhf.e_tot - (-107.5000635015)) < 1e-6


@skip_both
def test_n2_ccd_energy(n2_sto3g_rhf):
    """CCD on N2/STO-3G should match PySCF's CCD (CCSD with t1=0) to 1e-5 Ha."""
    from cupyccx.pyscf_interface import run_from_pyscf
    from cupyccx.method import CCOptions
    opts = CCOptions(max_iter=200, conv_energy=1e-9, conv_amp=1e-8)
    result = run_from_pyscf(n2_sto3g_rhf, method="CCD", opts=opts, verbose=False)
    assert result.converged
    assert abs(result.e_corr - (-0.15939390)) < 1e-5


@skip_both
def test_n2_lccd_energy(n2_sto3g_rhf):
    """LCCD on N2/STO-3G should match PSI4 reference to 1e-5 Ha."""
    from cupyccx.pyscf_interface import run_from_pyscf
    from cupyccx.method import CCOptions
    opts = CCOptions(max_iter=200, conv_energy=1e-9, conv_amp=1e-8)
    result = run_from_pyscf(n2_sto3g_rhf, method="LCCD", opts=opts, verbose=False)
    assert result.converged
    assert abs(result.e_corr - (-0.16290979)) < 1e-5


@skip_both
def test_n2_pccd_alpha1_beta1_matches_ccd(n2_sto3g_rhf):
    """pCCD(alpha=1, beta=1) must recover full CCD on N2/STO-3G."""
    from cupyccx.pyscf_interface import run_from_pyscf
    from cupyccx.method import CCOptions
    opts = CCOptions(max_iter=200, conv_energy=1e-9, conv_amp=1e-8)
    r_ccd  = run_from_pyscf(n2_sto3g_rhf, method="CCD",  opts=opts, verbose=False)
    r_pccd = run_from_pyscf(n2_sto3g_rhf, method="pCCD", opts=opts, verbose=False,
                            alpha=1.0, beta=1.0)
    assert r_pccd.converged
    assert abs(r_pccd.e_corr - r_ccd.e_corr) < 1e-7


@skip_both
def test_n2_dcd_energy(n2_sto3g_rhf):
    """DCD on N2/STO-3G should match reference to 1e-6 Ha."""
    from cupyccx.pyscf_interface import run_from_pyscf
    from cupyccx.method import CCOptions
    opts = CCOptions(max_iter=200, conv_energy=1e-9, conv_amp=1e-8)
    result = run_from_pyscf(n2_sto3g_rhf, method="DCD", opts=opts, verbose=False)
    assert result.converged
    assert abs(result.e_corr - (-0.16839616)) < 1e-6
