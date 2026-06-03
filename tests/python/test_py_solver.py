"""
Tests for the pure-Python CC solvers (py_solver.py).

All CPU tests run without GPU hardware. The GPU test is skipped if CuPy is absent.
Reference energies for N2/STO-3G (N-N bond = 2.118 Bohr) match the C++ solvers
and external codes (PSI4 for LCCD, PySCF CCSD with t1=0 for CCD and DCD).
"""

import numpy as np
import pytest

pytest.importorskip("pyscf", reason="pyscf required for reference data")

from pyscf import gto, scf as pyscf_scf

from cupyccx.scf_data import prepare_from_pyscf
from cupyccx.method import CCOptions, CCD
from cupyccx.py_solver import PyCCD, PyLCCD, PyDCD, PypCCD


def _cupy_available():
    try:
        import cupy  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def n2_data():
    """Converged RHF on N2 / STO-3G at experimental bond length (2.118 Bohr)."""
    mol = gto.M(atom="N 0 0 0; N 0 0 2.118", basis="sto-3g", unit="Bohr", verbose=0)
    mf = pyscf_scf.RHF(mol).run()
    return prepare_from_pyscf(mf, verbose=False)


# ---------------------------------------------------------------------------
# N2 / STO-3G energy tests
# ---------------------------------------------------------------------------
# Reference CCD:  -0.15939390 Ha  (PySCF CCSD with t1=0)
# Reference LCCD: -0.16290979 Ha  (PSI4)
# Reference DCD:  -0.16839616 Ha  (C++ cupyccx reference)

def test_pyccd_energy(n2_data):
    r = PyCCD.from_scf_data(n2_data).compute(n2_data.e_scf)
    assert r.converged, "PyCCD did not converge"
    assert abs(r.e_corr - (-0.15939390)) < 1e-6


def test_pylccd_energy(n2_data):
    r = PyLCCD.from_scf_data(n2_data).compute(n2_data.e_scf)
    assert r.converged, "PyLCCD did not converge"
    assert abs(r.e_corr - (-0.16290979)) < 1e-5


def test_pydcd_energy(n2_data):
    r = PyDCD.from_scf_data(n2_data).compute(n2_data.e_scf)
    assert r.converged, "PyDCD did not converge"
    assert abs(r.e_corr - (-0.16839616)) < 1e-6


def test_pypccd_alpha1_beta1_matches_pyccd(n2_data):
    """pCCD(alpha=1, beta=1) must recover full CCD — verifies diagram coefficients."""
    r_ccd = PyCCD.from_scf_data(n2_data).compute(n2_data.e_scf)
    r_pccd = PypCCD.from_scf_data(n2_data, alpha=1.0, beta=1.0).compute(n2_data.e_scf)
    assert r_ccd.converged and r_pccd.converged
    assert abs(r_ccd.e_corr - r_pccd.e_corr) < 1e-8


# ---------------------------------------------------------------------------
# T2 amplitude cross-check against C++ extension
# ---------------------------------------------------------------------------

def test_pyccd_t2_matches_cpp(n2_data):
    """PyCCD T2 amplitudes match C++ CCD to 1e-8 (roundtrip regression)."""
    pytest.importorskip("cupyccx._cupyccx")
    r_py = PyCCD.from_scf_data(n2_data).compute(n2_data.e_scf)
    r_cpp = CCD.from_scf_data(n2_data).compute(n2_data.e_scf)
    assert r_py.t2 is not None and r_cpp.t2 is not None
    np.testing.assert_allclose(r_py.t2, r_cpp.t2, atol=1e-8,
                               err_msg="PyCCD T2 deviates from C++ CCD")


# ---------------------------------------------------------------------------
# GPU test (skipped if CuPy is absent)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _cupy_available(), reason="CuPy not installed")
def test_pyccd_gpu_matches_cpu(n2_data):
    """GPU (CuPy) path must give the same energy as CPU (NumPy) path."""
    r_cpu = PyCCD.from_scf_data(n2_data, opts=CCOptions(use_gpu=False)).compute(n2_data.e_scf)
    r_gpu = PyCCD.from_scf_data(n2_data, opts=CCOptions(use_gpu=True)).compute(n2_data.e_scf)
    assert abs(r_cpu.e_corr - r_gpu.e_corr) < 1e-10
