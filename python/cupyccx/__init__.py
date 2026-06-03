"""
cupyccx — Coupled Cluster Doubles in C++/CUDA with Python bindings.

Quick-start
-----------
>>> from cupyccx import CCD, CCOptions
>>> result = CCD(n_occ, n_vir, eps, fock, eri).compute(e_scf=e_hf)
>>> print(f"E_corr = {result.e_corr:.10f} Ha")

PySCF interface
---------------
>>> from cupyccx.pyscf_interface import run_from_pyscf
>>> result = run_from_pyscf(mf, method="CCD")
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("cupyccx")
except PackageNotFoundError:
    __version__ = "0.0.0+dev"

# Try to import the compiled C++ extension
try:
    from cupyccx._cupyccx import CCOptions, CCResult, run_ccd, run_lccd, run_dcd, run_pccd, make_scf_data
    _HAS_EXTENSION = True
except ImportError:
    _HAS_EXTENSION = False

from cupyccx.method import CCD, LCCD, DCD, pCCD
from cupyccx.scf_data import SCFInputData, prepare_from_pyscf
from cupyccx.pyscf_interface import run_from_pyscf
from cupyccx.py_solver import PyCCD, PyLCCD, PyDCD, PypCCD

__all__ = [
    # C++ solvers
    "CCD",
    "LCCD",
    "DCD",
    "pCCD",
    "CCOptions",
    "CCResult",
    # Pure-Python solvers (debugging reference)
    "PyCCD",
    "PyLCCD",
    "PyDCD",
    "PypCCD",
    # Integral extraction
    "SCFInputData",
    "prepare_from_pyscf",
    # Convenience wrappers
    "run_from_pyscf",
    "run_ccd",
    "run_lccd",
    "run_dcd",
    "run_pccd",
]
