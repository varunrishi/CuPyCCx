"""
High-level Python wrappers around the C++ CCD/LCCD solvers.

These classes accept numpy arrays directly and present a clean API that mirrors
PySCF conventions where possible.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional

try:
    import cupyccx._cupyccx as _ext
    _HAS_EXT = True
except ImportError:
    _HAS_EXT = False

# Avoid a hard import cycle: scf_data imports nothing from method,
# but we use TYPE_CHECKING so the annotation is available at type-check
# time without importing at runtime.
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from cupyccx.scf_data import SCFInputData


@dataclass
class CCOptions:
    """Options forwarded to the C++ solver."""
    max_iter: int    = 100
    conv_energy: float = 1e-9
    conv_amp: float    = 1e-7
    use_diis: bool     = True
    diis_size: int     = 6
    use_gpu: bool      = False

    def _to_ext(self):
        if not _HAS_EXT:
            raise ImportError("cupyccx C++ extension not found. Build the project first.")
        opts = _ext.CCOptions()
        opts.max_iter    = self.max_iter
        opts.conv_energy = self.conv_energy
        opts.conv_amp    = self.conv_amp
        opts.use_diis    = self.use_diis
        opts.diis_size   = self.diis_size
        opts.use_gpu     = self.use_gpu
        return opts


@dataclass
class CCResult:
    """Mirrors _ext.CCResult but is constructable in pure Python."""
    e_corr: float
    e_total: float
    n_iter: int
    converged: bool
    t2: Optional[np.ndarray] = field(default=None, repr=False)

    @classmethod
    def from_ext(cls, ext_result) -> "CCResult":
        return cls(
            e_corr=ext_result.e_corr,
            e_total=ext_result.e_total,
            n_iter=ext_result.n_iter,
            converged=ext_result.converged,
            t2=np.array(ext_result.t2),
        )


class _CCDSolver:
    """Base class for CCD-family solvers."""

    _method_name: str = "CCD"

    def __init__(self,
                 n_occ: int,
                 n_vir: int,
                 eps: np.ndarray,
                 fock: np.ndarray,
                 eri_antisym: np.ndarray,
                 opts: Optional[CCOptions] = None):
        """
        Parameters
        ----------
        n_occ : int
            Number of occupied spin-orbitals.
        n_vir : int
            Number of virtual spin-orbitals.
        eps : ndarray, shape (n_occ + n_vir,)
            Orbital energies (occupied first).
        fock : ndarray, shape (n_mo, n_mo)
            Fock matrix in the MO basis.
        eri_antisym : ndarray, shape (n_mo, n_mo, n_mo, n_mo)
            Antisymmetrized ERIs <pq||rs> in physicist notation.
        opts : CCOptions, optional
            Solver options (defaults used if None).
        """
        self.n_occ = n_occ
        self.n_vir = n_vir
        self.eps    = np.ascontiguousarray(eps, dtype=np.float64)
        self.fock   = np.ascontiguousarray(fock, dtype=np.float64)
        self.eri    = np.ascontiguousarray(eri_antisym, dtype=np.float64)
        self.opts   = opts or CCOptions()

    @classmethod
    def from_scf_data(cls, data: "SCFInputData", opts: Optional[CCOptions] = None) -> "_CCDSolver":
        """
        Construct the solver from an :class:`~cupyccx.scf_data.SCFInputData` object.

        This is the preferred constructor when using ``prepare_from_pyscf``:

            data   = prepare_from_pyscf(mf)
            result = CCD.from_scf_data(data).compute(e_scf=data.e_scf, verbose=True)
        """
        return cls(
            data.n_occ, data.n_vir,
            data.eps, data.fock, data.eri_antisym,
            opts=opts,
        )

    def compute(self,
                e_scf: float = 0.0,
                callback: Optional[Callable] = None,
                verbose: bool = False) -> CCResult:
        """
        Run the CC iterations.

        Parameters
        ----------
        e_scf : float
            SCF (reference) energy added to the correlation energy.
        callback : callable, optional
            Called as ``callback(iter, e_corr, delta_e, rms_amp)`` each macro-iteration.
        verbose : bool
            If True, print iteration table to stdout.

        Returns
        -------
        CCResult
        """
        if not _HAS_EXT:
            raise ImportError("cupyccx C++ extension not found. Build the project first.")

        if verbose:
            print(f"\n{'':=<60}")
            print(f"  {self._method_name} Calculation")
            print(f"{'':=<60}")
            print(f"  {'Iter':>4}  {'E_corr':>18}  {'dE':>12}  {'RMS amp':>12}")
            print(f"  {'-'*4}  {'-'*18}  {'-'*12}  {'-'*12}")

        def _cb(it, e, de, rms):
            if verbose:
                print(f"  {it:>4}  {e:>18.10f}  {de:>12.4e}  {rms:>12.4e}")
            if callback is not None:
                callback(it, e, de, rms)

        run_fn = _ext.run_lccd if self._method_name == "LCCD" else _ext.run_ccd
        ext_opts = self.opts._to_ext()
        ext_opts.method = self._method_name

        ext_result = run_fn(
            self.n_occ, self.n_vir,
            self.eps, self.fock, self.eri,
            ext_opts, e_scf, _cb,
        )

        result = CCResult.from_ext(ext_result)

        if verbose:
            status = "CONVERGED" if result.converged else "NOT CONVERGED"
            print(f"{'':=<60}")
            print(f"  {status} in {result.n_iter} iterations")
            print(f"  E_corr  = {result.e_corr:20.12f} Ha")
            print(f"  E_total = {result.e_total:20.12f} Ha")
            print(f"{'':=<60}\n")

        return result


class CCD(_CCDSolver):
    """Full Coupled Cluster Doubles (CCD) solver.

    Includes all quadratic T2*T2 contributions via the W_oooo and W_vvvv
    intermediates.  Scales as O(N^6) with system size.
    """
    _method_name = "CCD"


class LCCD(_CCDSolver):
    """Linearized CCD (LCCD) solver.

    Drops the quadratic T2*T2 terms.  Useful as a perturbative check and
    faster than full CCD (same O(N^6) scaling but smaller prefactor).
    """
    _method_name = "LCCD"
