"""
PySCF convenience wrappers for cupyccx.

``run_from_pyscf`` is a one-liner that calls ``prepare_from_pyscf`` internally
and then runs the requested CC method.  If you need more control — inspecting
integrals, saving them to disk, or running multiple methods on the same
integral set — call ``prepare_from_pyscf`` directly:

    from cupyccx.scf_data import prepare_from_pyscf, SCFInputData
    from cupyccx import CCD, CCOptions

    data = prepare_from_pyscf(mf, frozen=1, verbose=True)
    data.save("integrals.npz")           # save for later

    result = CCD(data.n_occ, data.n_vir,
                 data.eps, data.fock, data.eri_antisym,
                 opts=CCOptions(use_gpu=True)).compute(e_scf=data.e_scf,
                                                       verbose=True)
"""

from __future__ import annotations

from typing import Optional, Union

from cupyccx.method import CCD, LCCD, DCD, pCCD, CCOptions, CCResult
from cupyccx.scf_data import SCFInputData, prepare_from_pyscf  # re-exported

_METHODS = {"CCD": CCD, "LCCD": LCCD, "DCD": DCD, "pCCD": pCCD}


def run_from_pyscf(
    mf,
    method: str = "CCD",
    opts: Optional[CCOptions] = None,
    frozen: Union[int, list, None] = None,
    verbose: bool = True,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> CCResult:
    """
    Run CCD or LCCD from a converged PySCF RHF or UHF object.

    This is a convenience wrapper.  It calls :func:`~cupyccx.scf_data.prepare_from_pyscf`
    to extract spin-orbital integrals and then runs the requested method.

    Parameters
    ----------
    mf : pyscf.scf.RHF | pyscf.scf.UHF
        Converged mean-field object.
    method : str
        ``"CCD"`` (default), ``"LCCD"``, ``"DCD"``, or ``"pCCD"``.
    opts : CCOptions, optional
        Solver options (convergence thresholds, DIIS, GPU flag).
    frozen : int or list of int, optional
        Spatial orbital indices to freeze (exclude from correlation).
        ``frozen=1`` freezes the lowest spatial MO.
    verbose : bool
        Print the iteration table.
    alpha : float
        pCCD alpha parameter (ignored for CCD/LCCD). Default 1.0.
    beta : float
        pCCD beta parameter (ignored for CCD/LCCD). Default 1.0.

    Returns
    -------
    CCResult

    Examples
    --------
    >>> from pyscf import gto, scf
    >>> from cupyccx.pyscf_interface import run_from_pyscf
    >>> mol = gto.M(atom="O 0 0 0; H 0 0.96 0; H 0.76 -0.48 0", basis="cc-pVDZ")
    >>> mf  = scf.RHF(mol).run()
    >>> result = run_from_pyscf(mf, method="CCD", verbose=True)
    >>> print(f"CCD E_corr = {result.e_corr:.10f} Ha")
    """
    if method not in _METHODS:
        raise ValueError(f"Unknown method {method!r}. Choose from {list(_METHODS)}")

    data = prepare_from_pyscf(mf, frozen=frozen, verbose=verbose)

    if method == "pCCD":
        solver = pCCD(
            data.n_occ, data.n_vir,
            data.eps, data.fock, data.eri_antisym,
            alpha=alpha, beta=beta, opts=opts or CCOptions(),
        )
    else:
        solver_cls = _METHODS[method]
        solver = solver_cls(
            data.n_occ, data.n_vir,
            data.eps, data.fock, data.eri_antisym,
            opts=opts or CCOptions(),
        )
    return solver.compute(e_scf=data.e_scf, verbose=verbose)
