"""
Extract SCF quantities from a converged PySCF mean-field object and package
them for use with cupyccx solvers.

The central function is ``prepare_from_pyscf``.  It returns an ``SCFInputData``
dataclass that holds numpy arrays in spin-orbital basis (occupied-first ordering)
and can be:

  - Passed directly to ``CCD`` / ``LCCD`` solvers
  - Inspected, saved to disk (``np.savez``), or fed into custom post-HF code
  - Constructed manually without PySCF at all (just fill the fields)

Spin-orbital ordering convention
---------------------------------
Block layout before reordering:   α₀ α₁ … α_{nmo-1}  β₀ β₁ … β_{nmo-1}
Final occupied-first layout:      α_occ | β_occ | α_vir | β_vir

All arrays follow physicist ERIs: <pq|rs> = ∫ φ_p*(1)φ_q*(2) r₁₂⁻¹ φ_r(1)φ_s(2)
Antisymmetrized:                  <pq||rs> = <pq|rs> − <pq|sr>
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Public data container
# ---------------------------------------------------------------------------

@dataclass
class SCFInputData:
    """
    All quantities needed for a cupyccx correlated calculation.

    Arrays are in **spin-orbital** basis, **occupied-first** ordering.
    Indices 0..n_occ-1 are occupied; n_occ..n_mo-1 are virtual.

    Attributes
    ----------
    n_occ : int
        Number of occupied spin-orbitals (correlated, i.e. after freezing).
    n_vir : int
        Number of virtual spin-orbitals.
    e_scf : float
        Total SCF energy in Hartree.
    eps : ndarray, shape (n_mo,)
        Orbital energies, occ first.
    fock : ndarray, shape (n_mo, n_mo)
        Fock matrix in MO spin-orbital basis.  Diagonal for canonical orbitals.
    eri_antisym : ndarray, shape (n_mo, n_mo, n_mo, n_mo)
        Antisymmetrized two-electron integrals <pq||rs> = <pq|rs> − <pq|sr>
        in physicist notation.
    reference : str
        "RHF" or "UHF".
    """

    n_occ: int
    n_vir: int
    e_scf: float
    eps: np.ndarray
    fock: np.ndarray
    eri_antisym: np.ndarray
    reference: str = "RHF"

    @property
    def n_mo(self) -> int:
        return self.n_occ + self.n_vir

    def __repr__(self) -> str:
        return (
            f"SCFInputData(reference={self.reference!r}, "
            f"n_occ={self.n_occ}, n_vir={self.n_vir}, "
            f"n_mo={self.n_mo}, e_scf={self.e_scf:.10f})"
        )

    # ------------------------------------------------------------------
    # Convenience I/O
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save all arrays to a compressed ``.npz`` file."""
        np.savez_compressed(
            path,
            n_occ=np.array(self.n_occ),
            n_vir=np.array(self.n_vir),
            e_scf=np.array(self.e_scf),
            eps=self.eps,
            fock=self.fock,
            eri_antisym=self.eri_antisym,
            reference=np.array(self.reference),
        )

    @classmethod
    def load(cls, path: str) -> "SCFInputData":
        """Load from a ``.npz`` file previously saved with :meth:`save`."""
        d = np.load(path, allow_pickle=False)
        return cls(
            n_occ=int(d["n_occ"]),
            n_vir=int(d["n_vir"]),
            e_scf=float(d["e_scf"]),
            eps=d["eps"],
            fock=d["fock"],
            eri_antisym=d["eri_antisym"],
            reference=str(d["reference"]),
        )


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def prepare_from_pyscf(
    mf,
    frozen: Union[int, list, None] = None,
    verbose: bool = True,
) -> SCFInputData:
    """
    Extract all quantities required by cupyccx from a converged PySCF SCF object.

    Supports RHF and UHF references.  Returns an :class:`SCFInputData` that can
    be passed directly to :class:`~cupyccx.method.CCD` / :class:`~cupyccx.method.LCCD`,
    or saved to disk and loaded in a later session.

    Parameters
    ----------
    mf : pyscf.scf.RHF | pyscf.scf.UHF
        Converged mean-field object.  Must have ``mf.converged == True``.
    frozen : int or list of int, optional
        Spatial orbital indices to exclude from the correlation space.
        ``frozen=2`` freezes orbitals 0 and 1 (lowest two).
        ``frozen=[0, 1]`` is equivalent.
        Only occupied orbitals may be frozen.
    verbose : bool
        Print a one-line summary of what was extracted.

    Returns
    -------
    SCFInputData

    Examples
    --------
    Basic usage::

        from pyscf import gto, scf
        from cupyccx.scf_data import prepare_from_pyscf
        from cupyccx import CCD

        mol = gto.M(atom="O 0 0 0; H 0 0.96 0; H 0.76 -0.48 0", basis="cc-pVDZ")
        mf  = scf.RHF(mol).run()

        data = prepare_from_pyscf(mf)
        print(data)                        # SCFInputData(...)
        print(data.eps)                    # orbital energies
        print(data.eri_antisym.shape)      # (n_mo, n_mo, n_mo, n_mo)

        result = CCD(data.n_occ, data.n_vir,
                     data.eps, data.fock, data.eri_antisym).compute(e_scf=data.e_scf)

    Save and reload integrals (avoids re-running PySCF)::

        data.save("h2o_ccpvdz.npz")
        data2 = SCFInputData.load("h2o_ccpvdz.npz")

    Frozen core::

        data = prepare_from_pyscf(mf, frozen=1)  # freeze 1s core of oxygen
    """
    try:
        from pyscf import scf as pyscf_scf
    except ImportError as exc:
        raise ImportError(
            "PySCF is required for prepare_from_pyscf. "
            "Install with: pip install pyscf"
        ) from exc

    if not mf.converged:
        raise RuntimeError(
            "The PySCF SCF object has not converged. "
            "Call mf.kernel() / mf.run() before extracting integrals."
        )

    # Detect reference type robustly (handles GHF subclasses too)
    is_uhf = isinstance(mf, pyscf_scf.uhf.UHF)
    is_rhf = isinstance(mf, pyscf_scf.rhf.RHF) and not is_uhf

    if is_rhf:
        return _prepare_rhf(mf, frozen, verbose)
    if is_uhf:
        return _prepare_uhf(mf, frozen, verbose)

    raise TypeError(
        f"mf must be an RHF or UHF object, got {type(mf).__name__}. "
        "GHF / ROHF are not yet supported."
    )


# ---------------------------------------------------------------------------
# RHF extraction
# ---------------------------------------------------------------------------

def _prepare_rhf(mf, frozen, verbose: bool) -> SCFInputData:
    from pyscf import ao2mo

    mol       = mf.mol
    mo_coeff  = np.asarray(mf.mo_coeff,  dtype=np.float64)  # (nao, nmo)
    mo_energy = np.asarray(mf.mo_energy, dtype=np.float64)  # (nmo,)
    mo_occ    = np.asarray(mf.mo_occ,    dtype=np.float64)  # (nmo,)
    nmo       = mo_coeff.shape[1]
    n_occ_s   = int(np.sum(mo_occ > 0))                     # spatial occ count

    active_occ_s, active_vir_s = _resolve_frozen(nmo, n_occ_s, frozen)
    n_act_occ = len(active_occ_s)
    n_act_vir = len(active_vir_s)

    # Combined active spatial index list (occ first, then vir)
    active_s = active_occ_s + active_vir_s
    na = len(active_s)  # total active spatial MOs

    # ---- Fock matrix: AO -> active MO -----------------------------------
    # get_fock() returns the AO Fock matrix, correct for any orbital set.
    # For converged canonical RHF, F_MO is diagonal = diag(mo_energy).
    fock_ao = np.asarray(mf.get_fock(), dtype=np.float64)
    mo_act  = mo_coeff[:, active_s]                          # (nao, na)
    fock_mo = mo_act.T @ fock_ao @ mo_act                    # (na, na)

    eps_act = mo_energy[active_s]                            # (na,)

    # ---- AO -> MO ERI transform (only active MOs) -----------------------
    # ao2mo.kernel returns (na*(na+1)/2)^2 in compact form; compact=False
    # gives the full na^4 array.
    eri_mo = ao2mo.kernel(mol, mo_act, compact=False).reshape(na, na, na, na)

    # Convert chemist (pq|rs) → physicist <pq|rs>:
    #   chemist (pq|rs) = ∫ φ_p(1)φ_q(1) r₁₂⁻¹ φ_r(2)φ_s(2) d1 d2
    #   physicist <pr|qs> = chemist (pq|rs)  →  <pq|rs> = chemist (pr|qs)
    eri_phys = eri_mo.transpose(0, 2, 1, 3)                  # (na, na, na, na)

    # ---- Spin-orbital expansion (fully vectorized, no Python loops) -----
    #
    # For RHF the spatial MO coefficients are the same for alpha and beta.
    # The spin-orbital ERI is non-zero only when spin(p)==spin(r) AND spin(q)==spin(s):
    #
    #   <p_α q_α | r_α s_α> = <pq|rs>  (αα|αα block)
    #   <p_β q_β | r_β s_β> = <pq|rs>  (ββ|ββ block)
    #   <p_α q_β | r_α s_β> = <pq|rs>  (αβ|αβ block)
    #   <p_β q_α | r_β s_α> = <pq|rs>  (βα|βα block)
    #
    # Using block ordering: α indices = 0..na-1, β indices = na..2*na-1
    n_so = 2 * na
    eri_so = np.zeros((n_so, n_so, n_so, n_so))
    eri_so[:na, :na, :na, :na] = eri_phys   # αα|αα
    eri_so[na:, na:, na:, na:] = eri_phys   # ββ|ββ
    eri_so[:na, na:, :na, na:] = eri_phys   # αβ|αβ
    eri_so[na:, :na, na:, :na] = eri_phys   # βα|βα
    eri_antisym = eri_so - eri_so.transpose(0, 1, 3, 2)      # <pq||rs>

    # ---- Spin-orbital Fock & eps ----------------------------------------
    # Block layout: [α_occ | α_vir | β_occ | β_vir] initially (mirrors eri_so)
    fock_so = np.zeros((n_so, n_so))
    fock_so[:na, :na] = fock_mo   # α block
    fock_so[na:, na:] = fock_mo   # β block (same coefficients for RHF)

    eps_so = np.zeros(n_so)
    eps_so[:na] = eps_act
    eps_so[na:] = eps_act

    # ---- Reorder: occupied first, then virtual --------------------------
    # Within block ordering, occupied spatial indices are 0..n_act_occ-1
    # (because we built active_s = occ + vir).
    alpha_occ = list(range(n_act_occ))
    beta_occ  = list(range(na, na + n_act_occ))
    alpha_vir = list(range(n_act_occ, na))
    beta_vir  = list(range(na + n_act_occ, n_so))
    order = alpha_occ + beta_occ + alpha_vir + beta_vir

    eps_out  = eps_so[order]
    fock_out = fock_so[np.ix_(order, order)]
    eri_out  = eri_antisym[np.ix_(order, order, order, order)]

    n_occ_so = 2 * n_act_occ
    n_vir_so = 2 * n_act_vir

    if verbose:
        _print_summary(
            mol, "RHF", n_occ_so, n_vir_so, mf.e_tot, frozen,
            n_frozen=n_occ_s - n_act_occ,
        )

    return SCFInputData(
        n_occ=n_occ_so,
        n_vir=n_vir_so,
        e_scf=float(mf.e_tot),
        eps=eps_out,
        fock=fock_out,
        eri_antisym=eri_out,
        reference="RHF",
    )


# ---------------------------------------------------------------------------
# UHF extraction
# ---------------------------------------------------------------------------

def _prepare_uhf(mf, frozen, verbose: bool) -> SCFInputData:
    from pyscf import ao2mo

    mol = mf.mol

    # For UHF, mo_coeff/mo_energy/mo_occ have shape (2, nao/nmo, nmo)
    mo_coeff_a  = np.asarray(mf.mo_coeff[0],  dtype=np.float64)  # (nao, nmo_a)
    mo_coeff_b  = np.asarray(mf.mo_coeff[1],  dtype=np.float64)  # (nao, nmo_b)
    mo_energy_a = np.asarray(mf.mo_energy[0], dtype=np.float64)
    mo_energy_b = np.asarray(mf.mo_energy[1], dtype=np.float64)
    mo_occ_a    = np.asarray(mf.mo_occ[0],    dtype=np.float64)
    mo_occ_b    = np.asarray(mf.mo_occ[1],    dtype=np.float64)

    nmo_a, nmo_b = mo_coeff_a.shape[1], mo_coeff_b.shape[1]
    n_occ_a = int(np.sum(mo_occ_a > 0))
    n_occ_b = int(np.sum(mo_occ_b > 0))

    # Resolve frozen separately for alpha and beta
    # For simplicity, frozen applies equally to both spins
    act_occ_a, act_vir_a = _resolve_frozen(nmo_a, n_occ_a, frozen)
    act_occ_b, act_vir_b = _resolve_frozen(nmo_b, n_occ_b, frozen)

    active_a = act_occ_a + act_vir_a
    active_b = act_occ_b + act_vir_b
    na, nb   = len(active_a), len(active_b)
    n_so     = na + nb

    # ---- Fock matrices ---------------------------------------------------
    fock_ao = mf.get_fock()                                  # (2, nao, nao)
    ca = mo_coeff_a[:, active_a]                             # (nao, na)
    cb = mo_coeff_b[:, active_b]                             # (nao, nb)

    fock_mo_a = ca.T @ np.asarray(fock_ao[0], dtype=np.float64) @ ca  # (na, na)
    fock_mo_b = cb.T @ np.asarray(fock_ao[1], dtype=np.float64) @ cb  # (nb, nb)

    eps_a = mo_energy_a[active_a]
    eps_b = mo_energy_b[active_b]

    # ---- ERI transforms for all required spin blocks --------------------
    # Physicist <pq|rs>:
    #   aaaa:  (pr|qs) with all alpha
    #   bbbb:  (pr|qs) with all beta
    #   aabb:  (p_α r_α | q_β s_β) -> <p_α q_β | r_α s_β>
    #   bbaa:  hermitian conjugate of aabb

    def _eri_phys(c1, c2, c3, c4, n1, n2, n3, n4):
        """ao2mo for four distinct MO sets, return physicist <pq|rs>."""
        raw = ao2mo.general(mol, (c1, c3, c2, c4), compact=False)
        return raw.reshape(n1, n3, n2, n4).transpose(0, 2, 1, 3)  # <pq|rs>

    eri_aaaa = _eri_phys(ca, ca, ca, ca, na, na, na, na)
    eri_bbbb = _eri_phys(cb, cb, cb, cb, nb, nb, nb, nb)
    eri_aabb = _eri_phys(ca, cb, ca, cb, na, nb, na, nb)  # <p_α q_β | r_α s_β>
    eri_bbaa = eri_aabb.transpose(1, 0, 3, 2)             # <p_β q_α | r_β s_α>

    # ---- Assemble spin-orbital ERI in block layout ----------------------
    # α indices = 0..na-1, β indices = na..na+nb-1
    eri_so = np.zeros((n_so, n_so, n_so, n_so))
    eri_so[:na, :na, :na, :na] = eri_aaaa
    eri_so[na:, na:, na:, na:] = eri_bbbb
    eri_so[:na, na:, :na, na:] = eri_aabb   # αβ|αβ
    eri_so[na:, :na, na:, :na] = eri_bbaa   # βα|βα
    eri_antisym = eri_so - eri_so.transpose(0, 1, 3, 2)

    # ---- Spin-orbital Fock & eps ----------------------------------------
    fock_so = np.zeros((n_so, n_so))
    fock_so[:na, :na] = fock_mo_a
    fock_so[na:, na:] = fock_mo_b

    eps_so = np.concatenate([eps_a, eps_b])

    # ---- Reorder: occupied first ----------------------------------------
    n_act_occ_a, n_act_occ_b = len(act_occ_a), len(act_occ_b)
    alpha_occ = list(range(n_act_occ_a))
    beta_occ  = list(range(na, na + n_act_occ_b))
    alpha_vir = list(range(n_act_occ_a, na))
    beta_vir  = list(range(na + n_act_occ_b, n_so))
    order = alpha_occ + beta_occ + alpha_vir + beta_vir

    eps_out  = eps_so[order]
    fock_out = fock_so[np.ix_(order, order)]
    eri_out  = eri_antisym[np.ix_(order, order, order, order)]

    n_occ_so = n_act_occ_a + n_act_occ_b
    n_vir_so = (na - n_act_occ_a) + (nb - n_act_occ_b)

    if verbose:
        _print_summary(
            mol, "UHF", n_occ_so, n_vir_so, mf.e_tot, frozen,
            n_frozen=(n_occ_a - n_act_occ_a) + (n_occ_b - n_act_occ_b),
        )

    return SCFInputData(
        n_occ=n_occ_so,
        n_vir=n_vir_so,
        e_scf=float(mf.e_tot),
        eps=eps_out,
        fock=fock_out,
        eri_antisym=eri_out,
        reference="UHF",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_frozen(
    nmo: int, n_occ_s: int, frozen
) -> tuple[list[int], list[int]]:
    """
    Return (active_occ_spatial, active_vir_spatial) index lists.

    ``frozen`` can be an int (freeze the lowest *n* occupied orbitals)
    or a list of explicit 0-based spatial MO indices to freeze.
    All frozen indices must be occupied.
    """
    if frozen is None:
        frozen_set: set[int] = set()
    elif isinstance(frozen, int):
        if frozen < 0 or frozen > n_occ_s:
            raise ValueError(
                f"frozen={frozen} is out of range for n_occ_spatial={n_occ_s}"
            )
        frozen_set = set(range(frozen))
    else:
        frozen_set = set(int(i) for i in frozen)
        invalid = [i for i in frozen_set if i >= n_occ_s]
        if invalid:
            raise ValueError(
                f"Frozen indices {invalid} are virtual orbitals "
                f"(n_occ_spatial={n_occ_s}). Only occupied orbitals can be frozen."
            )

    active_occ = [i for i in range(n_occ_s) if i not in frozen_set]
    active_vir = list(range(n_occ_s, nmo))
    return active_occ, active_vir


def _print_summary(
    mol, reference: str, n_occ: int, n_vir: int,
    e_scf: float, frozen, n_frozen: int
) -> None:
    formula = mol.atom_pure_symbol(0) if mol.natm else "?"
    print(
        f"[cupyccx] {reference} integrals extracted | "
        f"basis={mol.basis} | "
        f"n_occ={n_occ} n_vir={n_vir} n_frozen={n_frozen} | "
        f"E_SCF={e_scf:.10f} Ha"
    )
