"""
Pure-Python CCD/LCCD/DCD/pCCD solvers using NumPy (CPU) or CuPy (GPU).

Designed as a transparent debugging reference: every einsum mirrors the
corresponding C++ loop in src/cpp/ccd.cpp so energies and T2 amplitudes
can be cross-checked iteration-by-iteration.

Usage::

    from cupyccx.py_solver import PyCCD, PyLCCD, PyDCD, PypCCD
    r = PyCCD.from_scf_data(data).compute(data.e_scf)
    print(r.e_corr)

GPU (requires CuPy)::

    from cupyccx.method import CCOptions
    opts = CCOptions(use_gpu=True)
    r = PyCCD.from_scf_data(data, opts=opts).compute(data.e_scf)
"""

from __future__ import annotations

import warnings
import numpy as np
from typing import Optional

from cupyccx.method import CCOptions, CCResult


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------

def _get_xp(use_gpu: bool):
    if use_gpu:
        try:
            import cupy
            return cupy
        except ImportError:
            warnings.warn("CuPy not available; falling back to NumPy.", stacklevel=3)
    return np


def _to_np(x):
    """Convert a CuPy or NumPy array to a plain NumPy array."""
    if hasattr(x, 'get'):
        return x.get()
    return np.asarray(x)


def _slice_integrals(eri_antisym, eri_plain, o, xp):
    """Slice full spin-orbital ERI into blocks needed by CC solvers."""
    eri = xp.asarray(eri_antisym)
    oovv = eri[:o, :o, o:, o:]
    vvvv = eri[o:, o:, o:, o:]
    oooo = eri[:o, :o, :o, :o]
    ovvo = eri[:o, o:, o:, :o]
    oovv_plain = xp.asarray(eri_plain[:o, :o, o:, o:]) if eri_plain is not None else None
    return oovv, vvvv, oooo, ovvo, oovv_plain


def _make_D2(eps, o, v, xp):
    """D2[i,j,a,b] = eps_i + eps_j - eps_a - eps_b  (all positive for occ-first ordering)."""
    e = xp.asarray(eps)
    return (e[:o, None, None, None] + e[None, :o, None, None]
            - e[None, None, o:, None] - e[None, None, None, o:])


def _energy(oovv, t2, xp):
    return 0.25 * float(xp.einsum('ijab,ijab->', oovv, t2))


# ---------------------------------------------------------------------------
# DIIS extrapolation
# ---------------------------------------------------------------------------

class _DIIS:
    """Pulay DIIS for T2 amplitudes. Stores vectors as NumPy to keep CPU state."""

    def __init__(self, size: int = 6):
        self._size = size
        self._errs: list = []
        self._amps: list = []

    def push(self, amp_flat, err_flat):
        self._amps.append(_to_np(amp_flat).copy())
        self._errs.append(_to_np(err_flat).copy())
        if len(self._amps) > self._size:
            self._amps.pop(0)
            self._errs.pop(0)

    def extrapolate(self, shape, xp=np):
        n = len(self._errs)
        if n < 2:
            return xp.asarray(self._amps[-1]).reshape(shape)
        B = np.zeros((n + 1, n + 1))
        for i in range(n):
            for j in range(i, n):
                B[i, j] = float(np.dot(self._errs[i], self._errs[j]))
                B[j, i] = B[i, j]
        B[n, :n] = -1.0
        B[:n, n] = -1.0
        rhs = np.zeros(n + 1)
        rhs[n] = -1.0
        try:
            c = np.linalg.solve(B, rhs)
        except np.linalg.LinAlgError:
            return xp.asarray(self._amps[-1]).reshape(shape)
        result = np.zeros_like(self._amps[0])
        for i in range(n):
            result += float(c[i]) * self._amps[i]
        return xp.asarray(result).reshape(shape)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class _PyCC:
    """Base class for pure-Python CC solvers."""

    def __init__(self, data, opts: Optional[CCOptions] = None):
        self.data = data
        self.opts = opts or CCOptions()
        self.xp = _get_xp(self.opts.use_gpu)

    @classmethod
    def from_scf_data(cls, data, opts: Optional[CCOptions] = None):
        return cls(data, opts)

    def _build_residual(self, t2, oovv, vvvv, oooo, ovvo, oovv_plain, F_vv, F_oo):
        raise NotImplementedError

    def compute(self, e_scf: float = 0.0, callback=None, verbose: bool = False) -> CCResult:
        xp = self.xp
        data, opts = self.data, self.opts
        o, v = data.n_occ, data.n_vir
        oovv, vvvv, oooo, ovvo, oovv_plain = _slice_integrals(
            data.eri_antisym, data.eri_plain, o, xp)
        D2 = _make_D2(data.eps, o, v, xp)
        F_vv = xp.asarray(data.fock[o:, o:])
        F_oo = xp.asarray(data.fock[:o, :o])
        t2 = oovv / D2
        diis = _DIIS(opts.diis_size)
        e_prev = 0.0
        for it in range(1, opts.max_iter + 1):
            R = self._build_residual(t2, oovv, vvvv, oooo, ovvo, oovv_plain, F_vv, F_oo)
            t2_new = t2 - R / D2
            if opts.use_diis:
                diis.push(t2_new.ravel(), (t2_new - t2).ravel())
                if it >= 2:
                    t2_new = diis.extrapolate((o, o, v, v), xp)
            t2 = t2_new
            e_corr = _energy(oovv, t2, xp)
            de = abs(e_corr - e_prev)
            e_prev = e_corr
            rms = float(xp.sqrt(xp.mean(R ** 2)))
            if callback:
                callback(it, e_corr, de, rms)
            if verbose:
                print(f"  {it:4d}  {e_corr:.10f}  {de:.2e}  {rms:.2e}")
            if de < opts.conv_energy and rms < opts.conv_amp:
                return CCResult(e_corr=e_corr, e_total=e_scf + e_corr,
                                n_iter=it, converged=True, t2=_to_np(t2))
        return CCResult(e_corr=e_prev, e_total=e_scf + e_prev,
                        n_iter=opts.max_iter, converged=False, t2=_to_np(t2))


# ---------------------------------------------------------------------------
# PyCCD — Full CCD
# ---------------------------------------------------------------------------

class PyCCD(_PyCC):
    """
    Pure-Python CCD solver.

    Mirrors C++ CCD::build_residual (ccd.cpp).
    Architecture: dressed W intermediates (Q_B) + explicit Q_D, Q_C, Q_A.
    """

    def _build_residual(self, t2, oovv, vvvv, oooo, ovvo, oovv_plain, F_vv, F_oo):
        xp = self.xp
        # Dressed W intermediates — contribute Q_B (= 0.25 × B) to R
        # ccd.cpp build_W_oooo / build_W_vvvv
        W_oooo = oooo + 0.25 * xp.einsum('klcd,ijcd->klij', oovv, t2)
        W_vvvv = vvvv + 0.25 * xp.einsum('klcd,klab->abcd', oovv, t2)
        R = oovv.copy()
        # P(ab) f_bc t^ac_ij — ccd.cpp:67-78
        R += xp.einsum('bc,ijac->ijab', F_vv, t2) - xp.einsum('ac,ijbc->ijab', F_vv, t2)
        # -P(ij) f_kj t^ab_ik — ccd.cpp:80-91
        R -= xp.einsum('kj,ikab->ijab', F_oo, t2) - xp.einsum('ki,jkab->ijab', F_oo, t2)
        # (1/2) W_klij t^ab_kl + (1/2) W_abcd t^cd_ij — ccd.cpp:95-98
        R += 0.5 * xp.einsum('klij,klab->ijab', W_oooo, t2)
        R += 0.5 * xp.einsum('abcd,ijcd->ijab', W_vvvv, t2)
        # P(ij)P(ab) <kb||cj> t^ac_ik — ccd.cpp:101
        X = xp.einsum('kbcj,ikac->ijab', ovvo, t2)
        R += X - X.transpose(0, 1, 3, 2) - X.transpose(1, 0, 2, 3) + X.transpose(1, 0, 3, 2)
        # Q_D: (1/2) P(ij)P(ab) sum_{klcd} <kl||cd> t^ac_ik t^bd_jl — ccd.cpp:103-131
        # Y[i,a,l,d] = sum_{k,c} oovv[k,l,c,d]*t2[i,k,a,c]
        Y = xp.einsum('klcd,ikac->iald', oovv, t2)
        Z = xp.einsum('iald,jlbd->ijab', Y, t2)
        Zs = Z - Z.transpose(1, 0, 2, 3) - Z.transpose(0, 1, 3, 2) + Z.transpose(1, 0, 3, 2)
        R += 0.5 * Zs
        # Q_C: dressed F_vv — ccd.cpp:133-157
        # dF_vv[b,e] = -0.5 * sum_{k,l,d} oovv[k,l,e,d]*t2[k,l,b,d]
        dF_vv = -0.5 * xp.einsum('klcd,klbd->bc', oovv, t2)
        R -= xp.einsum('ac,ijbc->ijab', dF_vv, t2) - xp.einsum('bc,ijac->ijab', dF_vv, t2)
        # Q_A: dressed F_oo — ccd.cpp:159-183
        # dF_oo[k,j] = +0.5 * sum_{l,c,d} oovv[k,l,c,d]*t2[j,l,c,d]
        # R -= dF_oo[k,j]*t2[i,k,a,b] - dF_oo[k,i]*t2[j,k,a,b]
        dF_oo = 0.5 * xp.einsum('klcd,jlcd->kj', oovv, t2)
        R -= xp.einsum('kj,ikab->ijab', dF_oo, t2) - xp.einsum('ki,jkab->ijab', dF_oo, t2)
        return R


# ---------------------------------------------------------------------------
# PyLCCD — Linearized CCD (no T2*T2 terms)
# ---------------------------------------------------------------------------

class PyLCCD(_PyCC):
    """
    Pure-Python LCCD solver.

    Same as PyCCD but with bare (undressed) oooo/vvvv and no Q_D/Q_C/Q_A blocks.
    """

    def _build_residual(self, t2, oovv, vvvv, oooo, ovvo, oovv_plain, F_vv, F_oo):
        xp = self.xp
        R = oovv.copy()
        R += xp.einsum('bc,ijac->ijab', F_vv, t2) - xp.einsum('ac,ijbc->ijab', F_vv, t2)
        R -= xp.einsum('kj,ikab->ijab', F_oo, t2) - xp.einsum('ki,jkab->ijab', F_oo, t2)
        R += 0.5 * xp.einsum('klij,klab->ijab', oooo, t2)
        R += 0.5 * xp.einsum('abcd,ijcd->ijab', vvvv, t2)
        X = xp.einsum('kbcj,ikac->ijab', ovvo, t2)
        R += X - X.transpose(0, 1, 3, 2) - X.transpose(1, 0, 2, 3) + X.transpose(1, 0, 3, 2)
        return R


# ---------------------------------------------------------------------------
# PyDCD — Distinguishable Cluster Doubles
# ---------------------------------------------------------------------------

class PyDCD(_PyCC):
    """
    Pure-Python DCD solver.

    Mirrors DCD::build_residual in ccd.cpp (lines 518-619).
    Requires data.eri_plain (plain Coulomb ERIs) for the Q3 Coulomb-ring term.
    """

    def _build_residual(self, t2, oovv, vvvv, oooo, ovvo, oovv_plain, F_vv, F_oo):
        xp = self.xp
        if oovv_plain is None:
            raise ValueError(
                "DCD requires eri_plain. Use prepare_from_pyscf() which populates "
                "SCFInputData.eri_plain, or pass eri_plain to the solver constructor.")
        R = oovv.copy()
        # Fock terms — same as CCD linear part
        R += xp.einsum('bc,ijac->ijab', F_vv, t2) - xp.einsum('ac,ijbc->ijab', F_vv, t2)
        R -= xp.einsum('kj,ikab->ijab', F_oo, t2) - xp.einsum('ki,jkab->ijab', F_oo, t2)
        # Bare ladders (no W dressing)
        R += 0.5 * xp.einsum('klij,klab->ijab', oooo, t2)
        R += 0.5 * xp.einsum('abcd,ijcd->ijab', vvvv, t2)
        # Ring P(ij)P(ab) — bare ovvo
        X = xp.einsum('kbcj,ikac->ijab', ovvo, t2)
        R += X - X.transpose(0, 1, 3, 2) - X.transpose(1, 0, 2, 3) + X.transpose(1, 0, 3, 2)
        # Q1: +(1/4) P(ij) sum_{klcd} <kl||cd> t^cd_ik t^ab_lj — ccd.cpp:519-547
        # Y1[i,l] = sum_{k,c,d} oovv[k,l,c,d]*t2[i,k,c,d]
        Y1 = xp.einsum('klcd,ikcd->il', oovv, t2)
        R += 0.25 * (xp.einsum('il,ljab->ijab', Y1, t2) - xp.einsum('jl,liab->ijab', Y1, t2))
        # Q2: +(1/4) P(ab) sum_{klcd} <kl||cd> t^ac_kl t^db_ij — ccd.cpp:548-572
        # Y2[a,d] = sum_{k,l,c} oovv[k,l,c,d]*t2[k,l,a,c]
        Y2 = xp.einsum('klcd,klac->ad', oovv, t2)
        R += 0.25 * (xp.einsum('ad,ijdb->ijab', Y2, t2) - xp.einsum('bd,ijda->ijab', Y2, t2))
        # Q3: +(1/2) P(ij)P(ab) sum_{klcd} <kl|cd> t^ac_ik t^bd_jl — ccd.cpp:573-613
        # Z3[i,a,l,d] = sum_{k,c} oovv_plain[k,l,c,d]*t2[i,k,a,c]
        Z3 = xp.einsum('klcd,ikac->iald', oovv_plain, t2)
        Zb = xp.einsum('iald,jlbd->ijab', Z3, t2)
        R += 0.5 * (Zb - Zb.transpose(1, 0, 2, 3) - Zb.transpose(0, 1, 3, 2) + Zb.transpose(1, 0, 3, 2))
        return R


# ---------------------------------------------------------------------------
# PypCCD — Parameterized CCD
# ---------------------------------------------------------------------------

def _build_quad_Q(oovv, t2, cDc, cC, cA, cDx, cB, xp):
    """
    Compute the unsymmetrized quadratic tensor Q (mirrors build_quad_Q in ccd.cpp).
    Caller applies symmetrisation: R += Q + Q.transpose(1,0,3,2).

    P-term definitions (ccd.cpp:293-297):
      cDc (P1): D_c = sum_{klcd} oovv_plain[klcd]*t2[ilad]*t2[kjcb]
      cC  (P2): C   = sum_{klcd} oovv[klcd]*t2[klad]*t2[ijcb]
      cA  (P3): A   = sum_{klcd} oovv[klcd]*t2[ilcd]*t2[kjab]
      cDx (P4): D_x = sum_{klcd} oovv_plain[klcd]*t2[ildb]*t2[kjac]
      cB  (P5): B   = sum_{klcd} oovv[klcd]*t2[ijcd]*t2[klab]  (doubles under sym)
    """
    o = t2.shape[0]
    v = t2.shape[2]
    Q = xp.zeros((o, o, v, v), dtype=t2.dtype)
    if cDc != 0.0:
        # A[l,d,j,b] = sum_{k,c} oovv[k,l,c,d]*t2[k,j,c,b]
        A = xp.einsum('klcd,kjcb->ldjb', oovv, t2)
        Q += cDc * xp.einsum('ilad,ldjb->ijab', t2, A)
    if cC != 0.0:
        # Bmat[a,c] = sum_{k,l,d} oovv[k,l,c,d]*t2[k,l,a,d]
        Bmat = xp.einsum('klcd,klad->ac', oovv, t2)
        Q += cC * xp.einsum('ac,ijcb->ijab', Bmat, t2)
    if cA != 0.0:
        # Cmat[k,i] = sum_{l,c,d} oovv[k,l,c,d]*t2[i,l,c,d]
        Cmat = xp.einsum('klcd,ilcd->ki', oovv, t2)
        Q += cA * xp.einsum('ki,kjab->ijab', Cmat, t2)
    if cDx != 0.0:
        # H[k,c,i,b] = sum_{l,d} oovv[k,l,c,d]*t2[i,l,d,b]
        H = xp.einsum('klcd,ildb->kcib', oovv, t2)
        Q += cDx * xp.einsum('kcib,kjac->ijab', H, t2)
    if cB != 0.0:
        # M[k,l,i,j] = sum_{c,d} oovv[k,l,c,d]*t2[i,j,c,d]
        M = xp.einsum('klcd,ijcd->klij', oovv, t2)
        Q += cB * xp.einsum('klij,klab->ijab', M, t2)
    return Q


class PypCCD(_PyCC):
    """
    Pure-Python pCCD (parameterized CCD) solver.

    Mirrors pCCD::build_residual (ccd.cpp:687-710).
    At alpha=1, beta=1: recovers full CCD exactly.
    Does not require data.eri_plain; oovv (antisym) is used for all D terms.

    Parameters
    ----------
    alpha, beta : float
        pCCD parameters. Default 0.0 (no quadratic terms → LCCD-like).
    """

    def __init__(self, data, opts: Optional[CCOptions] = None,
                 alpha: float = 0.0, beta: float = 0.0):
        super().__init__(data, opts)
        self.alpha = alpha
        self.beta = beta

    @classmethod
    def from_scf_data(cls, data, opts: Optional[CCOptions] = None,
                      alpha: float = 0.0, beta: float = 0.0):
        return cls(data, opts, alpha=alpha, beta=beta)

    def _build_residual(self, t2, oovv, vvvv, oooo, ovvo, oovv_plain, F_vv, F_oo):
        xp = self.xp
        alpha, beta = self.alpha, self.beta
        R = oovv.copy()
        # Linear part: same as DCD (bare ladders + bare ring)
        R += xp.einsum('bc,ijac->ijab', F_vv, t2) - xp.einsum('ac,ijbc->ijab', F_vv, t2)
        R -= xp.einsum('kj,ikab->ijab', F_oo, t2) - xp.einsum('ki,jkab->ijab', F_oo, t2)
        R += 0.5 * xp.einsum('klij,klab->ijab', oooo, t2)
        R += 0.5 * xp.einsum('abcd,ijcd->ijab', vvvv, t2)
        X = xp.einsum('kbcj,ikac->ijab', ovvo, t2)
        R += X - X.transpose(0, 1, 3, 2) - X.transpose(1, 0, 2, 3) + X.transpose(1, 0, 3, 2)
        # Quadratic terms — pCCD uses oovv (antisym) for all D terms
        # (at alpha=beta=1 this recovers CCD exactly; ccd.cpp:700-707)
        Q = _build_quad_Q(
            oovv, t2,
            cDc=0.5 * beta,
            cC=-0.5 * beta,
            cA=-0.25 * (1.0 + alpha),
            cDx=-0.5 * beta,
            cB=alpha / 8.0,
            xp=xp,
        )
        R += Q + Q.transpose(1, 0, 3, 2)
        return R
