"""
Python-level tests for cupyccx.

These tests bypass the C++ extension and drive the Python API layer directly.
When the extension is available, they also exercise the bindings end-to-end.
"""

import numpy as np
import pytest

# -----------------------------------------------------------------------
# Helpers — build a tiny H2 STO-3G spin-orbital dataset analytically
# -----------------------------------------------------------------------

def h2_sto3g_data():
    """
    Minimal 2-orbital / 2-electron spin-orbital dataset for STO-3G H2.
    n_occ=2, n_vir=2, n_mo=4.
    """
    eps_occ, eps_vir = -0.57827, 0.67012
    K = 0.20636  # spatial (occ|vir) exchange integral

    eps  = np.array([eps_occ, eps_occ, eps_vir, eps_vir])
    fock = np.diag(eps)

    n_mo = 4
    eri = np.zeros((n_mo, n_mo, n_mo, n_mo))
    eri[0,1,2,3] =  K;  eri[1,0,3,2] =  K
    eri[0,1,3,2] = -K;  eri[1,0,2,3] = -K
    eri[2,3,0,1] =  K;  eri[3,2,1,0] =  K
    eri[3,2,0,1] = -K;  eri[2,3,1,0] = -K

    return dict(n_occ=2, n_vir=2, eps=eps, fock=fock, eri=eri)


# -----------------------------------------------------------------------
# Tests for the Python-layer CCOptions
# -----------------------------------------------------------------------

def test_ccoptions_defaults():
    from cupyccx.method import CCOptions
    opts = CCOptions()
    assert opts.max_iter == 100
    assert opts.conv_energy == pytest.approx(1e-9)
    assert not opts.use_gpu


def test_ccoptions_mutation():
    from cupyccx.method import CCOptions
    opts = CCOptions(max_iter=50, use_gpu=True)
    assert opts.max_iter == 50
    assert opts.use_gpu


# -----------------------------------------------------------------------
# Tests for the extension (skipped if not built)
# -----------------------------------------------------------------------

try:
    import cupyccx._cupyccx as _ext
    HAS_EXT = True
except ImportError:
    HAS_EXT = False

skip_no_ext = pytest.mark.skipif(not HAS_EXT, reason="_cupyccx extension not built")


@skip_no_ext
def test_run_ccd_converges():
    from cupyccx import run_ccd
    from cupyccx.method import CCOptions
    d = h2_sto3g_data()
    opts = CCOptions(conv_energy=1e-12, conv_amp=1e-10)
    r = run_ccd(d["n_occ"], d["n_vir"], d["eps"], d["fock"], d["eri"],
                opts=opts._to_ext())
    assert r.converged
    assert r.e_corr < 0
    assert abs(r.e_corr - (-0.04222)) < 5e-4


@skip_no_ext
def test_run_lccd_converges():
    from cupyccx import run_lccd
    from cupyccx.method import CCOptions
    d = h2_sto3g_data()
    opts = CCOptions(conv_energy=1e-12, conv_amp=1e-10)
    r = run_lccd(d["n_occ"], d["n_vir"], d["eps"], d["fock"], d["eri"],
                 opts=opts._to_ext())
    assert r.converged
    assert r.e_corr < 0


@skip_no_ext
def test_t2_antisymmetry():
    from cupyccx import run_ccd
    from cupyccx.method import CCOptions
    d = h2_sto3g_data()
    r = run_ccd(d["n_occ"], d["n_vir"], d["eps"], d["fock"], d["eri"],
                opts=CCOptions(conv_energy=1e-12)._to_ext())
    t2 = np.array(r.t2)
    n_occ, n_vir = d["n_occ"], d["n_vir"]
    assert t2.shape == (n_occ, n_occ, n_vir, n_vir)
    # t2[i,j,a,b] == -t2[j,i,a,b]
    np.testing.assert_allclose(t2, -t2.transpose(1,0,2,3), atol=1e-10)
    # t2[i,j,a,b] == -t2[i,j,b,a]
    np.testing.assert_allclose(t2, -t2.transpose(0,1,3,2), atol=1e-10)


@skip_no_ext
def test_callback_called():
    from cupyccx import run_ccd
    from cupyccx.method import CCOptions
    d = h2_sto3g_data()
    calls = []
    run_ccd(d["n_occ"], d["n_vir"], d["eps"], d["fock"], d["eri"],
            opts=CCOptions()._to_ext(),
            callback=lambda it, e, de, rms: calls.append(it))
    assert len(calls) > 0


@skip_no_ext
def test_ccresult_repr():
    from cupyccx import run_ccd
    from cupyccx.method import CCOptions
    d = h2_sto3g_data()
    r = run_ccd(d["n_occ"], d["n_vir"], d["eps"], d["fock"], d["eri"],
                opts=CCOptions()._to_ext())
    assert "CCResult" in repr(r)
