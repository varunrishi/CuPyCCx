"""
Example: CCD on water, demonstrating the three ways to drive cupyccx from PySCF.

Prerequisites:
    pip install pyscf
    pip install -e .        # build and install cupyccx

Run with:  python examples/pyscf_h2o.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from pyscf import gto, scf
from cupyccx.scf_data import prepare_from_pyscf, SCFInputData
from cupyccx import CCD, LCCD, CCOptions
from cupyccx.pyscf_interface import run_from_pyscf

# ------------------------------------------------------------------
# 1. Run PySCF HF
# ------------------------------------------------------------------
mol = gto.M(
    atom="""
        O   0.000  0.000  0.000
        H   0.000  0.757  0.586
        H   0.000 -0.757  0.586
    """,
    basis="cc-pVDZ",
    unit="Angstrom",
)
mf = scf.RHF(mol).run()
print(f"\nHF energy: {mf.e_tot:.10f} Ha\n")

# ------------------------------------------------------------------
# 2a. One-liner convenience wrapper
# ------------------------------------------------------------------
print("=== Path A: run_from_pyscf (one-liner) ===")
result_a = run_from_pyscf(mf, method="CCD", verbose=True)
print(f"  E_corr = {result_a.e_corr:.10f} Ha\n")

# ------------------------------------------------------------------
# 2b. Extract integrals first, then run — more control
# ------------------------------------------------------------------
print("=== Path B: prepare_from_pyscf → CCD.from_scf_data ===")
data = prepare_from_pyscf(mf, verbose=True)

# Inspect what was extracted
print(f"  data.n_occ        = {data.n_occ}")
print(f"  data.n_vir        = {data.n_vir}")
print(f"  data.eps[:4]      = {data.eps[:4]}")
print(f"  data.eri_antisym.shape = {data.eri_antisym.shape}\n")

opts   = CCOptions(max_iter=150, conv_energy=1e-9)
result_b = CCD.from_scf_data(data, opts=opts).compute(e_scf=data.e_scf, verbose=True)
print(f"  E_corr = {result_b.e_corr:.10f} Ha\n")

# ------------------------------------------------------------------
# 2c. Save integrals, reload, run — avoids repeating the HF step
# ------------------------------------------------------------------
print("=== Path C: save → load → CCD ===")
data.save("/tmp/h2o_ccpvdz.npz")
print("  Saved integrals to /tmp/h2o_ccpvdz.npz")

data_loaded = SCFInputData.load("/tmp/h2o_ccpvdz.npz")
print(f"  Loaded: {data_loaded}")
result_c = CCD.from_scf_data(data_loaded, opts=opts).compute(
    e_scf=data_loaded.e_scf, verbose=True
)
print(f"  E_corr = {result_c.e_corr:.10f} Ha\n")

# ------------------------------------------------------------------
# 3. Frozen-core CCD (freeze oxygen 1s = orbital index 0)
# ------------------------------------------------------------------
print("=== Frozen-core CCD (freeze O 1s) ===")
data_fc = prepare_from_pyscf(mf, frozen=1, verbose=True)
print(f"  Active: n_occ={data_fc.n_occ}, n_vir={data_fc.n_vir}")
result_fc = CCD.from_scf_data(data_fc).compute(e_scf=data_fc.e_scf, verbose=True)
print(f"  E_corr (frozen) = {result_fc.e_corr:.10f} Ha\n")

# ------------------------------------------------------------------
# 4. LCCD comparison
# ------------------------------------------------------------------
print("=== LCCD vs CCD ===")
result_lccd = LCCD.from_scf_data(data).compute(e_scf=data.e_scf, verbose=True)
print(f"  LCCD E_corr = {result_lccd.e_corr:.10f} Ha")
print(f"  CCD  E_corr = {result_b.e_corr:.10f} Ha")

# ------------------------------------------------------------------
# 5. GPU (uncomment if built with CUPYCCX_CUDA=ON)
# ------------------------------------------------------------------
# opts_gpu = CCOptions(use_gpu=True)
# result_gpu = CCD.from_scf_data(data, opts=opts_gpu).compute(e_scf=data.e_scf)
# print(f"  GPU CCD E_corr = {result_gpu.e_corr:.10f} Ha")
