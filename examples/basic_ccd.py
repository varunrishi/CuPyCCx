"""
Minimal example: run CCD on a manually constructed H2 STO-3G dataset.

This exercises the C++/Python interface without requiring PySCF.
Run with:  python examples/basic_ccd.py
"""

import numpy as np
import sys
import os

# Allow running from repo root before installation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from cupyccx.method import CCD, LCCD, CCOptions

# ------------------------------------------------------------------
# H2 STO-3G spin-orbital integrals (hand-coded, 2 occ + 2 vir)
# ------------------------------------------------------------------
eps_occ, eps_vir = -0.57827, 0.67012
K   = 0.20636  # (occ|vir) exchange integral

n_occ, n_vir = 2, 2
n_mo = n_occ + n_vir

eps  = np.array([eps_occ, eps_occ, eps_vir, eps_vir])
fock = np.diag(eps)

eri  = np.zeros((n_mo, n_mo, n_mo, n_mo))
# Antisymmetrized <pq||rs> for the (occ,occ,vir,vir) block
eri[0,1,2,3] =  K;  eri[1,0,3,2] =  K
eri[0,1,3,2] = -K;  eri[1,0,2,3] = -K
eri[2,3,0,1] =  K;  eri[3,2,1,0] =  K
eri[3,2,0,1] = -K;  eri[2,3,1,0] = -K

# ------------------------------------------------------------------
# Run LCCD
# ------------------------------------------------------------------
opts = CCOptions(conv_energy=1e-12, conv_amp=1e-10)

print("=" * 60)
print("  LCCD / H2 STO-3G (spin-orbital basis)")
print("=" * 60)
lccd   = LCCD(n_occ, n_vir, eps, fock, eri, opts=opts)
r_lccd = lccd.compute(e_scf=0.0, verbose=True)

# ------------------------------------------------------------------
# Run CCD
# ------------------------------------------------------------------
print("=" * 60)
print("  CCD / H2 STO-3G (spin-orbital basis)")
print("=" * 60)
ccd   = CCD(n_occ, n_vir, eps, fock, eri, opts=opts)
r_ccd = ccd.compute(e_scf=0.0, verbose=True)

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
print(f"LCCD E_corr = {r_lccd.e_corr:>18.12f} Ha")
print(f"CCD  E_corr = {r_ccd.e_corr:>18.12f} Ha")
print(f"T2 shape    = {r_ccd.t2.shape}")
