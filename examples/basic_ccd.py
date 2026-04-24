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
# H2 / STO-3G spin-orbital integrals from PySCF (R = 1.4 bohr)
# Spin-orbital ordering: 0=occ_α, 1=occ_β, 2=vir_α, 3=vir_β
#
# Reference energies (PySCF):
#   E_corr(MP2)  = -0.01315787 Ha
#   E_corr(CCSD) = -0.02056178 Ha
# cupyccx solver:
#   E_corr(LCCD) = -0.03288443 Ha
#   E_corr(CCD)  = -0.02141095 Ha
# ------------------------------------------------------------------
eps_occ, eps_vir = -0.57820298, 0.67026777

n_occ, n_vir = 2, 2
n_mo = n_occ + n_vir

eps  = np.array([eps_occ, eps_occ, eps_vir, eps_vir])
fock = np.diag(eps)

eri  = np.zeros((n_mo, n_mo, n_mo, n_mo))
# OOOO
eri[0,1,0,1] =  0.67459408; eri[0,1,1,0] = -0.67459408
eri[1,0,0,1] = -0.67459408; eri[1,0,1,0] =  0.67459408
# OOVV + VVOO
eri[0,1,2,3] =  0.18125791; eri[0,1,3,2] = -0.18125791
eri[1,0,2,3] = -0.18125791; eri[1,0,3,2] =  0.18125791
eri[2,3,0,1] =  0.18125791; eri[2,3,1,0] = -0.18125791
eri[3,2,0,1] = -0.18125791; eri[3,2,1,0] =  0.18125791
# OVOV same-spin
eri[0,2,0,2] =  0.48230608; eri[0,2,2,0] = -0.48230608
eri[2,0,0,2] = -0.48230608; eri[2,0,2,0] =  0.48230608
eri[1,3,1,3] =  0.48230608; eri[1,3,3,1] = -0.48230608
eri[3,1,1,3] = -0.48230608; eri[3,1,3,1] =  0.48230608
# OVOV opposite-spin
eri[0,3,0,3] =  0.66356399; eri[0,3,3,0] = -0.66356399
eri[3,0,0,3] = -0.66356399; eri[3,0,3,0] =  0.66356399
eri[1,2,1,2] =  0.66356399; eri[1,2,2,1] = -0.66356399
eri[2,1,1,2] = -0.66356399; eri[2,1,2,1] =  0.66356399
# OVOV opposite-spin cross terms
eri[0,3,1,2] = -0.18125791; eri[0,3,2,1] =  0.18125791
eri[3,0,1,2] =  0.18125791; eri[3,0,2,1] = -0.18125791
eri[1,2,0,3] = -0.18125791; eri[1,2,3,0] =  0.18125791
eri[2,1,0,3] =  0.18125791; eri[2,1,3,0] = -0.18125791
# VVVV
eri[2,3,2,3] =  0.69749535; eri[2,3,3,2] = -0.69749535
eri[3,2,2,3] = -0.69749535; eri[3,2,3,2] =  0.69749535

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
