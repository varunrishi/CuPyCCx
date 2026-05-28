"""
Example: all four methods on N2 / STO-3G at the experimental bond length.

Compares LCCD, CCD, DCD, and pCCD correlation energies.
Reference energies from Rishi, Perera, Bartlett, J. Chem. Phys. 144, 124117 (2016).

Run with:  python examples/n2_sto3g.py
"""

from pyscf import gto, scf
from cupyccx.pyscf_interface import run_from_pyscf
from cupyccx.method import CCOptions

mol = gto.M(
    atom="N 0 0 0; N 0 0 2.118",
    basis="sto-3g",
    unit="Bohr",
)
mf = scf.RHF(mol).run()
print(f"HF energy: {mf.e_tot:.10f} Ha\n")

opts = CCOptions(max_iter=200, conv_energy=1e-9, conv_amp=1e-8)

results = {}
for method in ["LCCD", "CCD", "DCD"]:
    results[method] = run_from_pyscf(mf, method=method, opts=opts, verbose=True)

# pCCD with alpha=1, beta=1 recovers full CCD
results["pCCD(1,1)"] = run_from_pyscf(mf, method="pCCD", alpha=1.0, beta=1.0,
                                       opts=opts, verbose=False)

print("\n" + "=" * 55)
print(f"  {'Method':<12}  {'E_corr (Ha)':>18}  {'Converged'}")
print("  " + "-" * 51)
for name, r in results.items():
    print(f"  {name:<12}  {r.e_corr:>18.10f}  {r.converged}")
print("=" * 55)

# Reference values (Rishi et al. 2016)
print("\nReference energies:")
print(f"  LCCD  -0.1629097927 Ha")
print(f"  CCD   -0.1593939001 Ha")
print(f"  DCD   -0.1683961600 Ha")
