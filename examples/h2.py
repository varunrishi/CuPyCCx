from pyscf import gto, scf, cc
from cupyccx.pyscf_interface import run_from_pyscf
from cupyccx.method import CCOptions

# Build H2 molecule (bond length = 1.4 Bohrs)
mol = gto.M(
    atom="H 0 0 0; H 0 0 1.4",
    basis="sto-3g",
    unit="Bohrs",
)
mf = scf.RHF(mol).run()
print(f"\nHF energy: {mf.e_tot:.10f} Ha")
mycc = cc.CCSD(mf).run()
print(f"\nCCSD energy from PySCF: {mycc.e_corr:.8f} Ha")
mycc = cc.CCSD(mf)
# #mycc.frozen = 1 # frozen core
# Modify update_amps to enforce CCD (set t1 = 0)
old_update_amps = mycc.update_amps
def update_amps(t1, t2, eris):
    t1, t2 = old_update_amps(t1, t2, eris)
    return t1*0, t2
mycc.update_amps = update_amps
mycc.kernel()
print('HF total energy', mf.e_tot)
print('CCD corr energy', mycc.e_corr)
print('CCD total energy', mycc.e_tot)



opts = CCOptions(max_iter=200, conv_energy=1e-9)

#for method in ["LCCD", "DCD", "CCD"]:
for method in ["CCD"]:
    result = run_from_pyscf(mf, method=method, opts=opts, verbose=True)
    print(f"{method:5s}  E_corr = {result.e_corr:.10f} Ha\n")  
    print(f"E_total = {result.e_total:.10f} Ha\n")

"""
# pCCD with custom parameters
from cupyccx.method import pCCD
from cupyccx.scf_data import prepare_from_pyscf

data = prepare_from_pyscf(mf)
result_pccd = pCCD.from_scf_data(data, alpha=1.0, beta=1.0, opts=opts).compute(
    e_scf=data.e_scf, verbose=True
)
print(f"pCCD  E_corr = {result_pccd.e_corr:.10f} Ha")

#With frozen core (freeze the two N 1s orbitals, which is standard for N2):

result_fc = run_from_pyscf(mf, method="CCD", frozen=2, opts=opts, verbose=True)
"""


"""
#  To compare against PySCF's own CCSD:

from pyscf import cc
mycc = cc.CCSD(mf).run()
print(f"PySCF CCSD E_corr = {mycc.e_corr:.10f} Ha")

#Save N2 integrals to disk (avoids re-running HF on subsequent runs):

from cupyccx.scf_data import prepare_from_pyscf, SCFInputData

data = prepare_from_pyscf(mf, frozen=2)
data.save("n2_ccpvdz.npz")

# Later:
data = SCFInputData.load("n2_ccpvdz.npz")
from cupyccx import CCD
result = CCD.from_scf_data(data, opts=opts).compute(e_scf=data.e_scf, verbose=True)
"""
