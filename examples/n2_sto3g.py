from pyscf import gto, scf, cc, ci
from cupyccx.pyscf_interface import run_from_pyscf
from cupyccx.method import CCOptions

# Build N2 molecule (experimental bond length = 2.118 Bohrs)
mol = gto.M(
    atom="N 0 0 0; N 0 0 2.118",
    basis="sto3g",
    unit="Bohrs",
)
mf = scf.RHF(mol).run()
print(f"\nHF energy: {mf.e_tot:.10f} Ha")
pt = mf.MP2()
pt.run()

print(f"MP2 Correlation Energy: {pt.e_corr: .6f} Hartree")
print(f"Total MP2 Energy: {pt.e_tot: .6f} Hartree")
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


opts = CCOptions(max_iter=1000, conv_energy=1e-6, conv_amp=1e-6)

#for method in ["LCCD", "DCD", "CCD"]:
for method in ["CCD"]:
    result = run_from_pyscf(mf, method=method, opts=opts, verbose=True)
    print(f"{method:5s}  E_corr = {result.e_corr:.10f} Ha\n")  
    print(f"E_total = {result.e_total:.10f} Ha\n")




"""
Reference numbers from PySCF are
HF energy: -107.5000635015 Ha
MP2 corr energy: -0.16262605822406 Ha 
E(CCSD) = -107.659711410291  E_corr = -0.1596479088302308
CCSD energy from PySCF: -0.15964791 Ha
CCD corr energy -0.15939390011736493
CCD total energy -107.65945740157817
LCCD corr energy from PSI4: -0.1629097909 Ha
LCCD + Quad Term B = -0.166120529098 Ha 

pCCD(1,1) corr energy : -0.1593939 ha
DCD corr energy : -0.16839616 Ha

CuPyCCx so far
LCCD   E_corr = 0.1629094 Ha 
CCD    E_corr = -0.152565 Ha


"""
# pCCD with custom parameters

from pyscf import gto, scf
from cupyccx.scf_data import prepare_from_pyscf
from cupyccx.method import pCCD, CCOptions


data = prepare_from_pyscf(mf)
opts = CCOptions(max_iter=100)

result = pCCD.from_scf_data(data, alpha=1, beta=1, opts=opts).compute(
    e_scf=data.e_scf, verbose=True
)
print(f"pCCD(alpha=1,beta=1) E_corr = {result.e_corr:.10f} Ha")


# pCCD directly
from cupyccx.method import pCCD
from cupyccx.scf_data import prepare_from_pyscf

result = run_from_pyscf(mf, method="pCCD", alpha=1.0, beta=1.0, opts=opts, verbose=True)

print(f"{method:5s}  E_corr = {result.e_corr:.10f} Ha\n")  
print(f"E_total = {result.e_total:.10f} Ha\n")

#



#data = prepare_from_pyscf(mf)
#result_pccd = pCCD.from_scf_data(data, alpha=1.0, beta=1.0, opts=opts).compute()
#print(f"pCCD  E_corr = {result_pccd.e_corr:.10f} Ha")

#With frozen core (freeze the two N 1s orbitals, which is standard for N2):

# DCD
result = run_from_pyscf(mf, method="DCD", opts=opts, verbose=True)

print(f"{method:5s}  E_corr = {result.e_corr:.10f} Ha\n")
print(f"E_total = {result.e_total:.10f} Ha\n")


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
