"""
GPU performance profiling script for cupyccx.

Designed to be run under NVIDIA Nsight Systems:

    nsys profile --trace=cuda,cublas,osrt \\
                 --output=cupyccx_profile \\
                 python examples/profile_gpu.py

or Nsight Compute (single-kernel metrics):

    ncu --set full --target-processes all \\
        -o cupyccx_ncu \\
        python examples/profile_gpu.py

The script runs a fixed number of CCD iterations (not to convergence) on
progressively larger systems so that the profiler captures a representative
mix of DGEMM sizes and host↔device transfer overhead.

Prerequisites:
    pip install pyscf
    pip install -e . -C cmake.define.CUPYCCX_CUDA=ON -C cmake.define.CUPYCCX_CUDA_ARCH=<arch>
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from pyscf import gto, scf
from cupyccx.scf_data import prepare_from_pyscf
from cupyccx.method import CCD, DCD, CCOptions

PROFILE_ITERS = 5   # fixed iteration count — do not converge

def build_mol(basis):
    return gto.M(
        atom="N 0 0 0; N 0 0 2.118",
        basis=basis,
        unit="Bohr",
        verbose=0,
    )

def run_fixed(cls, data, n_iter, label):
    opts = CCOptions(
        use_gpu=True,
        max_iter=n_iter,
        conv_energy=0.0,   # never converge early on energy
        conv_amp=0.0,      # never converge early on amplitudes
    )
    t0 = time.perf_counter()
    r  = cls.from_scf_data(data, opts=opts).compute(e_scf=data.e_scf, verbose=False)
    dt = time.perf_counter() - t0
    print(f"  {label:30s}  n_occ={data.n_occ:3d}  n_vir={data.n_vir:3d}"
          f"  {n_iter} iters  {dt:.3f}s")

def main():
    systems = [
        ("sto-3g",   "N2/STO-3G   (small)"),
        ("cc-pVDZ",  "N2/cc-pVDZ  (medium)"),
        ("cc-pVTZ",  "N2/cc-pVTZ  (large)"),
    ]

    print(f"Running {PROFILE_ITERS} GPU iterations per method/system")
    print(f"{'Method':<6}  {'System':<30}  {'n_occ':>5}  {'n_vir':>5}  {'iters':>5}  {'wall':>8}")
    print("-" * 72)

    for basis, label in systems:
        mol  = build_mol(basis)
        mf   = scf.RHF(mol).run()
        data = prepare_from_pyscf(mf, verbose=False)

        run_fixed(CCD, data, PROFILE_ITERS, f"CCD  {label}")
        run_fixed(DCD, data, PROFILE_ITERS, f"DCD  {label}")

    print("\nDone. Open the .nsys-rep file in Nsight Systems to inspect the timeline.")

if __name__ == "__main__":
    main()
