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

The script runs a fixed number of iterations (not to convergence) on
progressively larger systems so that the profiler captures a representative
mix of DGEMM sizes and host↔device transfer overhead.

Four implementations are profiled side-by-side:
  - C++ CCD  (cuBLAS DGEMMs via custom CUDA kernels, dressed W tensors on GPU)
  - C++ DCD  (cuBLAS; bare W_vvvv/W_oooo cached once per molecule)
  - PyCCD    (CuPy einsum — same diagrams as C++ CCD)
  - PyDCD    (CuPy einsum — same diagrams as C++ DCD)

This lets you compare kernel dispatch overhead, memory traffic patterns, and
DGEMM utilisation between the hand-written CUDA path and the CuPy path directly
in the Nsight Systems timeline (look for nvrtc / cuBLAS ranges).

Prerequisites:
    pip install pyscf cupy-cuda12x   # or cupy-cuda11x for CUDA 11
    pip install -e . -C cmake.define.CUPYCCX_CUDA=ON -C cmake.define.CUPYCCX_CUDA_ARCH=<arch>

Use --basis/--impl to scope a run to a single system/implementation — this
is what lets you capture separate Nsight reports per implementation instead
of interleaving C++ and CuPy ranges in one shared timeline:

    nsys profile --trace=cuda,cublas,osrt --output=ccpvtz_cpp \\
        python examples/profile_gpu.py --basis cc-pVTZ --impl cpp
    nsys profile --trace=cuda,cublas,osrt --output=ccpvtz_cupy \\
        python examples/profile_gpu.py --basis cc-pVTZ --impl cupy
"""

import argparse
import time
import warnings

from pyscf import gto, scf
from cupyccx.scf_data import prepare_from_pyscf
from cupyccx.method import CCD, DCD, CCOptions
from cupyccx.py_solver import PyCCD, PyDCD

try:
    import cupy  # noqa: F401
    _CUPY_OK = True
except ImportError:
    _CUPY_OK = False
    warnings.warn("CuPy not found — PyCCD/PyDCD rows will be skipped.")

PROFILE_ITERS = 5   # fixed iteration count — do not converge

def build_mol(basis):
    return gto.M(
        atom="N 0 0 0; N 0 0 2.118",
        basis=basis,
        unit="Bohr",
        verbose=0,
    )

def run_fixed_cpp(cls, data, n_iter, label):
    """Time n_iter iterations of a C++ solver (use_gpu=True)."""
    opts = CCOptions(
        use_gpu=True,
        max_iter=n_iter,
        conv_energy=0.0,
        conv_amp=0.0,
    )
    t0 = time.perf_counter()
    cls.from_scf_data(data, opts=opts).compute(e_scf=data.e_scf, verbose=False)
    dt = time.perf_counter() - t0
    print(f"  {label:42s}  n_occ={data.n_occ:3d}  n_vir={data.n_vir:3d}"
          f"  {n_iter} iters  {dt:.3f}s")

def run_fixed_py(cls, data, n_iter, label, **kwargs):
    """Time n_iter iterations of a pure-Python CuPy solver."""
    opts = CCOptions(
        use_gpu=True,
        max_iter=n_iter,
        conv_energy=0.0,
        conv_amp=0.0,
    )
    t0 = time.perf_counter()
    cls.from_scf_data(data, opts=opts, **kwargs).compute(e_scf=data.e_scf, verbose=False)
    dt = time.perf_counter() - t0
    print(f"  {label:42s}  n_occ={data.n_occ:3d}  n_vir={data.n_vir:3d}"
          f"  {n_iter} iters  {dt:.3f}s")

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", choices=["sto-3g", "cc-pVDZ", "cc-pVTZ", "all"],
                         default="all", help="restrict the run to one system (default: all)")
    parser.add_argument("--impl", choices=["cpp", "cupy", "both"],
                         default="both", help="restrict the run to one implementation (default: both)")
    return parser.parse_args()

def main():
    args = parse_args()

    systems = [
        ("sto-3g",  "N2/STO-3G  (small)"),
        ("cc-pVDZ", "N2/cc-pVDZ (medium)"),
        ("cc-pVTZ", "N2/cc-pVTZ (large)"),
    ]
    if args.basis != "all":
        systems = [s for s in systems if s[0] == args.basis]

    print(f"Running {PROFILE_ITERS} GPU iterations per system  (basis={args.basis}, impl={args.impl})")
    print(f"{'Method + System':<42}  {'n_occ':>5}  {'n_vir':>5}  {'iters':>5}  {'wall':>8}")
    print("-" * 79)

    for basis, label in systems:
        mol  = build_mol(basis)
        mf   = scf.RHF(mol).run()
        data = prepare_from_pyscf(mf, verbose=False)

        if args.impl in ("cpp", "both"):
            run_fixed_cpp(CCD,  data, PROFILE_ITERS, f"C++  CCD  {label}")
            # run_fixed_cpp(DCD,  data, PROFILE_ITERS, f"C++  DCD  {label}")
        if args.impl in ("cupy", "both") and _CUPY_OK:
            run_fixed_py(PyCCD, data, PROFILE_ITERS, f"CuPy CCD  {label}")
            # run_fixed_py(PyDCD, data, PROFILE_ITERS, f"CuPy DCD  {label}")
        print()

    print("Done. Open the .nsys-rep file in Nsight Systems to inspect the timeline.")
    print("Filter by process name to separate C++ cuBLAS ranges from CuPy nvrtc ranges.")

if __name__ == "__main__":
    main()
