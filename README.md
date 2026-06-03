<p align="center">
  <img src="logo.svg" alt="CuPyCCx" width="420"/>
</p>

<p align="center">
  High-performance <strong>Coupled Cluster Doubles</strong> — CCD, LCCD, DCD, and pCCD —<br/>
  accelerated in C++ and CUDA, with a clean Python interface and seamless <a href="https://pyscf.org">PySCF</a> integration.<br/><br/>
  <strong>Author: <a href="https://github.com/varunrishi">Varun Rishi</a></strong>
</p>

<p align="center">
  <a href="https://github.com/varunrishi/CuPyCCx/actions/workflows/ci.yml">
    <img src="https://github.com/varunrishi/CuPyCCx/actions/workflows/ci.yml/badge.svg" alt="CI"/>
  </a>
  <a href="https://doi.org/10.1063/1.4944087">
    <img src="https://img.shields.io/badge/DOI-10.1063%2F1.4944087-blue?style=flat-square" alt="DOI"/>
  </a>
  <a href="https://colab.research.google.com/github/varunrishi/CuPyCCx/blob/master/examples/colab_gpu_test.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/>
  </a>
</p>

## Methods

| Class  | Description |
|--------|-------------|
| `LCCD` | Linearized CCD — quadratic T₂ terms dropped |
| `CCD`  | Full Coupled Cluster Doubles |
| `DCD`  | Distinguishable Cluster Doubles (Kats & Manby 2013) — three explicit quadratic terms with plain Coulomb ERIs |
| `pCCD` | Parameterized CCD — interpolates between LCCD and CCD via α, β |

All methods work in the **spin-orbital** basis. `CCD` and `LCCD` use antisymmetrized ERIs `<pq‖rs>`; `DCD` and `pCCD` additionally require plain Coulomb ERIs `<pq|rs>` (provided automatically by `prepare_from_pyscf`).

The DCD and pCCD implementations follow [[1]](#references): V. Rishi, A. Perera, R. J. Bartlett, *J. Chem. Phys.* **144**, 124117 (2016).

### pCCD parameters

`pCCD` scales the quadratic T₂·T₂ diagrams via two parameters:

```
Quadratic = A/2 + α·(A/2 + B) + β·(C + D)
```

where A, B, C are ladder/mixed diagrams and D = D_c + D_x is the full ring diagram.

| α | β | Equivalent to |
|---|---|---------------|
| 0 | 0 | A/2 only |
| 0 | 1 | A/2 + C + full ring D |
| 1 | 0 | A + B (ladder only) |
| 1 | 1 | Full CCD |

## Repository layout

```
CuPyCCx/
├── include/cupyccx/       # C++ headers (types, integrals, solvers, tensor ops)
├── src/
│   ├── cpp/               # CCD/LCCD drivers, CPU tensor contractions (BLAS)
│   ├── cuda/              # cuBLAS GPU tensor contractions
│   └── python/            # pybind11 bindings
├── python/cupyccx/        # Python package (method.py, py_solver.py, pyscf_interface.py)
├── tests/
│   ├── cpp/               # GoogleTest unit tests
│   └── python/            # pytest tests
├── examples/
│   ├── basic_ccd.py       # standalone (no PySCF required)
│   ├── pyscf_h2o.py       # H₂O CCD/LCCD via PySCF integrals
│   └── n2_sto3g.py        # N₂/STO-3G: all four methods with reference energies
├── CMakeLists.txt
└── pyproject.toml
```

## Build

### Requirements

- CMake ≥ 3.20
- C++17 compiler (GCC ≥ 11 or Clang ≥ 14)
- Eigen3 ≥ 3.4
- BLAS (OpenBLAS or MKL recommended)
- Python ≥ 3.9 + pybind11 ≥ 2.12
- *(optional)* CUDA Toolkit ≥ 11.8 for GPU support

### CPU-only (recommended starting point)

```bash
pip install -e ".[dev]"
```

or with CMake directly:

```bash
cmake -B build -DCUPYCCX_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

### GPU build

```bash
pip install -e . -C cmake.define.CUPYCCX_CUDA=ON \
                 -C cmake.define.CUPYCCX_CUDA_ARCH=80   # A100 = sm_80
```

or via CMake:

```bash
cmake -B build -DCUPYCCX_CUDA=ON -DCUPYCCX_CUDA_ARCH=80
cmake --build build -j$(nproc)
```

## Quick-start

### Standalone (no PySCF)

```python
import numpy as np
from cupyccx.method import CCD, CCOptions

# Supply spin-orbital integrals as numpy arrays
result = CCD(n_occ, n_vir, eps, fock, eri_antisym,
             opts=CCOptions(use_gpu=False)).compute(e_scf=e_hf, verbose=True)

print(f"E_corr = {result.e_corr:.10f} Ha")
print(f"T2 shape = {result.t2.shape}")   # (n_occ, n_occ, n_vir, n_vir)
```

### With PySCF

```python
from pyscf import gto, scf
from cupyccx.pyscf_interface import run_from_pyscf

mol = gto.M(atom="O 0 0 0; H 0 0.96 0; H 0.76 -0.48 0", basis="cc-pVDZ")
mf  = scf.RHF(mol).run()

result = run_from_pyscf(mf, method="CCD", verbose=True)
print(f"CCD E_corr = {result.e_corr:.10f} Ha")
```

### DCD via `run_from_pyscf`

```python
result = run_from_pyscf(mf, method="DCD", verbose=True)
print(f"DCD E_corr = {result.e_corr:.10f} Ha")
```

DCD replaces the CCD ring-ring quadratic term with three terms from Kats & Manby (2013):
two using antisymmetric ERIs with P(ij) and P(ab) permutations, and one using plain
Coulomb ERIs `<pq|rs>` with the full P(ij)P(ab) permutation.

### pCCD via `run_from_pyscf`

Pass `alpha` and `beta` directly:

```python
result = run_from_pyscf(mf, method="pCCD", alpha=0.5, beta=0.8, verbose=True)
print(f"pCCD E_corr = {result.e_corr:.10f} Ha")
```

### pCCD via `from_scf_data` (more control)

```python
from cupyccx.scf_data import prepare_from_pyscf
from cupyccx.method import pCCD, CCOptions

data = prepare_from_pyscf(mf)
opts = CCOptions(max_iter=200, conv_energy=1e-9)

result = pCCD.from_scf_data(data, alpha=0.5, beta=0.8, opts=opts).compute(
    e_scf=data.e_scf, verbose=True
)
print(f"pCCD E_corr = {result.e_corr:.10f} Ha")
```

### GPU acceleration

```python
from cupyccx.method import CCD, CCOptions

result = CCD(..., opts=CCOptions(use_gpu=True)).compute(verbose=True)
```

> **Try it on a free GPU →** [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/varunrishi/CuPyCCx/blob/master/examples/colab_gpu_test.ipynb)
>
> The notebook builds CuPyCCx with CUDA, runs CPU vs GPU correctness checks for CCD, DCD, and pCCD on N₂, and benchmarks the speedup on a larger cc-pVDZ system.

## Running tests

```bash
# Python tests
pytest tests/python/ -v

# C++ tests (after CMake build)
ctest --test-dir build --output-on-failure
```

## GPU backends

CuPyCCx provides two independent GPU implementations that can be compared side-by-side:

### Native CUDA (C++ extension)

The primary path, activated by `CCOptions(use_gpu=True)` with a CUDA build.
Tensor contractions are routed through the `tensor_ops` dispatcher
(`include/cupyccx/tensor_ops.hpp`) to hand-written cuBLAS DGEMM kernels in
`src/cuda/tensor_ops_gpu.cu`. Key optimisations:

- **Integral caching** — `gpu_upload_integrals()` uploads static W tensors (vvvv, oooo, ovvo) to the device once per molecule; DCD reuses them across all iterations.
- **W_vvvv elimination** — CCD avoids building the O(v⁴) dressed W_vvvv on CPU; instead the Q_B correction is computed as two back-to-back DGEMMs on the device (`contract_oovv_t2_t2`).
- **Ring P(ij)P(ab)** — all four permutation terms are fused into a single DGEMM + scatter kernel (`k_scatter_Pijab`), avoiding four separate passes.

### Pure-Python CuPy

A transparent reference implementation in `python/cupyccx/py_solver.py`, also activated via `CCOptions(use_gpu=True)` (requires `pip install cupy-cuda12x`). Uses `xp.einsum` throughout — every contraction is annotated with the corresponding C++ line in `ccd.cpp` so energies and T₂ amplitudes can be cross-checked iteration-by-iteration.

| Class   | GPU backend |
|---------|-------------|
| `CCD`   | cuBLAS DGEMM (C++ extension) |
| `DCD`   | cuBLAS DGEMM (C++ extension) |
| `PyCCD` | CuPy einsum |
| `PyDCD` | CuPy einsum |

```python
from cupyccx.method import CCD, CCOptions
from cupyccx.py_solver import PyCCD

# C++ cuBLAS path
r_cpp = CCD.from_scf_data(data, opts=CCOptions(use_gpu=True)).compute(data.e_scf)

# CuPy einsum path (same answer, useful for debugging)
r_py  = PyCCD.from_scf_data(data, opts=CCOptions(use_gpu=True)).compute(data.e_scf)

print(abs(r_cpp.e_corr - r_py.e_corr))   # < 1e-8 Ha
```

Run `examples/profile_gpu.py` under Nsight Systems to see both paths in the same timeline — the C++ path appears as tight cuBLAS ranges; the CuPy path as nvrtc JIT compilation followed by cuBLAS.

## Design notes

- **Tensor contractions** are routed through the `tensor_ops` dispatcher (`include/cupyccx/tensor_ops.hpp`). CPU calls use CBLAS; GPU calls use cuBLAS DGEMM.
- **DIIS** extrapolation is on by default (`CCOptions.use_diis = True`, up to 6 vectors).
- **MP2 amplitudes** are used as the initial guess for T₂.
- The Python `CCOptions` dataclass mirrors the C++ `CCOptions` struct and is converted by `_to_ext()` before being passed to the compiled extension.

## References

<a name="references"></a>

[1] V. Rishi, A. Perera, R. J. Bartlett, "Assessing the distinguishable cluster approximation based on the triple bond-breaking in the nitrogen molecule," *J. Chem. Phys.* **144**, 124117 (2016). https://doi.org/10.1063/1.4944087

## License

MIT
