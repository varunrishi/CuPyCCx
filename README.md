# cupyccx

**Author:** Varun Rishi

**Coupled Cluster Doubles** — C++ implementations with CUDA GPU acceleration and Python bindings, including a PySCF interface.

## Methods

| Class  | Description |
|--------|-------------|
| `LCCD` | Linearized CCD — quadratic T₂ terms dropped |
| `CCD`  | Full Coupled Cluster Doubles |

Both methods work in the **spin-orbital** basis and use antisymmetrized ERIs `<pq‖rs>`.

## Repository layout

```
cupyccx/
├── include/cupyccx/       # C++ headers (types, integrals, solvers, tensor ops)
├── src/
│   ├── cpp/               # CCD/LCCD drivers, CPU tensor contractions (BLAS)
│   ├── cuda/              # cuBLAS GPU tensor contractions
│   └── python/            # pybind11 bindings
├── python/cupyccx/        # Python package (method.py, pyscf_interface.py)
├── tests/
│   ├── cpp/               # GoogleTest unit tests
│   └── python/            # pytest tests
├── examples/
│   ├── basic_ccd.py       # standalone (no PySCF required)
│   └── pyscf_h2o.py       # H₂O CCD via PySCF integrals
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

### GPU acceleration

```python
from cupyccx.method import CCD, CCOptions

result = CCD(..., opts=CCOptions(use_gpu=True)).compute(verbose=True)
```

## Running tests

```bash
# Python tests
pytest tests/python/ -v

# C++ tests (after CMake build)
ctest --test-dir build --output-on-failure
```

## Design notes

- **Tensor contractions** are routed through the `tensor_ops` dispatcher (`include/cupyccx/tensor_ops.hpp`). CPU calls use CBLAS; GPU calls use cuBLAS DGEMM and fall back to CPU loops where a DGEMM reformulation requires a tensor transpose not yet implemented.
- **DIIS** extrapolation is on by default (`CCOptions.use_diis = True`, up to 6 vectors).
- **MP2 amplitudes** are used as the initial guess for T₂.
- The Python `CCOptions` dataclass mirrors the C++ `CCOptions` struct and is converted by `_to_ext()` before being passed to the compiled extension.

## License

MIT
