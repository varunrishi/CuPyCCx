#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cupyccx/ccd.hpp"
#include "cupyccx/integrals.hpp"
#include "cupyccx/types.hpp"

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Helper: convert a 4-D numpy array (shape [n0,n1,n2,n3]) to Tensor4
// ---------------------------------------------------------------------------
static cupyccx::Tensor4 numpy_to_tensor4(const py::array_t<double,
                                          py::array::c_style | py::array::forcecast>& arr) {
    if (arr.ndim() != 4)
        throw std::invalid_argument("Expected a 4-D array");
    cupyccx::Tensor4 t(arr.shape(0), arr.shape(1), arr.shape(2), arr.shape(3));
    std::memcpy(t.ptr(), arr.data(), t.size() * sizeof(double));
    return t;
}

// Helper: convert Tensor4 → numpy array (zero-copy via buffer protocol)
static py::array_t<double> tensor4_to_numpy(const cupyccx::Tensor4& t) {
    return py::array_t<double>(
        {t.n0, t.n1, t.n2, t.n3},
        {t.n1 * t.n2 * t.n3 * sizeof(double),
               t.n2 * t.n3 * sizeof(double),
                      t.n3 * sizeof(double),
                             sizeof(double)},
        t.ptr());
}

// ---------------------------------------------------------------------------
// Build an SCFData from numpy inputs (Python-facing constructor)
// ---------------------------------------------------------------------------
static cupyccx::SCFData make_scf_data(int n_occ, int n_vir,
                                       const py::array_t<double>& eps_arr,
                                       const py::array_t<double>& fock_arr,
                                       const py::array_t<double>& eri_arr) {
    cupyccx::SCFData scf;
    scf.n_occ = n_occ;
    scf.n_vir = n_vir;
    scf.n_mo  = n_occ + n_vir;

    // Orbital energies
    {
        auto buf = eps_arr.unchecked<1>();
        scf.eps.resize(scf.n_mo);
        for (int p = 0; p < scf.n_mo; ++p) scf.eps(p) = buf(p);
    }

    // Fock matrix
    {
        auto buf = fock_arr.unchecked<2>();
        scf.fock.resize(scf.n_mo, scf.n_mo);
        for (int p = 0; p < scf.n_mo; ++p)
        for (int q = 0; q < scf.n_mo; ++q)
            scf.fock(p, q) = buf(p, q);
    }

    // ERI in antisymmetrized physicist notation <pq||rs>
    {
        auto buf = eri_arr.unchecked<4>();
        scf.eri_antisym = cupyccx::Tensor4(scf.n_mo, scf.n_mo, scf.n_mo, scf.n_mo);
        for (int p = 0; p < scf.n_mo; ++p)
        for (int q = 0; q < scf.n_mo; ++q)
        for (int r = 0; r < scf.n_mo; ++r)
        for (int s = 0; s < scf.n_mo; ++s)
            scf.eri_antisym(p, q, r, s) = buf(p, q, r, s);
    }

    cupyccx::validate_scf(scf);
    return scf;
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------
PYBIND11_MODULE(_cupyccx, m) {
    m.doc() = "cupyccx — Coupled Cluster Doubles (C++/CUDA) Python bindings";

    // ------------------------------------------------------------------
    // CCOptions
    // ------------------------------------------------------------------
    py::class_<cupyccx::CCOptions>(m, "CCOptions")
        .def(py::init<>())
        .def_readwrite("max_iter",    &cupyccx::CCOptions::max_iter)
        .def_readwrite("conv_energy", &cupyccx::CCOptions::conv_energy)
        .def_readwrite("conv_amp",    &cupyccx::CCOptions::conv_amp)
        .def_readwrite("use_diis",    &cupyccx::CCOptions::use_diis)
        .def_readwrite("diis_size",   &cupyccx::CCOptions::diis_size)
        .def_readwrite("use_gpu",     &cupyccx::CCOptions::use_gpu)
        .def_readwrite("method",      &cupyccx::CCOptions::method)
        .def("__repr__", [](const cupyccx::CCOptions& o) {
            return "<CCOptions method=" + o.method
                 + " max_iter=" + std::to_string(o.max_iter)
                 + " use_gpu=" + (o.use_gpu ? "True" : "False") + ">";
        });

    // ------------------------------------------------------------------
    // CCResult
    // ------------------------------------------------------------------
    py::class_<cupyccx::CCResult>(m, "CCResult")
        .def_readonly("e_corr",    &cupyccx::CCResult::e_corr)
        .def_readonly("e_total",   &cupyccx::CCResult::e_total)
        .def_readonly("n_iter",    &cupyccx::CCResult::n_iter)
        .def_readonly("converged", &cupyccx::CCResult::converged)
        .def_property_readonly("t2", [](const cupyccx::CCResult& r) {
            return tensor4_to_numpy(r.t2);
        })
        .def("__repr__", [](const cupyccx::CCResult& r) {
            return "<CCResult e_corr=" + std::to_string(r.e_corr)
                 + " converged=" + (r.converged ? "True" : "False")
                 + " n_iter=" + std::to_string(r.n_iter) + ">";
        });

    // ------------------------------------------------------------------
    // SCFData factory function (exposed as a Python function)
    // ------------------------------------------------------------------
    m.def("make_scf_data", &make_scf_data,
          py::arg("n_occ"), py::arg("n_vir"),
          py::arg("eps"), py::arg("fock"), py::arg("eri_antisym"),
          R"doc(
Construct an SCFData object from numpy arrays.

Parameters
----------
n_occ : int
    Number of occupied spin-orbitals.
n_vir : int
    Number of virtual spin-orbitals.
eps : ndarray, shape (n_mo,)
    Orbital energies in MO basis (occupied first, then virtual).
fock : ndarray, shape (n_mo, n_mo)
    Fock matrix in MO basis.
eri_antisym : ndarray, shape (n_mo, n_mo, n_mo, n_mo)
    Antisymmetrized two-electron integrals <pq||rs> = <pq|rs> - <pq|sr>
    in physicist notation, MO basis.
)doc");

    // ------------------------------------------------------------------
    // Top-level run functions (preferred Python API)
    // ------------------------------------------------------------------
    m.def("run_lccd",
        [](int n_occ, int n_vir,
           const py::array_t<double>& eps,
           const py::array_t<double>& fock,
           const py::array_t<double>& eri,
           const cupyccx::CCOptions&  opts,
           double e_scf,
           py::object callback) -> cupyccx::CCResult
        {
            auto scf = make_scf_data(n_occ, n_vir, eps, fock, eri);
            cupyccx::CCD solver(scf, opts, "LCCD");
            if (!callback.is_none()) {
                solver.set_callback([&callback](int it, double e, double de, double rms) {
                    callback(it, e, de, rms);
                });
            }
            return solver.compute(e_scf);
        },
        py::arg("n_occ"), py::arg("n_vir"),
        py::arg("eps"), py::arg("fock"), py::arg("eri_antisym"),
        py::arg("opts")     = cupyccx::CCOptions{},
        py::arg("e_scf")    = 0.0,
        py::arg("callback") = py::none(),
        R"doc(Run a Linearized CCD (LCCD) calculation and return the CCResult.)doc");

    m.def("run_ccd",
        [](int n_occ, int n_vir,
           const py::array_t<double>& eps,
           const py::array_t<double>& fock,
           const py::array_t<double>& eri,
           const cupyccx::CCOptions&  opts,
           double e_scf,
           py::object callback) -> cupyccx::CCResult
        {
            auto scf = make_scf_data(n_occ, n_vir, eps, fock, eri);
            cupyccx::CCD solver(scf, opts);
            if (!callback.is_none()) {
                solver.set_callback([&callback](int it, double e, double de, double rms) {
                    callback(it, e, de, rms);
                });
            }
            return solver.compute(e_scf);
        },
        py::arg("n_occ"), py::arg("n_vir"),
        py::arg("eps"), py::arg("fock"), py::arg("eri_antisym"),
        py::arg("opts")     = cupyccx::CCOptions{},
        py::arg("e_scf")    = 0.0,
        py::arg("callback") = py::none(),
        R"doc(Run a full CCD calculation and return the CCResult.)doc");

    m.def("run_pccd",
        [](int n_occ, int n_vir,
           const py::array_t<double>& eps,
           const py::array_t<double>& fock,
           const py::array_t<double>& eri,
           double alpha,
           double beta,
           const cupyccx::CCOptions&  opts,
           double e_scf,
           py::object callback) -> cupyccx::CCResult
        {
            auto scf = make_scf_data(n_occ, n_vir, eps, fock, eri);
            cupyccx::pCCD solver(scf, opts, alpha, beta);
            if (!callback.is_none()) {
                solver.set_callback([&callback](int it, double e, double de, double rms) {
                    callback(it, e, de, rms);
                });
            }
            return solver.compute(e_scf);
        },
        py::arg("n_occ"), py::arg("n_vir"),
        py::arg("eps"), py::arg("fock"), py::arg("eri_antisym"),
        py::arg("alpha")    = 1.0,
        py::arg("beta")     = 1.0,
        py::arg("opts")     = cupyccx::CCOptions{},
        py::arg("e_scf")    = 0.0,
        py::arg("callback") = py::none(),
        R"doc(Run a parameterized CCD calculation and return the CCResult.

Quadratic in CCD  = A + B + C + D
Quadratic in pCCD = A/2 + alpha*(A/2 + B) + beta*(C + D)

At alpha=1, beta=1 recovers full CCD.)doc");
}
