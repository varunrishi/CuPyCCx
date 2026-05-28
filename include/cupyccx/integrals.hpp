#pragma once

#include "types.hpp"
#include <stdexcept>

namespace cupyccx {

// ---------------------------------------------------------------------------
// Integral slicing helpers
// Access sub-blocks of the full MO ERI tensor using chemical/physics notation
// ---------------------------------------------------------------------------

// <ij||ab>  (i,j = occ; a,b = vir) — the doubles-driving block
inline Tensor4 slice_oovv(const SCFData& scf) {
    const int o = scf.n_occ, v = scf.n_vir;
    Tensor4 oovv(o, o, v, v);
    for (int i = 0; i < o; ++i)
    for (int j = 0; j < o; ++j)
    for (int a = 0; a < v; ++a)
    for (int b = 0; b < v; ++b)
        oovv(i, j, a, b) = scf.eri_antisym(i, j, o + a, o + b);
    return oovv;
}

// <ij|ab> plain Coulomb (not antisymmetrized) — used by DCD/pCCD for D_c and D_x
inline Tensor4 slice_oovv_plain(const SCFData& scf) {
    const int o = scf.n_occ, v = scf.n_vir;
    Tensor4 oovv(o, o, v, v);
    for (int i = 0; i < o; ++i)
    for (int j = 0; j < o; ++j)
    for (int a = 0; a < v; ++a)
    for (int b = 0; b < v; ++b)
        oovv(i, j, a, b) = scf.eri_plain(i, j, o + a, o + b);
    return oovv;
}

// <ab||cd>  (a,b,c,d = vir) — ladder diagram integral
inline Tensor4 slice_vvvv(const SCFData& scf) {
    const int o = scf.n_occ, v = scf.n_vir;
    Tensor4 vvvv(v, v, v, v);
    for (int a = 0; a < v; ++a)
    for (int b = 0; b < v; ++b)
    for (int c = 0; c < v; ++c)
    for (int d = 0; d < v; ++d)
        vvvv(a, b, c, d) = scf.eri_antisym(o+a, o+b, o+c, o+d);
    return vvvv;
}

// <kl||ij>  (k,l,i,j = occ) — ring diagram integral
inline Tensor4 slice_oooo(const SCFData& scf) {
    const int o = scf.n_occ;
    Tensor4 oooo(o, o, o, o);
    for (int k = 0; k < o; ++k)
    for (int l = 0; l < o; ++l)
    for (int i = 0; i < o; ++i)
    for (int j = 0; j < o; ++j)
        oooo(k, l, i, j) = scf.eri_antisym(k, l, i, j);
    return oooo;
}

// <kb||cj>  (k = occ, b = vir, c = vir, j = occ) — mixed block for ring+ladder
inline Tensor4 slice_ovvo(const SCFData& scf) {
    const int o = scf.n_occ, v = scf.n_vir;
    Tensor4 ovvo(o, v, v, o);
    for (int k = 0; k < o; ++k)
    for (int b = 0; b < v; ++b)
    for (int c = 0; c < v; ++c)
    for (int j = 0; j < o; ++j)
        ovvo(k, b, c, j) = scf.eri_antisym(k, o+b, o+c, j);
    return ovvo;
}

// Orbital-energy denominator  D_{ij}^{ab} = eps_i + eps_j - eps_a - eps_b
inline Tensor4 make_D2(const SCFData& scf) {
    const int o = scf.n_occ, v = scf.n_vir;
    Tensor4 D(o, o, v, v);
    for (int i = 0; i < o; ++i)
    for (int j = 0; j < o; ++j)
    for (int a = 0; a < v; ++a)
    for (int b = 0; b < v; ++b)
        D(i, j, a, b) = scf.eps(i) + scf.eps(j) - scf.eps(o + a) - scf.eps(o + b);
    return D;
}

// Validate SCFData dimensions for sanity
inline void validate_scf(const SCFData& scf) {
    if (scf.n_occ <= 0 || scf.n_vir <= 0)
        throw std::invalid_argument("n_occ and n_vir must be positive");
    if (scf.eps.size() != static_cast<Eigen::Index>(scf.n_mo))
        throw std::invalid_argument("eps length does not match n_mo");
    if (scf.eri_antisym.n0 != static_cast<std::size_t>(scf.n_mo))
        throw std::invalid_argument("ERI tensor dimension does not match n_mo");
}

}  // namespace cupyccx
