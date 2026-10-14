"""Condensed homogeneous plane-strain elasticity on uniform periodic grids."""

import numpy as np


def coherent_elasticity(grid, phase, misfit=0.08, lam=2.0, shear=1.0, mean_strain=0.0):
    grid.spectrum()  # Validate spatial/boundary assumptions.
    kx = 2 * np.pi * np.fft.fftfreq(len(grid.x), grid.dx[0])
    ky = 2 * np.pi * np.fft.fftfreq(len(grid.y), grid.dy[0])
    # Real spectral derivatives have no representable Nyquist sine mode.
    if len(kx) % 2 == 0:
        kx[len(kx) // 2] = 0
    if len(ky) % 2 == 0:
        ky[len(ky) // 2] = 0
    kx, ky = np.meshgrid(kx, ky)
    kk = kx * kx + ky * ky
    eigen = misfit * (phase + 1) / 2
    eigen_hat = np.fft.fft2(eigen)
    # Three-dimensional isotropic material, with eps_zz=0 and eps*_zz=eigen.
    source = (2 * shear + 3 * lam) * eigen_hat
    denom = (lam + 2 * shear) * kk
    inv = np.divide(1.0, denom, out=np.zeros_like(denom), where=denom > 0)
    ux, uy = -1j * kx * source * inv, -1j * ky * source * inv
    exx = np.fft.ifft2(1j * kx * ux).real + mean_strain - eigen
    eyy = np.fft.ifft2(1j * ky * uy).real + mean_strain - eigen
    exy = np.fft.ifft2(0.5j * (ky * ux + kx * uy)).real
    ezz = -eigen
    trace = exx + eyy + ezz
    sxx = 2 * shear * exx + lam * trace
    syy = 2 * shear * eyy + lam * trace
    szz = 2 * shear * ezz + lam * trace
    sxy = 2 * shear * exy
    density = shear * (exx * exx + eyy * eyy + ezz * ezz + 2 * exy * exy) + lam * trace * trace / 2
    force = -misfit * (sxx + syy + szz) / 2
    divx = np.fft.ifft2(1j * (kx * np.fft.fft2(sxx) + ky * np.fft.fft2(sxy))).real
    divy = np.fft.ifft2(1j * (kx * np.fft.fft2(sxy) + ky * np.fft.fft2(syy))).real
    return {"energy": grid.integrate(density), "force": force,
            "sxx": sxx, "syy": syy, "szz": szz, "sxy": sxy,
            "exx": exx + eigen, "eyy": eyy + eigen, "exy": exy,
            "equilibrium_residual": float(np.sqrt(np.mean(divx ** 2 + divy ** 2)))}
