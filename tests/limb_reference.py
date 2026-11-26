"""Independent lens-centered quadrature for finite-source regression tests."""
from functools import lru_cache

import numpy as np


@lru_cache(None)
def nodes(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return (x + 1) / 2, w / 2


def magnification(u, rho, a, order=40):
    """I(mu)/I(1) = 1-a+a*mu. The unlensed mean brightness is 1-a/3."""
    u = np.asarray(u)
    z = u / rho
    result = np.ones_like(u, dtype=float)
    x, w = nodes(order)
    angle = np.pi*x
    v = (1-np.cos(angle))/2
    dv = np.pi*np.sin(angle)*w/2
    # Beyond this radius even the closest surface element has A-1 < 8e-11.
    active = u-rho < 400
    far = active & (z >= 8)
    mu, weights = nodes(8)
    cosphi = np.cos(np.arange(16)*np.pi/8)
    brightness = 2*mu*weights*(1-a+a*mu)/(1-a/3)
    for ids in np.array_split(np.flatnonzero(far), max(1, np.count_nonzero(far)//256)):
        radial = rho*np.sqrt(1-mu**2)
        s = u[ids,None,None]**2 + radial[:,None]**2 - 2*u[ids,None,None]*radial[:,None]*cosphi
        result[ids] = np.mean((s+2)/np.sqrt(s*(s+4)), axis=2) @ brightness
    # Lens-centered rays cancel the point-lens singularity against the
    # radial area element. All ray lengths below are in source-radius units.
    for inside in (True, False):
        mask = active & (z < 8) & ((z < 1) == inside)
        for ids in np.array_split(np.flatnonzero(mask), max(1, np.count_nonzero(mask)//32)):
            if ids.size == 0:
                continue
            zz = z[ids,None]
            if inside:
                theta = np.pi*np.r_[v/2, (1+v)/2]
                angular = np.tile(dv, 2)
                center = zz*np.cos(theta)
                half = np.sqrt(np.maximum(0, (1-zz*np.sin(theta))*(1+zz*np.sin(theta))))
            else:
                phi = np.pi*x/2
                half = np.cos(phi)
                center = np.sqrt((zz-np.sin(phi))*(zz+np.sin(phi)))
                angular = w*half/center
            lo = center-half
            start = np.maximum(lo, 0)
            length = np.where(lo >= 0, 2*half, center+half)
            r = start[...,None] + length[...,None]*v
            mu_ray = np.sqrt(np.maximum(0, (r-lo[...,None])*length[...,None]*(1-v)))
            s = (rho*r)**2
            radial_int = ((s+2)/np.sqrt(s+4)*(1-a+a*mu_ray)) @ dv
            result[ids] = np.sum(angular*length*radial_int, axis=-1)/(rho*(1-a/3))
    return result
