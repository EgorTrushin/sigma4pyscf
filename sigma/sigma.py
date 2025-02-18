#!/usr/bin/env python3
import json
import os
import pickle
import numpy as np
from numpy import linalg
from pyscf import scf
from pyscf.gw.rpa import (
    RPA,
    _get_scaled_legendre_roots,
)


def kernel(rpa, eris=None, nw=50, x0=2.5, pkl=None):
    """
    RPA and sigma-functional correlation and total energy

    Args:
        eris : Cholesky decomposed ERI in OV subspace
        nw : number of frequency point on imaginary axis
        x0: scaling factor for frequency grid
        pkl: name of pkl-file to store sigma-values and other relevant data

    Returns:
        e_hf : EXX energy
        e_corr : sigma-functional correlation energy
        e_corr_rpa : RPA correlation energy
    """

    if eris is None:
        eris = rpa.ao2mo()

    # Compute exact exchange energy (EXX)
    e_hf = rpa.get_e_hf()

    # Grids for integration on imaginary axis
    freqs, wts = _get_scaled_legendre_roots(nw, x0)

    # Compute correlation energy
    nw = len(freqs)
    x, c = get_spline_coeffs(rpa)
    if pkl is not None:
        allsigmas = []
    e_corr_rpa = 0.0
    e_corr_sigma = 0.0
    e_ov = rpa.make_e_ov()
    f_ov = rpa.make_f_ov()
    for w in range(nw):
        Pi = rpa.make_dielectric_matrix(freqs[w], e_ov, f_ov, eris, blksize=None)
        sigmas, _ = linalg.eigh(-Pi)
        if pkl is not None:
            allsigmas.append(sigmas)
        ec_w_rpa = 0.0
        ec_w_sigma = 0.0
        for sigma in sigmas:
            if sigma > 0.0:
                ec_w_rpa += np.log(1.0 + sigma) - sigma
                ec_w_sigma += -cspline_integr(c, x, sigma)
            else:
                assert abs(sigma) < 1.0e-14
        e_corr_rpa += 1.0 / (2.0 * np.pi) * ec_w_rpa * wts[w]
        e_corr_sigma += 1.0 / (2.0 * np.pi) * ec_w_sigma * wts[w]

    e_corr_sigma += e_corr_rpa

    if pkl is not None:
        with open(pkl, "wb") as fileObj:
            pickle.dump(e_hf, fileObj)
            pickle.dump(wts, fileObj)
            pickle.dump(allsigmas, fileObj)

    return e_hf, e_corr_sigma, e_corr_rpa


class SIGMA(RPA):
    def __init__(self, mf, frozen=None, auxbasis=None, param=None):
        super().__init__(mf, frozen, auxbasis)

        self.param = param

        self.e_corr_rpa = None
        self.e_tot_rpa = None

    def kernel(self, eris=None, nw=50, x0=2.5, pkl=None):
        """
        Args:
            eris : Cholesky decomposed ERI in OV subspace
            nw : number of frequency point on imaginary axis
            x0: scaling factor for frequency grid
            pkl: name of pkl-file to store sigma-values and other relevant data
        """

        res = kernel(self, eris=eris, nw=nw, x0=x0, pkl=pkl)

        self.e_hf, self.e_corr, self.e_corr_rpa = res

        self._finalize()
        self.e_tot_rpa = self.e_hf + self.e_corr_rpa

        return self.e_corr


# read x and c for parametrizations from json-file
with open(os.path.dirname(os.path.realpath(__file__)) + "/json/params.json") as file_obj:
    params = json.load(file_obj)


def get_spline_coeffs(rpa):
    assert rpa._scf.xc in ["pbe", "pbe0", "b3lyp", "tpss"]
    if rpa.param is None and rpa._scf.xc in ["pbe", "pbe0"]:
        param_name = "_".join((rpa._scf.xc, "S1"))
    elif rpa.param is None and rpa._scf.xc in ["b3lyp", "tpss"]:
        param_name = "_".join((rpa._scf.xc, "W1"))
    else:
        param_name = "_".join((rpa._scf.xc, rpa.param))
    assert param_name in params.keys()
    return params[param_name]["x"], params[param_name]["c"]


def cspline_integr(c, x, s):
    """Integrate analytically cubic spline representation of sigma-functional
       'correction' from 0 to s.

    First interval of spline is treated as linear.
    Last interval of spline is treated as a constant.

    Args:
        c: Coefficients of spline
        x: Ordinates of spline. Have to be non-negative and increasingly order
        s: Sigma-value for which one integrate. Has to be positive.

    Returns:
        integral: resulting integral
    """
    m = np.searchsorted(x, s)  # determine to which interval s belongs

    # evaluate integral
    integral = 0.0
    if m == 1:
        integral = 0.5 * c[1][0] * s
    if m > 1 and m < len(x):
        h = s - x[m - 1]
        integral = (
            0.5 * c[1][0] * x[1] ** 2 / s
            + (c[0][m - 1] * h + c[1][m - 1] / 2.0 * h ** 2 + c[2][m - 1] / 3.0 * h ** 3 + c[3][m - 1] / 4.0 * h ** 4)
            / s
        )
        for i in range(2, m):
            h = x[i] - x[i - 1]
            integral += (
                c[0][i - 1] * h + c[1][i - 1] / 2.0 * h ** 2 + c[2][i - 1] / 3.0 * h ** 3 + c[3][i - 1] / 4.0 * h ** 4
            ) / s
    if m == len(x):
        integral = 0.5 * c[1][0] * x[1] ** 2 / s
        for i in range(2, m):
            h = x[i] - x[i - 1]
            integral += (
                c[0][i - 1] * h + c[1][i - 1] / 2.0 * h ** 2 + c[2][i - 1] / 3.0 * h ** 3 + c[3][i - 1] / 4.0 * h ** 4
            ) / s
        integral += c[0][-1] * (1.0 - x[-1] / s)

    return integral * s
