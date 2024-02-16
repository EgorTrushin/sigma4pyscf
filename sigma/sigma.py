#!/usr/bin/env python3
import json
import os
import numpy as np
from numpy import linalg
from pyscf import scf
from pyscf.gw.rpa import (
    RPA,
    get_rho_response,
    _get_scaled_legendre_roots,
    _mo_energy_without_core,
    _mo_without_core,
)


def kernel(rpa, mo_energy, mo_coeff, Lpq=None, nw=50, x0=2.5):
    """
    RPA and sigma-functional correlation and total energy

    Args:
        Lpq : density fitting 3-center integral in MO basis.
        nw : number of frequency point on imaginary axis.
        x0: scaling factor for frequency grid.

    Returns:
        e_tot : sigma-functional total energy
        e_tot_rpa : RPA total energy
        e_hf : EXX energy
        e_corr : sigma-functional correlation energy
        e_corr_rpa : RPA correlation energy
    """
    mf = rpa._scf
    # only support frozen core
    if rpa.frozen is not None:
        assert isinstance(rpa.frozen, int)
        assert rpa.frozen < rpa.nocc

    if Lpq is None:
        Lpq = rpa.ao2mo(mo_coeff)

    # Grids for integration on imaginary axis
    freqs, wts = _get_scaled_legendre_roots(nw, x0)

    # Compute HF energy (EXX)
    dm = mf.make_rdm1()
    rhf = scf.RHF(rpa.mol)
    e_hf = rhf.energy_elec(dm=dm)[0]
    e_hf += mf.energy_nuc()

    # Compute correlation energy
    e_corr_rpa, e_corr = get_ecorr(rpa, Lpq, freqs, wts)

    # Compute totol energy
    e_tot = e_hf + e_corr
    e_tot_rpa = e_hf + e_corr_rpa

    return e_tot, e_tot_rpa, e_hf, e_corr, e_corr_rpa


def get_ecorr(rpa, Lpq, freqs, wts):
    """
    Compute correlation energy
    """
    mo_energy = _mo_energy_without_core(rpa, rpa._scf.mo_energy)
    nocc = rpa.nocc
    nw = len(freqs)

    x, c = get_spline_coeffs(rpa)

    e_corr_rpa = 0.0
    e_corr_sigma = 0.0
    for w in range(nw):
        Pi = get_rho_response(freqs[w], mo_energy, Lpq[:, :nocc, nocc:])
        sigmas, _ = linalg.eigh(-Pi)
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

    return e_corr_rpa, e_corr_sigma


class SIGMA(RPA):
    def __init__(self, mf, frozen=None, auxbasis=None, param=None):
        super().__init__(mf, frozen, auxbasis)

        self.param = param

        self.e_corr_rpa = None
        self.e_tot_rpa = None

    def kernel(self, mo_energy=None, mo_coeff=None, Lpq=None, nw=50, x0=2.5):
        """
        Args:
            mo_energy : 1D array (nmo), mean-field mo energy
            mo_coeff : 2D array (nmo, nmo), mean-field mo coefficient
            Lpq : 3D array (naux, nmo, nmo), 3-index ERI
            nw: interger, grid number
            x0: real, scaling factor for frequency grid

        Returns:
            self.e_tot : sigma-functional total energy
            self.e_tot_rpa : RPA total eenrgy
            self.e_hf : EXX energy
            self.e_corr : sigma-functional correlation energy
            self.e_corr_rpa : RPA correlation energy
        """
        if mo_coeff is None:
            mo_coeff = _mo_without_core(self, self._scf.mo_coeff)
        if mo_energy is None:
            mo_energy = _mo_energy_without_core(self, self._scf.mo_energy)

        self.dump_flags()
        self.e_tot, self.e_tot_rpa, self.e_hf, self.e_corr, self.e_corr_rpa = kernel(
            self, mo_energy, mo_coeff, Lpq=Lpq, nw=nw, x0=x0
        )

        return self.e_corr


# read x and c for parametrizations from json-file
with open(os.path.dirname(os.path.realpath(__file__)) + "/json/params.json") as file_obj:
    params = json.load(file_obj)


def get_spline_coeffs(rpa):
    assert rpa._scf.xc in ["pbe", "pbe0", "b3lyp", "tpss"]
    if rpa.param is None and rpa._scf.xc in ["pbe", "pbe0"]:
        param_name = "_".join((rpa._scf.xc, "S2"))
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
            + (
                c[0][m - 1] * h
                + c[1][m - 1] / 2.0 * h**2
                + c[2][m - 1] / 3.0 * h**3
                + c[3][m - 1] / 4.0 * h**4
            )
            / s
        )
        for i in range(2, m):
            h = x[i] - x[i - 1]
            integral += (
                c[0][i - 1] * h
                + c[1][i - 1] / 2.0 * h**2
                + c[2][i - 1] / 3.0 * h**3
                + c[3][i - 1] / 4.0 * h**4
            ) / s
    if m == len(x):
        integral = 0.5 * c[1][0] * x[1] ** 2 / s
        for i in range(2, m):
            h = x[i] - x[i - 1]
            integral += (
                c[0][i - 1] * h
                + c[1][i - 1] / 2.0 * h**2
                + c[2][i - 1] / 3.0 * h**3
                + c[3][i - 1] / 4.0 * h**4
            ) / s
        integral += c[0][-1] * (1.0 - x[-1] / s)

    return integral * s
