#!/usr/bin/env python
import pickle
import numpy as np
from numpy import linalg
from pyscf import scf
from pyscf.gw.urpa import URPA, get_rho_response, _mo_energy_without_core, _mo_without_core
from .sigma import _get_scaled_legendre_roots, cspline_integr, get_spline_coeffs


def kernel(rpa, mo_energy, mo_coeff, Lpq=None, pkl=None, nw=50, x0=2.5):
    """
    RPA correlation and total energy (or sigma-functional)

    Args:
        Lpq : density fitting 3-center integral in MO basis
        pkl: name of pkl-file to store sigma-values and other relevant data
        nw : number of frequency point on imaginary axis
        x0: scaling factor for frequency grid

    Returns:
        e_tot : sigma_functional total energy
        e_tot_rpa : RPA total energy
        e_hf : EXX energy
        e_corr : sigma-functional correlation energy
        e_corr_rpa : RPA correlation energy
    """
    mf = rpa._scf
    # only support frozen core
    if rpa.frozen is not None:
        assert isinstance(rpa.frozen, int)
        assert rpa.frozen < rpa.nocc[0] and rpa.frozen < rpa.nocc[1]

    if Lpq is None:
        Lpq = rpa.ao2mo(mo_coeff)

    # Grids for integration on imaginary axis
    freqs, wts = _get_scaled_legendre_roots(nw, x0)

    # Compute energy (EXX)
    dm = mf.make_rdm1()
    uhf = scf.UHF(rpa.mol)
    e_hf = uhf.energy_elec(dm=dm)[0]
    e_hf += mf.energy_nuc()

    # Compute correlation energy
    e_corr_rpa, e_corr, allsigmas = get_ecorr(rpa, Lpq, freqs, wts, pkl)

    if pkl is not None:
        with open(pkl, "wb") as fileObj:
            pickle.dump(e_hf, fileObj)
            pickle.dump(wts, fileObj)
            pickle.dump(allsigmas, fileObj)

    # Compute totol energy
    e_tot = e_hf + e_corr
    e_tot_rpa = e_hf + e_corr_rpa

    return e_tot, e_tot_rpa, e_hf, e_corr, e_corr_rpa


def get_ecorr(rpa, Lpq, freqs, wts, pkl):
    """
    Compute correlation energy
    """
    mo_energy = _mo_energy_without_core(rpa, rpa._scf.mo_energy)
    nocca, noccb = rpa.nocc
    nw = len(freqs)

    x, c = get_spline_coeffs(rpa)

    if pkl is not None:
        allsigmas = []

    e_corr_rpa = 0.0
    e_corr_sigma = 0.0
    for w in range(nw):
        Pi = get_rho_response(freqs[w], mo_energy, Lpq[0, :, :nocca, nocca:], Lpq[1, :, :noccb, noccb:])
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
        return e_corr_rpa, e_corr_sigma, np.array(allsigmas)
    else:
        return e_corr_rpa, e_corr_sigma, None


class USIGMA(URPA):
    def __init__(self, mf, frozen=None, auxbasis=None, param=None):
        super().__init__(mf, frozen, auxbasis)

        self.param = param

        self.e_corr_rpa = None
        self.e_tot_rpa = None

    def kernel(self, mo_energy=None, mo_coeff=None, Lpq=None, pkl=None, nw=50, x0=2.5):
        """
        Args:
            mo_energy : 2D array (2, nmo), mean-field mo energy
            mo_coeff : 3D array (2, nmo, nmo), mean-field mo coefficient
            Lpq : 4D array (2, naux, nmo, nmo), 3-index ERI
            pkl: name of pkl-file to store sigma-values and other relevant data
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
            self, mo_energy, mo_coeff, Lpq=Lpq, pkl=pkl, nw=nw, x0=x0
        )

        return self.e_corr
