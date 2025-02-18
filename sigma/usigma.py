#!/usr/bin/env python
import numpy as np
from pyscf import lib
from pyscf.gw.urpa import URPA
from .sigma import SIGMA


def make_dielectric_matrix(omega, e_ov, f_ov, eris, blksize=None):
    """
    Compute dielectric matrix at a given frequency omega

    Args:
        omega : float, frequency
        e_ov : 1D array (nocc * nvir), orbital energy differences
        eris : DF ERI object

    Returns:
        diel : 2D array (naux, naux), dielectric matrix
    """
    assert blksize is not None

    nocc, nvir, naux = eris.nocc, eris.nvir, eris.naux

    isreal = eris.dtype == np.float64

    diel = np.zeros((naux, naux), dtype=eris.dtype)

    for s in [0, 1]:
        chi0 = (2.0 * e_ov[s] * f_ov[s] / (omega ** 2 + e_ov[s] ** 2)).ravel()
        for p0, p1 in lib.prange(0, nocc[s] * nvir[s], blksize):
            ovL = eris.get_ov_blk(s, p0, p1)
            ovL_chi = (ovL.T * chi0[p0:p1]).T
            if isreal:
                lib.ddot(ovL_chi.T, ovL, c=diel, beta=1)
            else:
                lib.dot(ovL_chi.T, ovL.conj(), c=diel, beta=1)
            ovL = ovL_chi = None

    return diel


class USIGMA(URPA):
    get_e_hf = SIGMA.get_e_hf
    kernel = SIGMA.kernel
    _finalize = SIGMA._finalize

    def __init__(self, mf, frozen=None, auxbasis=None, param=None):
        super().__init__(mf, frozen, auxbasis)

        self.param = param

        self.e_corr_rpa = None
        self.e_tot_rpa = None

    def make_e_ov(self):
        """
        Compute orbital energy differences
        """
        split_mo_energy = self.split_mo_energy()
        e_ov = [(split_mo_energy[s][1][:, None] - split_mo_energy[s][2]).ravel() for s in [0, 1]]

        if self.nocc[1] > 0:
            gap = [-e_ov[s].max() for s in [0, 1]]
        else:
            gap = (-e_ov[0].max(),)

        if np.min(gap) < 1e-3:
            print("RPA code is not well-defined for degenerate systems!")
            print("Lowest orbital energy difference: % 6.4e", np.min(gap))

        return e_ov

    def make_f_ov(self):
        """
        Compute orbital occupation number differences
        """
        split_mo_occ = self.split_mo_occ()
        return [(split_mo_occ[s][1][:, None] - split_mo_occ[s][2]).ravel() for s in [0, 1]]

    def make_dielectric_matrix(self, omega, e_ov=None, f_ov=None, eris=None, max_memory=None, blksize=None):
        """
        Args:
            omega : float, frequency
            e_ov : 1D array (nocc * nvir), orbital energy differences
            mo_coeff :  (nao, nmo), mean-field mo coefficient
            cderi_ov :  (naux, nocc, nvir), Cholesky decomposed ERI in OV subspace.

        Returns:
            diel : 2D array (naux, naux), dielectric matrix
        """
        if e_ov is None:
            e_ov = self.make_e_ov()
        if f_ov is None:
            f_ov = self.make_f_ov()
        if eris is None:
            eris = self.ao2mo()
        if max_memory is None:
            max_memory = self.max_memory

        if blksize is None:
            mem_avail = max_memory - lib.current_memory()[0]
            nocc, nvir, naux = eris.nocc, eris.nvir, eris.naux
            dsize = eris.dsize
            mem_blk = 2 * naux * dsize / 1e6  # ovL and ovL*chi0
            blksize = max(1, min(max(nocc) * max(nvir), int(np.floor(mem_avail * 0.7 / mem_blk))))
        else:
            blksize = min(blksize, e_ov.size)

        diel = make_dielectric_matrix(omega, e_ov, f_ov, eris, blksize=blksize)

        return diel
