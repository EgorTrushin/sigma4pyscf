from pyscf import gto, dft
from sigma.usigma import USIGMA


def calc_nh():
    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = [[7, (0.0, 0.0, 0.129649)], [1, (0.0, 0.0, -0.907543)]]
    mol.basis = {"N": "aug-cc-pwCVQZ", "H": "aug-cc-pVQZ"}
    mol.spin = 2
    mol.build()

    mf = dft.UKS(mol, xc="pbe").density_fit(auxbasis={"N": "aug-cc-pwCVQZ-RIFIT", "H": "aug-cc-pVQZ-RIFIT"}).run()

    sigma = USIGMA(mf)
    sigma.kernel()

    return sigma.e_corr_rpa, sigma.e_tot_rpa, sigma.e_corr, sigma.e_tot


def test_answer():
    e_corr_rpa, e_tot_rpa, e_corr, e_tot = calc_nh()
    assert abs(e_corr_rpa + 0.3903608275) < 1e-6
    assert abs(e_tot_rpa + 55.3670285558) < 1e-6
    assert abs(e_corr + 0.2840086425) < 1e-6
    assert abs(e_tot + 55.2606763708) < 1e-6
