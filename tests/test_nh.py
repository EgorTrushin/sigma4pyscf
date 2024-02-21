from pyscf import gto, dft
from sigma.usigma import USIGMA
from auxbasis_nh import AUXBASIS

def calc_co():
    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = [
        [7 , (0. , 0. , 0.129649)],
        [1 , (0. , 0. ,-0.907543)]]
    mol.basis = {'N': 'augccpwcvqz', 'H': 'augccpvqz'}
    mol.spin = 2
    mol.build()

    mf = dft.UKS(mol, xc='pbe').density_fit(auxbasis=AUXBASIS).run()

    sigma = USIGMA(mf)
    sigma.kernel()

    return sigma.e_corr_rpa, sigma.e_tot_rpa, sigma.e_corr, sigma.e_tot


def test_answer():
    e_corr_rpa, e_tot_rpa, e_corr, e_tot = calc_co()
    assert (abs(e_corr_rpa + 0.3903602822) < 1e-6)
    assert (abs(e_tot_rpa + 55.3670281990) < 1e-6)
    assert (abs(e_corr + 0.2954264966) < 1e-6)
    assert (abs(e_tot + 55.2720944134) < 1e-6)
