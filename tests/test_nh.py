import basis_set_exchange
from pyscf import gto, dft
from sigma.usigma import USIGMA

AUXBASIS = {"N": gto.load(basis_set_exchange.api.get_basis('aug-cc-pwCVQZ-RIFIT', elements='N', fmt='nwchem'), 'N'),
            "H": gto.load(basis_set_exchange.api.get_basis('aug-cc-pVQZ-RIFIT', elements='H', fmt='nwchem'), 'H')}

def calc_nh():
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
    e_corr_rpa, e_tot_rpa, e_corr, e_tot = calc_nh()
    assert (abs(e_corr_rpa + 0.3903608275) < 1e-6)
    assert (abs(e_tot_rpa + 55.3670285558) < 1e-6)
    assert (abs(e_corr + 0.2840086425) < 1e-6)
    assert (abs(e_tot + 55.2606763708) < 1e-6)
