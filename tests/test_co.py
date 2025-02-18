import basis_set_exchange
from pyscf import gto, dft
from sigma.sigma import SIGMA

AUXBASIS = {"C": gto.load(basis_set_exchange.api.get_basis('aug-cc-pwCVQZ-RIFIT', elements='C', fmt='nwchem'), 'C'),
            "O": gto.load(basis_set_exchange.api.get_basis('aug-cc-pwCVQZ-RIFIT', elements='O', fmt='nwchem'), 'O')}

def calc_co():
    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = [
        [6 , (0. , 0. ,-0.646514)],
        [8 , (0. , 0. , 0.484886)]]
    mol.basis = 'augccpwcvqz'
    mol.build()

    mf = dft.RKS(mol, xc='pbe').density_fit(auxbasis=AUXBASIS).run()

    sigma = SIGMA(mf)
    sigma.kernel()

    return sigma.e_corr_rpa, sigma.e_tot_rpa, sigma.e_corr, sigma.e_tot


def test_answer():
    e_corr_rpa, e_tot_rpa, e_corr, e_tot = calc_co()
    assert (abs(e_corr_rpa + 0.8026575065) < 1e-6)
    assert (abs(e_tot_rpa + 113.5699111213) < 1e-6)
    assert (abs(e_corr + 0.6033189902) < 1e-6)
    assert (abs(e_tot + 113.3705726050) < 1e-6)
