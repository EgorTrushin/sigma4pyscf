# sigma-pyscf

**Python library for σ-functional calculations based on [PySCF](https://pyscf.org/).** 

## Examples
Spin-restricted σ-functional calculation for CO molecule:
```python
from pyscf import gto, dft
from sigma.sigma import SIGMA

mol = gto.Mole()
mol.verbose = 0
mol.atom = [
    [6 , (0. , 0. ,-0.646514)],
    [8 , (0. , 0. , 0.484886)]]
mol.basis = 'augccpwcvtz'
mol.build()

mf = dft.RKS(mol).density_fit()
mf.xc = 'pbe'
mf.kernel()

sigma = SIGMA(mf)
sigma.kernel()
print(f'RPA:   E_corr={sigma.e_corr_rpa:.10f}  E_tot={sigma.e_tot_rpa:.10f}')
print(f'SIGMA: E_corr={sigma.e_corr:.10f}  E_tot={sigma.e_tot:.10f}')
```
Spin-unrestricted σ-functional calculation for NH molecule:
```python
from pyscf import gto, dft
from sigma.usigma import USIGMA

mol = gto.Mole()
mol.verbose = 0
mol.atom = [
    [7 , (0. , 0. , 0.129649)],
    [1 , (0. , 0. ,-0.907543)]]
mol.basis = {'N': 'augccpwcvtz', 'H': 'augccpvtz'}
mol.spin = 2
mol.build()

mf = dft.UKS(mol).density_fit()
mf.xc = 'pbe'
mf.kernel()

sigma = USIGMA(mf)
sigma.kernel()
print(f'RPA:   E_corr={sigma.e_corr_rpa:.10f}  E_tot={sigma.e_tot_rpa:.10f}')
print(f'SIGMA: E_corr={sigma.e_corr:.10f}  E_tot={sigma.e_tot:.10f}')
```
See also examples in *example* directory:
- [W4-11RE.ipynb](https://github.com/EgorTrushin/sigma-pyscf/blob/main/examples/W4-11RE/W4-11RE.ipynb): calculation using RPA and σ-functional for the W4-11RE dataset and comparisions of deviations

## Choice of parametrizations and basis sets
Currently, the σ-functional can be used in conjunction with the PBE, TPSS,  PBE0 and B3LYP exchange-correlation functionals. The available parametrizations for each functional are listed in the following table:
| functional | available parametrizations |
|------------|--------------|
| PBE        | P1<sup>1</sup> W1<sup>2</sup> S1<sup>4</sup> S2<sup>4</sup>  |
| TPSS       | W1<sup>2</sup> |
| B3LYP      | W1<sup>2</sup> |
| PBE0       | W1<sup>1</sup> S1<sup>4</sup> S2<sup>4</sup> |

S2 parametrization is default for PBE and PBE0, whereas W1 parametrization is default for TPSS and B3LYP. The parametrization can be chosen during an initialization of SIGMA/USIGMA objects, e.g.:
```python
sigma = SIGMA(mf, param='P1')
```
Originally provided parametrizations were developed for the aug-cc-pwCVQZ orbital basis set, but have been tested to be applicable to the aug-cc-pwCVTZ orbital basis set as well.

*Special parametrizations for smaller orbital basis sets are coming soon.*

## References
#### Main publications
1. E. Trushin, A. Thierbach, A. Görling. Towards chemical accuracy at low computational cost: Density-functional theory with σ-functionals for the correlation energy – [J. Chem. Phys. 154 014104 (2021)](https://doi.org/10.1063/5.0026849)
2. S. Fauser, E. Trushin, C. Neiss, A. Görling. Chemical accuracy with σ-functionals for the Kohn-Sham correlation energy optimized for different input orbitals and eigenvalues – [J. Chem. Phys. 155 134111 (2021)](https://doi.org/10.1063/5.0059641)
3. J. Erhard, S. Fauser, E. Trushin, A. Görling. Scaled σ-functionals for the Kohn-Sham correlation energy with scaling functions from the homogeneous electron gas – [J. Chem. Phys. 157 114105 (2022)](https://doi.org/10.1063/5.0101641)
4. C. Neiss, S. Fauser, A. Görling. Geometries and vibrational frequencies with Kohn–Sham methods using σ-functionals for the correlation energy - [J. Chem. Phys. 158, 044107 (2023)](https://doi.org/10.1063/5.0129524)
#### Additional publications:
- M. Glasbrenner, D. Graf, C. Ochsenfeld. Benchmarking the Accuracy of the Direct Random Phase Approximation and σ-Functionals for NMR Shieldings - [J. Chem. Theory Comput. 2022, 18, 1, 192–205](https://doi.org/10.1021/acs.jctc.1c00866)
- Y. Lemke, D. Graf, J. Kussmann, C. Ochsenfeld. An assessment of orbital energy corrections for the direct random phase approximation and explicit σ-functionals - [Mol. Phys. 121:9-10 (2023)](https://doi.org/10.1080/00268976.2022.2098862)

## Current ToDo List
- [ ] Example Jupyter Notebook with calculation of GW100 dataset
- [ ] Tests
- [ ] flake8, project.yaml, etc
- [ ] Parametrizations for smaller basis sets
- [ ] Scaled σ-functionals
- [ ] Option to store σ-values in dat/pkl-file like in Molpro
