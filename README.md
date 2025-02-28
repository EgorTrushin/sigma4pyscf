# sigma4pyscf

**Python library for σ-functional calculations based on [PySCF](https://pyscf.org/).** 

---

$\sigma$-functionals model the exchange-correlation kernel missing within the Randon Phase Approximation (RPA) by a function of the Hartree kernel and the Kohn-Sham response function that is optimized using thermochemical reference data, and provide significant improvements over the RPA at almost no cost.

As an example, below are the error distributions for the reaction energies of the [W4-11RE dataset](https://pubs.rsc.org/en/content/articlelanding/2017/cp/c7cp00757d) from PBE (PBE0), RPA@PBE (RPA@PBE0), and $\sigma$@PBE ($\sigma$@PBE0) calculations with the corresponding mean absolute errors.

<img src="/examples/W4-11RE/PBE_W4_11RE.png" alt="drawing" width="400"/><img src="examples/W4-11RE/PBE0_W4_11RE.png" alt="drawing" width="400"/>

---

Alternative implementations are available in
- Molpro Quantum Chemistry Software: see [corresponding section in the manual](https://www.molpro.net/manual/doku.php?id=kohn-sham_random-phase_approximation#rirpa_program).
- Amsterdam Density Functional (ADF): see [corresponding section in the manual](https://www.scm.com/doc/ADF/Input/Density_Functional.html#sigma-functional).

---

### Examples
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

mf = dft.RKS(mol, xc='pbe').density_fit().run()

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

mf = dft.UKS(mol, xc='pbe').density_fit().run()

sigma = USIGMA(mf)
sigma.kernel()
print(f'RPA:   E_corr={sigma.e_corr_rpa:.10f}  E_tot={sigma.e_tot_rpa:.10f}')
print(f'SIGMA: E_corr={sigma.e_corr:.10f}  E_tot={sigma.e_tot:.10f}')
```
See also examples in *example* directory:
- [W4-11RE.ipynb](/examples/W4-11RE/W4-11RE.ipynb): calculation using RPA and σ-functional for the W4-11RE dataset and comparisions of deviations
- [write_and_read_sigmas.ipynb](/examples/write_and_read_sigmas.ipynb): saving of a pickle file with Hartree-Fock energy, frequency integration weights, sigma values; then reading of this file and evaluation of correlation and total energy.

### Choice of parametrizations and basis sets
Currently, the σ-functional can be used in conjunction with the PBE, TPSS,  PBE0 and B3LYP exchange-correlation functionals. The available parametrizations for each functional are listed in the following table:
| functional | available parametrizations |
|------------|--------------|
| PBE        | P1<sup>1</sup> W1<sup>2</sup> S1<sup>4</sup> S2<sup>4</sup> A1<sup>5</sup> A2<sup>5</sup> S1re<sup>6</sup> |
| TPSS       | W1<sup>2</sup> |
| B3LYP      | W1<sup>2</sup> |
| PBE0       | W1<sup>1</sup> S1<sup>4</sup> S2<sup>4</sup> S1re<sup>6</sup> |

S1 parametrization is default for PBE and PBE0, whereas W1 parametrization is default for TPSS and B3LYP. The parametrization can be chosen during an initialization of SIGMA/USIGMA objects, e.g.:
```python
sigma = SIGMA(mf, param='P1')
```
Originally provided parametrizations were developed for the aug-cc-pwCVQZ orbital basis set, but have been tested to be applicable to the aug-cc-pwCVTZ orbital basis set as well. Special parameterizations for small basis sets, namely cc-pVTZ for atoms from H to Ne and aug-cc-pwCVTZ for heavier atoms, are called S1re. Unlike the original parameterizations, S1re's parameters have been optimized for 24 frequency points instead of 50.

### Frequency integration
The numerical frequency integration is performed with a Gauss-Legendre quadrature schemes. The weights $\tilde{\omega}_i$ and nodes $\tilde{x}_i$ of Gauss-Legendre quadrature for the interval [-1,1] are mapped into the interval [0,∞] as follows
$$x_i = w_0\frac{1 + \tilde{x}_i}{1 - \tilde{x}_i}$$
$$\omega_i = \tilde{\omega}_i \frac{2\omega_0}{(1 - \tilde{x}_i)^2}$$

Default number of frequency points is 50 and $\omega_0=2.5$. These values might be changed during the call of kernel function as follows:
```python
sigma.kernel(nw=100, x0=2.0)
```

### Frozen-core orbitals
The number of frozen core orbitals can be specified in the same way as in the PySCF RPA implementation. For example:
```python
sigma = SIGMA(mf, frozen=2)
```

### References
#### Main publications:
1. E. Trushin, A. Thierbach, A. Görling. Towards chemical accuracy at low computational cost: Density-functional theory with σ-functionals for the correlation energy – [J. Chem. Phys. 154 014104 (2021)](https://doi.org/10.1063/5.0026849)
2. S. Fauser, E. Trushin, C. Neiss, A. Görling. Chemical accuracy with σ-functionals for the Kohn-Sham correlation energy optimized for different input orbitals and eigenvalues – [J. Chem. Phys. 155 134111 (2021)](https://doi.org/10.1063/5.0059641)
3. J. Erhard, S. Fauser, E. Trushin, A. Görling. Scaled σ-functionals for the Kohn-Sham correlation energy with scaling functions from the homogeneous electron gas – [J. Chem. Phys. 157 114105 (2022)](https://doi.org/10.1063/5.0101641)
4. C. Neiss, S. Fauser, A. Görling. Geometries and vibrational frequencies with Kohn–Sham methods using σ-functionals for the correlation energy - [J. Chem. Phys. 158, 044107 (2023)](https://doi.org/10.1063/5.0129524)
5. Y. Lemke, C. Ochsenfeld. Highly accurate σ- and τ-functionals for beyond-RPA methods with approximate exchange kernels - [J. Chem. Phys. 159, 194104 (2023)](https://doi.org/10.1063/5.0173042)
6. S. Fauser, A. Förster, L. Redeker, C. Neiss, J. Erhard, E. Trushin, A. Görling. Basis set requirements of σ-functionals for Gaussian- and Slater-type basis functions and comparison with range-separated hybrid and double hybrid functionals - [J. Chem. Theory Comput. 2024, 20, 6, 2404–2422](https://doi.org/10.1021/acs.jctc.3c01132)

#### Additional publications:
- M. Glasbrenner, D. Graf, C. Ochsenfeld. Benchmarking the Accuracy of the Direct Random Phase Approximation and σ-Functionals for NMR Shieldings - [J. Chem. Theory Comput. 2022, 18, 1, 192–205](https://doi.org/10.1021/acs.jctc.1c00866)
- Y. Lemke, D. Graf, J. Kussmann, C. Ochsenfeld. An assessment of orbital energy corrections for the direct random phase approximation and explicit σ-functionals - [Mol. Phys. 121:9-10 (2023)](https://doi.org/10.1080/00268976.2022.2098862)
- D. Dhingra, A. Shori, A. Förster. Chemically accurate singlet-triplet gaps of organic chromophores and linear acenes by the random phase approximation and σ-functionals - [J. Chem. Phys. 159, 194105 (2023)](https://doi.org/10.1063/5.0177528)
- S. Fauser, V. Drontschenko, C. Ochsenfeld, A. Görling. Accurate NMR Shieldings with σ-Functionals - [J. Chem. Theory Comput. 2024, 20, 14, 6028–6036](https://doi.org/10.1021/acs.jctc.4c00512)
