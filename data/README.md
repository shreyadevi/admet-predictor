# Datasets

Both files are public benchmark datasets, redistributed here in the form used
by the MoleculeNet benchmark (obtained from the DeepChem repository, MIT
licence). Column names were simplified; no rows were changed.

| File | Endpoint | Molecules | Labels |
|---|---|---|---|
| `esol.csv` | Aqueous solubility | 1,128 | `logS`: measured log10 solubility in mol/L |
| `tox21.csv.gz` | Tox21 toxicity panel | 8,014 | 12 binary assay outcomes (1 = active); blank = not tested |

## Sources

- **ESOL** - Delaney J.S. (2004) ESOL: Estimating aqueous solubility directly
  from molecular structure. *Journal of Chemical Information and Computer
  Sciences* 44, 1000-1005.
- **Tox21** - Tox21 Data Challenge 2014, U.S. National Institutes of Health,
  EPA and FDA collaboration. Assays: androgen receptor (NR-AR, NR-AR-LBD),
  aryl hydrocarbon receptor (NR-AhR), aromatase (NR-Aromatase), estrogen
  receptor (NR-ER, NR-ER-LBD), PPAR-gamma (NR-PPAR-gamma), and stress
  responses: antioxidant response (SR-ARE), genotoxicity (SR-ATAD5), heat
  shock (SR-HSE), mitochondrial membrane potential (SR-MMP), p53 (SR-p53).
- **Benchmark** - Wu Z. et al. (2018) MoleculeNet: a benchmark for molecular
  machine learning. *Chemical Science* 9, 513-530.
