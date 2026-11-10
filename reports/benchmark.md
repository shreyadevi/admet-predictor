# Benchmark results

Test-set metrics, mean ± standard deviation over 3 seed(s). Random split: 80/10/10 by molecule. Scaffold split: 80/10/10 by Bemis-Murcko scaffold, so test molecules come from chemical series unseen in training.

## solubility (Absorption)

Aqueous solubility (ESOL, Delaney 2004). Poorly soluble drugs are poorly absorbed.

| Model | Split | RMSE (log10(mol/L)) ↓ | MAE ↓ | R² ↑ | Spearman ↑ |
|---|---|---|---|---|---|
| baseline | rdkit2d | random | 2.191 ± 0.150 | 1.762 ± 0.089 | -0.010 ± 0.014 | 0.000 |
| gradient_boosting | morgan+rdkit2d | random | 0.560 ± 0.066 | 0.422 ± 0.037 | 0.932 ± 0.023 | 0.961 ± 0.013 |
| gradient_boosting | rdkit2d | random | 0.566 ± 0.060 | 0.435 ± 0.033 | 0.931 ± 0.023 | 0.960 ± 0.011 |
| linear | rdkit2d | random | 0.734 ± 0.157 | 0.525 ± 0.042 | 0.878 ± 0.068 | 0.943 ± 0.028 |
| random_forest | rdkit2d | random | 0.613 ± 0.051 | 0.464 ± 0.037 | 0.919 ± 0.021 | 0.959 ± 0.012 |
| random_forest | morgan | random | 1.149 ± 0.106 | 0.904 ± 0.073 | 0.723 ± 0.020 | 0.846 ± 0.015 |
| baseline | rdkit2d | scaffold | 2.263 | 1.646 | -0.275 | 0.000 |
| gradient_boosting | morgan+rdkit2d | scaffold | 0.970 | 0.711 | 0.766 | 0.867 |
| gradient_boosting | rdkit2d | scaffold | 0.988 | 0.740 | 0.757 | 0.863 |
| linear | rdkit2d | scaffold | 1.012 | 0.751 | 0.745 | 0.868 |
| random_forest | rdkit2d | scaffold | 0.985 ± 0.004 | 0.738 ± 0.003 | 0.758 ± 0.002 | 0.849 ± 0.001 |
| random_forest | morgan | scaffold | 1.754 ± 0.006 | 1.348 ± 0.008 | 0.234 ± 0.005 | 0.579 ± 0.004 |

## tox21 (Toxicity)

Tox21 panel: 12 in-vitro assays for nuclear-receptor and stress-response activity.

| Model | Split | ROC-AUC ↑ | PR-AUC ↑ | Balanced accuracy ↑ |
|---|---|---|---|---|
| baseline | rdkit2d | random | 0.500 | 0.079 ± 0.004 | 0.500 |
| gradient_boosting | morgan+rdkit2d | random | 0.857 ± 0.011 | 0.529 ± 0.031 | 0.711 ± 0.016 |
| gradient_boosting | rdkit2d | random | 0.853 ± 0.008 | 0.518 ± 0.030 | 0.707 ± 0.020 |
| linear | rdkit2d | random | 0.815 ± 0.027 | 0.344 ± 0.020 | 0.761 ± 0.021 |
| random_forest | rdkit2d | random | 0.857 ± 0.008 | 0.514 ± 0.037 | 0.692 ± 0.020 |
| random_forest | morgan | random | 0.831 ± 0.023 | 0.457 ± 0.044 | 0.711 ± 0.021 |
| baseline | rdkit2d | scaffold | 0.500 | 0.105 | 0.500 |
| gradient_boosting | morgan+rdkit2d | scaffold | 0.763 | 0.405 | 0.637 |
| gradient_boosting | rdkit2d | scaffold | 0.760 | 0.406 | 0.635 |
| linear | rdkit2d | scaffold | 0.720 | 0.270 | 0.674 |
| random_forest | rdkit2d | scaffold | 0.776 ± 0.005 | 0.401 ± 0.004 | 0.608 ± 0.003 |
| random_forest | morgan | scaffold | 0.734 ± 0.003 | 0.346 ± 0.002 | 0.630 ± 0.002 |
