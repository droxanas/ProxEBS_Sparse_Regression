# Proximal Empirical Bayes for Sparse Regression with Posterior Decision Support

Code and supplementary material for the manuscript

> **Proximal Empirical Bayes for Sparse Regression with Posterior Decision Support**  
> Dimitrios Roxanas

The repository contains the final reproducibility scripts for the numerical studies reported in the manuscript and supplement. The code implements observation-scale calibration, SAPG empirical-Bayes calibration of a global Laplace shrinkage parameter, nonsmoothed MAP estimation by FISTA, posterior simulation with MYULA, posterior-informed sparse decisions, affine constraints, and the geometry-aware posterior calculation used in Study 2.

## Repository structure

```text
README.md
requirements.txt
LICENSE

data/
    diabetes-442-10.csv
    zhou_figure2_intervals.csv
    zhou_figure2_metadata.csv

src/
    prox.py
    model.py
    noise.py
    fista.py
    myula.py
    sapg.py
    selection.py
    diagnostics.py
    simulation.py

    study1_sparse_regression.py
    study2_soft_affine.py
    study3_diabetes.py
    supp_hard_constraint.py
    supp_scale_moreau.py

supplement.pdf
```

The scripts create a `figures/` directory when needed. Numerical summaries are printed to the terminal; no intermediate chains or result CSV files are written.

## Requirements

Python 3.10 or later is recommended.

Install the Python dependencies from the repository root with

```bash
python -m pip install -r requirements.txt
```

## Reproducing the numerical studies

Run all scripts from the repository root.

Main-paper studies:

```bash
python src/study1_sparse_regression.py
python src/study2_soft_affine.py
python src/study3_diabetes.py
```

Supplementary experiments:

```bash
python src/supp_hard_constraint.py
python src/supp_scale_moreau.py
```

All random-number seeds used by the reported experiments are fixed in the corresponding scripts.

Study 1 reproduces the two unconstrained sparse-regression regimes and the 4,000-versus-40,000-draw posterior refinement diagnostics. Study 2 reproduces the soft-affine experiment, the baseline and geometry-aware calculations at the strongest affine setting, and the reported matched-time timestep audit. Study 3 reproduces the diabetes analysis, including the extended SAPG calibration, nonsmoothed and smoothed MAP calculations, the 80,000-draw posterior analysis, the full decision grid, and the external interval comparison. The two supplementary scripts reproduce the hard homogeneous sum-zero calibration experiment and the observation-scale/Moreau-parameter sensitivity analysis.

## Diabetes data and external benchmark

`data/diabetes-442-10.csv` is the diabetes dataset used in the public ProxMCMC demonstration of Zhou, Heng, Chi and Zhou (2024), preserving the original filename and operational scaling used there. The Study 3 script checks the SHA-256 hash of this file before running.

`zhou_figure2_intervals.csv` and `zhou_figure2_metadata.csv` contain the benchmark interval endpoints and provenance information reconstructed from numerical output stored in the authors' public `Lasso diabetes.ipynb` notebook. The benchmark algorithms are not rerun here.

Source repository:

https://github.com/xinkai-zhou/ProxMCMCExamples

Reference:

X. Zhou, Q. Heng, E. C. Chi and H. Zhou (2024), *Proximal MCMC for Bayesian Inference of Constrained and Regularized Estimation*, **The American Statistician**, 78(4), 379--390.  
https://doi.org/10.1080/00031305.2024.2308821

The underlying diabetes data originate from:

B. Efron, T. Hastie, I. Johnstone and R. Tibshirani (2004), *Least Angle Regression*, **The Annals of Statistics**, 32(2), 407--499.  
https://doi.org/10.1214/009053604000000067

## Supplementary material

The current supplementary manuscript is included as `supplement.pdf`. It records the additional derivations, complete decision grids, numerical diagnostics, hard-constraint calibration experiment, sensitivity analysis, and diabetes benchmark details accompanying the submitted manuscript.

## Citation

If you use this code, please cite the accompanying manuscript. Full publication details will be added here when available.

## License

Except for the third-party data and benchmark material in `data/`, the
contents of this repository, including the source code and
`supplement.pdf`, are released under the GNU General Public License
v3.0; see `LICENSE`.

The files in `data/` retain the provenance and any applicable terms of
their original sources, as described above.
