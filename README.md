# OU_Branching_Bacteria

Code and reproducible analysis for hybrid Ornstein-Uhlenbeck (OU) / OU-Branching modeling of bacterial mutation-frequency dynamics. This repository contains the core hierarchical PyMC model, downstream visualization scripts, supplementary figure generation, and a separate Figure 5 model-comparison workflow.

All data used in this study consist of bacterial mutation-frequency measurements derived from laboratory strains. All code and processed data are publicly available through this repository. No proprietary datasets or access restrictions apply.

---

## Overview

This repository implements two connected but distinct workflows:

1. **Core OU / OU-Branching inference and visualization workflow**  
   Built from `mut_freq_data.csv` using a hierarchical PyMC model. This branch produces the main posterior trace, time grid, summaries, diagnostics, posterior predictive checks, and downstream figure panels.

2. **Figure 5 model-comparison workflow**  
   Built from `series_auto.csv` and `patient_group.csv`, this branch fits three competing models:
   - random walk
   - OU
   - OU-Branching  
   and then compares them using LOO / WAIC-style predictive summaries.

The repository is organized for manuscript reproducibility rather than as a general-purpose software package.

## Repository structure

```text
repo_root/
  data/
    S1_Table_clean.csv
    model_compare_loo.csv
    mut_freq_data.csv

  scripts/
    compare_fig5_models.py
    fig5_common.py
    figure3_hybrid_ou_branching.py
    fit_fig5_ou_branch_nb.py
    fit_fig5_ou_nb.py
    fit_fig5_rw_nb.py
    hierarchical_ou_core_model_pymc.py
    ou_multifigure_ABCB.py
    plot_fig5_delta_elpd_ou-branching.py
    publication_ready_ou_visualization_suite.py
    s3_figure.py

  README.md

### Core scripts

- `hierarchical_ou_core_model_pymc.py`  
  Core hierarchical PyMC model. This is the main upstream script for the OU-based bacterial mutation-frequency analysis.

- `ou_multifigure_ABCB.py`  
  Uses the core posterior outputs to generate the multi-panel Figure 2 composite.

- `figure3_hybrid_ou_branching.py`  
  Generates the main Figure 3.

- `publication_ready_ou_visualization_suite.py`  
  Produces publication-ready parameter visualizations and posterior summaries, including multiple Figure 2 panels and related trajectory/PPC outputs.

- `s4_figure.py`  
  Generates a supplementary OU-Branching figure from the core posterior outputs.

- `Supplementary_Information.tex`  
  Compiles the supplementary PDF using the supplementary figure assets and the cleaned S1 table.

### Figure 5 workflow

- `fig5_common.py`  
  Shared helper utilities for the Figure 5 model-fitting and comparison scripts.

- `fit_fig5_rw_nb.py`  
  Fits the random-walk baseline model.

- `fit_fig5_ou_nb.py`  
  Fits the OU baseline model.

- `fit_fig5_ou_branch_nb.py`  
  Fits the OU-Branching model.

- `compare_fig5_models.py`  
  Compares fitted Figure 5 models and exports the model-comparison table.

- `plot_fig5_delta_elpd_ou-branching.py`  
  Generates the Figure 5 delta-ELPD plot from the comparison table.

### Data and summary files

- `mut_freq_data.csv`  
  Input dataset for the core hierarchical OU model.

- `S1_Table_clean.csv`  
  Cleaned supplementary table; used in the supplementary information package.

- `model_compare_loo.csv`  
  Model-comparison output table for the Figure 5 workflow.

Additional workflow inputs referenced in the analysis summary:

- `series_auto.csv`
- `patient_group.csv`

## Workflow summary

## 1. Core model workflow

### Step 1. Fit the core hierarchical model

```bash
python hierarchical_ou_core_model_pymc.py

Input
	•	mut_freq_data.csv

Primary outputs
	•	trace_core.nc
	•	times.npy
	•	az.summary
	•	MCMC diagnostics
	•	posterior predictive checks

Role
	•	core model fit
	•	upstream source for Figure 2, Figure 3, supplementary figures, and posterior diagnostics

Step 2. Generate the Figure 2 multi-panel composite

python ou_multifigure_ABCB.py

Inputs
	•	trace_core.nc
	•	times.npy

Output
	•	ou_multifigure_ABCD.png

Role
	•	Figure 2 composite

Note: the script name uses ABCB, while the output in your workflow sheet appears as ou_multifigure_ABCD.png. If that output filename is correct, I would leave a brief note in the repo so readers do not think it is a typo.

Step 3. Generate Figure 3

python figure3_hybrid_ou_branching.py

Inputs
	•	trace_core.nc
	•	times.npy

Output
	•	Figure3_Hybrid_OU_Branching.png

Role
	•	Figure 3

Step 4. Generate publication-ready parameter visualizations

python publication_ready_ou_visualization_suite.py

Inputs
	•	trace_core.nc
	•	times.npy

Outputs / panels
	•	ridgeplot for μ (log10 mutation frequency) → Figure 2A
	•	ridgeplot for diffusion scale σ → Figure 2B
	•	ridgeplot for mean-reversion rate θ
	•	shrinkage plot for μ_bg
	•	OU trajectories
	•	posterior predictive checks
	•	diffusion comparison

Role
	•	figure-panel production for the main manuscript, especially Figure 2 and related posterior summaries

Because this script generates several different outputs, it is a good idea to save them into a dedicated figure/output directory if you have not already done so.

Step 5. Generate the supplementary OU-Branching figure

python s4_figure.py

Inputs
	•	trace_core.nc
	•	times.npy

Outputs
	•	S4_OU_branching.png
	•	S4_OU_branching.pdf

Role
	•	supplementary figure generation

In your workflow sheet, this appears to have been renumbered at some stage. If the final manuscript now calls this S3 rather than S4, update the README to reflect the final journal numbering only.

Step 6. Compile supplementary information

pdflatex Supplementary_Information.tex
pdflatex Supplementary_Information.tex

Inputs
	•	S2_Figure.png
	•	S3_Figure.png
	•	S4_Figure.png
	•	S1_Table_clean.csv

Output
	•	Supplementary_Information.pdf

Role
	•	full supplementary information package, including the cleaned complete dataset table

2. Figure 5 model-comparison workflow

Shared helper script
	•	fig5_common.py

This script supports the Figure 5 fitting scripts and likely contains shared utilities for:
	•	data loading
	•	group ordering
	•	plotting helpers
	•	common summary logic

Step 7. Fit the random-walk baseline

python fit_fig5_rw_nb.py

Inputs
	•	series_auto.csv
	•	patient_group.csv

Outputs
	•	posterior.nc
	•	posterior_summary.csv
	•	group_order_used.csv
	•	loo_waic.csv

Step 8. Fit the OU baseline

python fit_fig5_ou_nb.py

Inputs
	•	series_auto.csv
	•	patient_group.csv

Outputs
	•	posterior.nc
	•	posterior_summary.csv
	•	group_order_used.csv
	•	loo_waic.csv

Step 9. Fit the OU-Branching model

python fit_fig5_ou_branch_nb.py

Inputs
	•	series_auto.csv
	•	patient_group.csv

Outputs
	•	posterior.nc
	•	posterior_summary.csv
	•	group_order_used.csv
	•	loo_waic.csv

Step 10. Compare the three Figure 5 models

python compare_fig5_models.py

Inputs
	•	out_rw/posterior.nc
	•	out_ou/posterior.nc
	•	out_ou_branch/posterior.nc

Output
	•	model_compare_loo.csv

Role
	•	integrates fitted model outputs into a single predictive-comparison table

This implies that the three fitting scripts should write results into separate output folders such as:

out_rw/
out_ou/
out_ou_branch/

Step 11. Plot the Figure 5 delta-ELPD summary

python plot_fig5_delta_elpd_ou-branching.py

Input
	•	model_compare_loo.csv

Outputs
	•	Figure5A_deltaELPD.pdf
	•	Figure5A_deltaELPD.png

Role
	•	Figure 5 model-comparison panel

Your workflow sheet also includes plot_fig5A_delta_elpd.py. If that script is no longer in the repository, I would omit it from the public README and keep only the final script name actually present in GitHub.

End-to-end run order

A typical full run from the repository root is:

python hierarchical_ou_core_model_pymc.py
python ou_multifigure_ABCB.py
python figure3_hybrid_ou_branching.py
python publication_ready_ou_visualization_suite.py
python s4_figure.py

pdflatex Supplementary_Information.tex
pdflatex Supplementary_Information.tex

python fit_fig5_rw_nb.py
python fit_fig5_ou_nb.py
python fit_fig5_ou_branch_nb.py
python compare_fig5_models.py
python plot_fig5_delta_elpd_ou-branching.py

Input/output map

Core analysis branch
	•	mut_freq_data.csv
→ hierarchical_ou_core_model_pymc.py
→ trace_core.nc, times.npy, summaries, diagnostics, PPC
	•	trace_core.nc + times.npy
→ ou_multifigure_ABCB.py
→ Figure 2 composite
	•	trace_core.nc + times.npy
→ figure3_hybrid_ou_branching.py
→ Figure 3
	•	trace_core.nc + times.npy
→ publication_ready_ou_visualization_suite.py
→ Figure 2 panels, trajectories, PPC, diffusion comparison
	•	trace_core.nc + times.npy
→ s4_figure.py
→ supplementary OU-Branching figure

Figure 5 comparison branch
	•	series_auto.csv + patient_group.csv
→ fit_fig5_rw_nb.py
→ random-walk posterior outputs
	•	series_auto.csv + patient_group.csv
→ fit_fig5_ou_nb.py
→ OU posterior outputs
	•	series_auto.csv + patient_group.csv
→ fit_fig5_ou_branch_nb.py
→ OU-Branching posterior outputs
	•	out_rw/posterior.nc + out_ou/posterior.nc + out_ou_branch/posterior.nc
→ compare_fig5_models.py
→ model_compare_loo.csv
	•	model_compare_loo.csv
→ plot_fig5_delta_elpd_ou-branching.py
→ Figure5A_deltaELPD.pdf, Figure5A_deltaELPD.png

Software environment

This repository is intended to run in Python 3 with a standard scientific Python stack. Typical dependencies include:
	•	numpy
	•	pandas
	•	matplotlib
	•	scipy
	•	pymc
	•	arviz
	•	xarray
	•	netCDF4
	•	openpyxl

A minimal install might look like:

pip install numpy pandas matplotlib scipy pymc arviz xarray netCDF4 openpyxl

If LaTeX compilation is used for the supplement, a working TeX installation is also required.

Reproducibility notes
	•	The repository is manuscript-specific.
	•	The core PyMC model is the upstream dependency for the main posterior-based figures.
	•	The Figure 5 workflow is a separate comparison pipeline.
	•	Some figure numbering in the workflow sheet appears to reflect renumbering during manuscript revision; the public README should use only the final journal numbering.
	•	Exact posterior summaries may vary slightly depending on package versions, random seeds, and MCMC settings.

For a clean archival release, it is helpful to include:
	•	requirements.txt or environment.yml
	•	exact package versions
	•	final figure numbering after journal submission
	•	a release tag corresponding to the submission version

Data availability

All data used in this study consist of bacterial mutation-frequency measurements derived from laboratory strains. All processed data and reproducible code required for the analyses are included in this repository. No proprietary datasets or access restrictions apply.

Author

Seung-Hwan Kim

DOI
10.5281/zenodo.19619166

[![DOI](https://zenodo.org/badge/1118885358.svg)](https://doi.org/10.5281/zenodo.19619165)
