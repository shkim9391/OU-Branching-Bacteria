# OU_Branching_Bacteria

Code and reproducible analysis for hybrid Ornstein-Uhlenbeck (OU) / OU-Branching modeling of bacterial mutation-frequency dynamics across laboratory *Escherichia coli* strains. The repository includes processed mutation-frequency data, hierarchical PyMC model code, model-comparison scripts, visualization utilities, and manuscript figure-generation workflows.

## Overview

This repository accompanies a study of bacterial mutation-frequency dynamics using stochastic evolutionary models. The main goal is to compare constrained mean-reverting dynamics and branching extensions against simpler baselines, and to visualize how these models capture mutation-frequency patterns across strains.

The repository includes:

- processed mutation-frequency input data,
- reusable PyMC model code,
- scripts for fitting OU, OU-Branching, and random-walk baselines,
- model-comparison outputs based on leave-one-out predictive criteria,
- figure-generation scripts for main and supplementary manuscript figures,
- cleaned supplementary table files used in the study.

All data used in this study consist of bacterial mutation-frequency measurements derived from laboratory strains. Based on the current repository notes, the strains include *Escherichia coli* WT, PriA, and recG. All code and processed data are publicly available through this repository. No proprietary datasets or access restrictions apply.

## Repository contents

The repository currently contains the following files:

```text
README.md
S1_Table_clean.csv
compare_fig5_models.py
fig5_common.py
figure3_hybrid_ou_branching.py
fit_fig5_ou_branch_nb.py
fit_fig5_ou_nb.py
fit_fig5_rw_nb.py
hierarchical_ou_core_model_pymc.py
model_compare_loo.csv
mut_freq_data.csv
ou_multifigure_ABCB.py
plot_fig5_delta_elpd_ou-branching.py
publication_ready_ou_visualization_suite.py
s3_figure.py

## Scientific focus

This project studies mutation-frequency trajectories under alternative stochastic models of evolutionary change.

The main modeling ideas are:
	•	OU model: a mean-reverting stochastic process that captures constrained dynamics around a latent equilibrium or preferred state.
	•	OU-Branching model: an extension of OU dynamics that allows branch-specific divergence, potentially better capturing heterogeneous evolutionary structure.
	•	Random-walk baseline: a diffusion-like model without stabilizing pull, used as a simpler comparator.
	•	Hierarchical Bayesian inference: implemented in PyMC to estimate model parameters, quantify uncertainty, and compare predictive performance across models.

The repository is designed for manuscript reproducibility and figure generation rather than as a packaged software library.

Data files

mut_freq_data.csv

Primary processed mutation-frequency dataset used throughout the analysis.

Typical contents likely include:
	•	strain identifiers,
	•	mutation or feature identifiers,
	•	mutation-frequency measurements,
	•	timepoint or replicate information,
	•	model-ready values used in fitting and plotting scripts.

This is the main input file for the model-fitting workflows.

S1_Table_clean.csv

Cleaned supplementary table used in the manuscript.

This file is likely intended as a reader-friendly summary table for reporting processed values, derived summaries, or manuscript-facing metadata.

model_compare_loo.csv

Processed model-comparison table, likely containing leave-one-out predictive performance summaries.

Typical contents may include:
	•	model names,
	•	ELPD or related predictive criteria,
	•	standard errors,
	•	pairwise comparison quantities,
	•	ranking summaries used in Figure 5 or related analyses.

Core model code

hierarchical_ou_core_model_pymc.py

Core PyMC implementation of the hierarchical OU modeling framework.

This script likely contains:
	•	reusable model-building functions,
	•	prior specifications,
	•	likelihood setup,
	•	latent-process structure for OU dynamics,
	•	helper routines shared across multiple fitting scripts.

This file can be viewed as the modeling backbone of the repository.

Model-fitting scripts

fit_fig5_ou_branch_nb.py

Fits the OU-Branching model used in the Figure 5 comparison workflow.

Based on the filename, this script likely uses a negative-binomial or related count-aware observation model layered on top of the OU-Branching latent process. Typical outputs may include posterior summaries, saved fit objects, or comparison-ready metrics.

fit_fig5_ou_nb.py

Fits the non-branching OU model used as a baseline or comparator in Figure 5.

This script likely estimates mean-reverting dynamics without branch-specific divergence, enabling direct comparison with the branching extension.

fit_fig5_rw_nb.py

Fits the random-walk baseline model used in Figure 5.

This model serves as a simpler comparison point relative to OU-based dynamics and helps assess whether mean reversion and/or branching improve predictive fit.

Comparison and plotting utilities

compare_fig5_models.py

Compares the fitted models used in Figure 5.

Typical responsibilities may include:
	•	loading outputs from the model-fitting scripts,
	•	computing pairwise predictive comparisons,
	•	summarizing differences in expected log predictive density,
	•	exporting comparison tables for downstream plotting.

plot_fig5_delta_elpd_ou-branching.py

Generates Figure 5 or a major Figure 5 panel showing model-comparison results, especially delta-ELPD contrasts involving the OU-Branching model.

Typical outputs may include:
	•	bar plots or interval plots of predictive differences,
	•	manuscript-ready comparison panels,
	•	labeled summaries for model ranking.

fig5_common.py

Shared helper functions for the Figure 5 analysis and plotting workflow.

This file likely centralizes common tasks such as:
	•	file loading,
	•	summary formatting,
	•	color/style definitions,
	•	axis labeling,
	•	repeated transformations used across Figure 5 scripts.

Figure-generation scripts

figure3_hybrid_ou_branching.py

Generates the main Figure 3 for the manuscript.

Based on the filename, this script likely visualizes the hybrid OU-Branching framework or its application to the bacterial mutation-frequency data. Possible outputs include conceptual diagrams, trajectory summaries, or model-based visualizations.

ou_multifigure_ABCB.py

Generates a multi-panel OU-related figure, possibly involving multiple strain or condition comparisons.

The exact meaning of ABCB depends on the manuscript, but this script likely assembles several related panels into a composite output.

publication_ready_ou_visualization_suite.py

Produces finalized manuscript-quality visualizations.

This script appears intended to collect, standardize, or export polished figure panels for publication use. It may include formatting refinements, layout logic, or unified plotting functions used across figures.

s3_figure.py

Generates Supplementary Figure S3.

This script likely produces a manuscript supplement figure used to support the main analysis, such as diagnostics, additional comparisons, or robustness analyses.

Typical workflow

A typical analysis workflow from the repository root may look like this:

python fit_fig5_ou_branch_nb.py
python fit_fig5_ou_nb.py
python fit_fig5_rw_nb.py

python compare_fig5_models.py
python plot_fig5_delta_elpd_ou-branching.py

python figure3_hybrid_ou_branching.py
python ou_multifigure_ABCB.py
python publication_ready_ou_visualization_suite.py
python s3_figure.py

A practical interpretation of the workflow is:
	1.	fit the competing models,
	2.	compare their predictive performance,
	3.	generate the main comparison plots,
	4.	generate manuscript-quality main and supplementary figures.

Suggested analysis flow in words

The intended pipeline is approximately:
	1.	Load the processed mutation-frequency dataset from mut_freq_data.csv.
	2.	Fit the candidate stochastic models:
	•	OU-Branching
	•	OU baseline
	•	random-walk baseline
	3.	Summarize model performance using leave-one-out criteria.
	4.	Compare models using delta-ELPD or related predictive metrics.
	5.	Generate main and supplementary visualizations for the manuscript.
	6.	Export cleaned tables and final figure assets.

Typical Python loading examples

Load the processed mutation-frequency data

import pandas as pd

mut = pd.read_csv("mut_freq_data.csv")
print(mut.head())

Load the cleaned supplementary table

import pandas as pd

s1 = pd.read_csv("S1_Table_clean.csv")
print(s1.head())

Load the model-comparison summaries

import pandas as pd

loo = pd.read_csv("model_compare_loo.csv")
print(loo.head())

Software environment

This repository is intended to run in Python 3 with a standard scientific Python environment.

A typical environment will likely require:
	•	numpy
	•	pandas
	•	matplotlib
	•	scipy
	•	pymc
	•	arviz

Depending on the exact plotting or analysis code, additional packages may also be needed.

A simple installation example is:

pip install numpy pandas matplotlib scipy pymc arviz

A conda-based environment is also reasonable:

conda create -n ou-branching-bacteria python=3.11
conda activate ou-branching-bacteria
pip install numpy pandas matplotlib scipy pymc arviz

Reproducibility notes

This repository is intended to be fully reproducible from the processed data and scripts included here.

Important notes:
	•	all data used in the study are processed and included in the repository,
	•	no proprietary or restricted datasets are required,
	•	the workflow relies on Python-based scripts rather than hidden manual steps,
	•	figure scripts may assume that model-fitting outputs already exist,
	•	exact numerical results may vary slightly depending on software versions, random seeds, or MCMC configuration.

For stronger reproducibility, it is recommended to archive:
	•	the exact repository commit used for manuscript submission,
	•	the exact package versions used in the Python environment,
	•	any saved posterior summary files or fit objects generated during the workflow.

Recommended additions

For a cleaner public release, you may want to add:
	•	requirements.txt
	•	environment.yml
	•	.gitignore
	•	LICENSE
	•	a figures/ directory for exported outputs
	•	a short note describing which script generates which manuscript figure

Suggested requirements.txt

numpy
pandas
matplotlib
scipy
pymc
arviz

Example figure mapping

A simple figure map could be:
	•	figure3_hybrid_ou_branching.py → Main Figure 3
	•	plot_fig5_delta_elpd_ou-branching.py → Main Figure 5 comparison panel
	•	s3_figure.py → Supplementary Figure S3
	•	publication_ready_ou_visualization_suite.py → final manuscript-ready visual outputs

Data availability

All data used in this study consist of bacterial mutation-frequency measurements derived from laboratory strains (Escherichia coli WT, PriA, and recG). All code and processed data are publicly available through this repository. The full computational workflow, including PyMC scripts, preprocessing routines, and visualization scripts, is reproducible from the repository. No proprietary datasets or restrictions apply.

Contact

Author: Seung-Hwan Kim
