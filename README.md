OU-Branching-Bacteria

This repository contains code, processed data, model outputs, and figure-generation scripts for the manuscript:

Hierarchical Bayesian OU–Branching Models of Bacterial Mutation Evolution

The repository reproduces the main figures and supplementary figures for the Bioinformatics Advances submission.

Repository overview

OU-Branching-Bacteria/
├── data/                 # Processed input data and metadata
├── scripts/              # Model-fitting and figure-generation scripts
├── results/              # Model outputs and figure-supporting CSV files
├── figures/              # Main and supplementary figure files
└── manuscript/           # Figure legends and supplementary information files

Input data

The primary processed input files are located in data/processed/.

data/processed/
├── mut_freq_data.csv
├── series_auto.csv
└── patient_group.csv

mut_freq_data.csv is used for the core OU–Branching model and Figures 2–4.

series_auto.csv and patient_group.csv are used for the Figure 5 model-comparison workflow.

Main workflow

The full workflow consists of four stages:

1. Generate Figure 1 conceptual workflow.
2. Fit the core hierarchical OU model.
3. Generate Figures 2–4 and Supplementary Figures S2–S4 from the core posterior.
4. Fit Figure 5 baseline models, compare them by PSIS-LOO, and generate Figure 5.

All scripts should be run from the project root.

Figure 1: OU–Branching workflow schematic

Script:

python scripts/figure1_ou_branching_workflow_adjusted.py

Inputs:

None.

Outputs:

figures/main/Figure1_OU_Branching_workflow_adjusted.pdf
figures/main/Figure1_OU_Branching_workflow_adjusted.png
figures/main/Figure1_OU_Branching_workflow_adjusted.svg

Core hierarchical OU model

Script:

python scripts/hierarchical_ou_core_model_pymc_marginal.py

Input:

data/processed/mut_freq_data.csv

Outputs:

results/core_model/trace_core.nc
data/metadata/times.npy
data/metadata/bg_categories.npy

This script fits the core hierarchical Bayesian OU model used by Figures 2–4 and Supplementary Figures S2–S4.

Figure 2: Bacterial validation platform and OU parameter inference

Script:

python scripts/generate_figure2_from_trace_core.py

Inputs:

data/processed/mut_freq_data.csv
results/core_model/trace_core.nc
data/metadata/times.npy
data/metadata/bg_categories.npy

Outputs:

figures/main/Figure2_bacterial_validation_OU_parameters_real.png
figures/main/Figure2_bacterial_validation_OU_parameters_real.pdf
figures/main/Figure2_bacterial_validation_OU_parameters_real.svg
results/figure2/posterior_mu_sigma_long.csv
results/figure2/figure2_parameter_summary.csv

Figure 3: OU–Branching count layer

Script:

python scripts/generate_figure3_ou_branching_count_layer.py

Inputs:

data/processed/mut_freq_data.csv
results/core_model/trace_core.nc
data/metadata/times.npy
data/metadata/bg_categories.npy

Outputs:

figures/main/Figure3_OU_branching_count_layer_real.png
figures/main/Figure3_OU_branching_count_layer_real.pdf
figures/main/Figure3_OU_branching_count_layer_real.svg
results/figure3/figure3_simulated_latent_probability_counts.csv

Figure 4: Posterior predictive validation

Script:

python scripts/generate_figure4_posterior_predictive_validation.py

Inputs:

data/processed/mut_freq_data.csv
results/core_model/trace_core.nc
data/metadata/bg_categories.npy

Outputs:

figures/main/Figure4_posterior_predictive_validation_real.png
figures/main/Figure4_posterior_predictive_validation_real.pdf
figures/main/Figure4_posterior_predictive_validation_real.svg
results/figure4/figure4_ppc_summary.csv

Figure 5: Benchmarking and model ablation

Figure 5 requires fitting three model variants before generating the figure:

1. Random-walk negative-binomial model.
2. OU negative-binomial model.
3. OU–Branching negative-binomial model.

Step 1: Fit random-walk baseline

Script:

python scripts/fit_fig5_rw_nb.py

Inputs:

data/processed/series_auto.csv
data/processed/patient_group.csv

Outputs:

results/figure5/out_rw/posterior.nc
results/figure5/out_rw/posterior_summary.csv
results/figure5/out_rw/group_order_used.csv
results/figure5/out_rw/loo_waic.csv

Step 2: Fit OU-only baseline

Script:

python scripts/fit_fig5_ou_nb.py

Inputs:

data/processed/series_auto.csv
data/processed/patient_group.csv

Outputs:

results/figure5/out_ou/posterior.nc
results/figure5/out_ou/posterior_summary.csv
results/figure5/out_ou/group_order_used.csv
results/figure5/out_ou/loo_waic.csv

Step 3: Fit OU–Branching model

Script:

python scripts/fit_fig5_ou_branch_nb.py

Inputs:

data/processed/series_auto.csv
data/processed/patient_group.csv

Outputs:

results/figure5/out_ou_branch/posterior.nc
results/figure5/out_ou_branch/posterior_summary.csv
results/figure5/out_ou_branch/group_order_used.csv
results/figure5/out_ou_branch/loo_waic.csv

Step 4: Compare Figure 5 models

Script:

python scripts/compare_fig5_models.py

Inputs:

results/figure5/out_rw/posterior.nc
results/figure5/out_ou/posterior.nc
results/figure5/out_ou_branch/posterior.nc

Output:

results/figure5/model_comparison_loo.csv

Note: The filename model_comparison_loo.csv is used consistently because it is the expected input for the Figure 5 plotting script.

Step 5: Generate Figure 5

Script:

python scripts/generate_figure5_benchmarking_ablation.py

Input:

results/figure5/model_comparison_loo.csv

Outputs:

figures/main/Figure5_benchmarking_ablation_real.png
figures/main/Figure5_benchmarking_ablation_real.pdf
figures/main/Figure5_benchmarking_ablation_real.svg
results/figure5/figure5_model_comparison_table_used.csv
results/figure5/figure5_model_ablation_matrix.csv

Supplementary Figure S1: Dataset overview

Script:

python scripts/generate_supp_figure_s1_dataset_overview_v3.py

Inputs:

The script uses the processed mutation-frequency dataset and standardized input structure.

Recommended input:

data/processed/mut_freq_data.csv

Outputs:

figures/supplementary/supp_figure_s1_dataset_overview.png
figures/supplementary/supp_figure_s1_dataset_overview.pdf
results/supplementary/supp_figure_s1_standardized_input_snapshot.csv

Supplementary Figure S2: MCMC diagnostics

Script:

python scripts/generate_supp_figure_s2_mcmc_diagnostics_v4.py

Input:

results/core_model/trace_core.nc

Outputs:

figures/supplementary/supp_figure_s2_mcmc_diagnostics.png
figures/supplementary/supp_figure_s2_mcmc_diagnostics.pdf
results/supplementary/supp_figure_s2_arviz_summary.csv

Supplementary Figure S3: Posterior predictive count PMFs

Script:

python scripts/generate_supp_figure_s3_ppc_count_pmfs.py

Inputs:

results/core_model/trace_core.nc
data/processed/mut_freq_data.csv

Outputs:

figures/supplementary/supp_figure_s3_ppc_count_pmfs.png
figures/supplementary/supp_figure_s3_ppc_count_pmfs.pdf
results/supplementary/supp_figure_s3_ppc_count_pmf_summary.csv
results/supplementary/supp_figure_s3_ppc_count_pmf_values.csv

Supplementary Figure S4: Sensitivity analysis

Script:

python scripts/generate_supp_figure_s4_sensitivity_analysis_v2.py

Inputs:

results/core_model/trace_core.nc
data/processed/mut_freq_data.csv

Outputs:

figures/supplementary/supp_figure_s4_sensitivity_analysis.png
figures/supplementary/supp_figure_s4_sensitivity_analysis.pdf
results/supplementary/supp_figure_s4_parameter_sensitivity.csv
results/supplementary/supp_figure_s4_coverage_sensitivity.csv
results/supplementary/supp_figure_s4_count_layer_sensitivity.csv

Recommended execution order

To reproduce the full workflow from the project root:

python scripts/figure1_ou_branching_workflow_adjusted.py
python scripts/hierarchical_ou_core_model_pymc_marginal.py
python scripts/generate_figure2_from_trace_core.py
python scripts/generate_figure3_ou_branching_count_layer.py
python scripts/generate_figure4_posterior_predictive_validation.py
python scripts/fit_fig5_rw_nb.py
python scripts/fit_fig5_ou_nb.py
python scripts/fit_fig5_ou_branch_nb.py
python scripts/compare_fig5_models.py
python scripts/generate_figure5_benchmarking_ablation.py
python scripts/generate_supp_figure_s1_dataset_overview_v3.py
python scripts/generate_supp_figure_s2_mcmc_diagnostics_v4.py
python scripts/generate_supp_figure_s3_ppc_count_pmfs.py
python scripts/generate_supp_figure_s4_sensitivity_analysis_v2.py

Reproducibility notes

* Scripts should be run from the project root.
* Local hard-coded paths should be avoided.
* All outputs should be written to figures/ or results/.
* The posterior file trace_core.nc is required for Figures 2–4 and Supplementary Figures S2–S4.
* The Figure 5 model-comparison workflow requires separate output directories for the random-walk, OU-only, and OU–Branching models.
* Large posterior files may be excluded from GitHub and provided through an archived release if necessary.

Main figure workflow table

Figure	Script	Main inputs	Main outputs
Figure 1	figure1_ou_branching_workflow_adjusted.py	None	Figure 1 PDF/PNG/SVG
Core model	hierarchical_ou_core_model_pymc_marginal.py	mut_freq_data.csv	trace_core.nc, times.npy, bg_categories.npy
Figure 2	generate_figure2_from_trace_core.py	mut_freq_data.csv, trace_core.nc, times.npy, bg_categories.npy	Figure 2 PDF/PNG/SVG, parameter summaries
Figure 3	generate_figure3_ou_branching_count_layer.py	mut_freq_data.csv, trace_core.nc, times.npy, bg_categories.npy	Figure 3 PDF/PNG/SVG, simulated latent/probability/count CSV
Figure 4	generate_figure4_posterior_predictive_validation.py	mut_freq_data.csv, trace_core.nc, bg_categories.npy	Figure 4 PDF/PNG/SVG, PPC summary
Figure 5	generate_figure5_benchmarking_ablation.py	model_comparison_loo.csv	Figure 5 PDF/PNG/SVG, comparison and ablation CSVs

Supplementary figure workflow table

Supplementary figure	Script	Main inputs	Main outputs
Supplementary Figure S1	generate_supp_figure_s1_dataset_overview_v3.py	mut_freq_data.csv	S1 PDF/PNG, standardized input snapshot
Supplementary Figure S2	generate_supp_figure_s2_mcmc_diagnostics_v4.py	trace_core.nc	S2 PDF/PNG, ArviZ summary
Supplementary Figure S3	generate_supp_figure_s3_ppc_count_pmfs.py	trace_core.nc, mut_freq_data.csv	S3 PDF/PNG, PMF summary and values
Supplementary Figure S4	generate_supp_figure_s4_sensitivity_analysis_v2.py	trace_core.nc, mut_freq_data.csv	S4 PDF/PNG, sensitivity CSVs

Citation

A DOI-backed archived release will be provided through Zenodo after manuscript submission or acceptance.
