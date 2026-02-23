# Unravelling the wind impact of clusters of storms, a case study over the French insurer Generali
Laura Hasbini<sup>1,2\*</sup>, Pascal Yiou<sup>1</sup>, Quentin Hénaff<sup>2</sup>, Laurent Boissier<sup>2</sup>, Arthur Perringaux<sup>2</sup>

<sup>1</sup> Laboratoire des Sciences du Climat et de l’Environnement, UMR 8212CEA-CNRS-UVSQ,
Université Paris-Saclay, Gif-sur-Yvette, France.
<sup>2</sup> Generali France SAS, 93210, Saint Denis, France.

\* corresponding authors: laura.hasbini@lsce.ipsl.fr

**Publication DOI:** [10.5194/egusphere-2025-3138](https://doi.org/10.5194/egusphere-2025-3138)

## Abstract
Winter windstorms are the most damaging natural hazard in Europe in terms of insured losses, with impacts often arising from clusters of storms rather than isolated events. In reinsurance practice, losses are aggregated over sequences of storms affecting the same region within a limited time window. Yet, attributing individual damages to specific storm events remains challenging. The distribution of costs between insurance and reinsurance companies critically depends on this attribution, making robust and transparent criteria essential for a fair allocation of losses. This study introduces a method to systematically link individual insurance claims to extra-tropical cyclones, enabling event-based attribution of damages. The method is applied to the Generali France loss portfolio to build a catalogue linking individual claims to storm events. The resulting catalogue provides a foundation for risk assessment, loss modelling, and reinsurance applications. We focus on storm clusters, defined as successive storms affecting the same region within a 96-hour period. We show that damaging storms within clusters are more intense than isolated events, with lower minimum sea-level pressure and higher vorticity. Losses within clusters are dominated by a single storm, accounting on average for about 70% of total cluster losses, while the remaining storms collectively contribute the residual losses. Overall, 85% of windstorm-related losses over the 1998-2024 period are associated with clustered events, and damaging storms occur in clusters more frequently than expected from the full population of extratropical cyclones. These results highlight the importance of explicitly accounting for storm clustering in insurance and reinsurance risk management.

---
## Repository content

### Bash Scripts 

Shell scripts used to run the **core data processing and analysis pipeline**.  
They are designed to be executed sequentially.

| Script | Description |
|------:|------------|
| `1_storm_preprocessing.sh` | Preprocessing of storm data |
| `2_association_claims_storms_per_year.sh` | Association of claims and storms on a yearly basis |
| `3_association_claims_storms_varying_radius.sh` | Sensitivity analysis with varying spatial association radius |
| `4_association_gather_files.sh` | Aggregation of association outputs |
| `5_cluster_computation.sh` | Computation of storm and impact clusters |

### Tracks Processes

Python scripts used for **storm track processing, footprint computation, and impact clustering**.  
These scripts form the **core processing logic** and are typically called directly or wrapped by shell pipelines.

| Script | Description |
|------:|------------|
| `tracks_cluster.py` | Perform clustering of storm tracks |
| `tracks_cluster_impact.py` | Cluster impacts associated with storm tracks |
| `tracks_convert_TE.py` | Convert track data to the TE (TempestExtreme) format |
| `tracks_filter_FR.py` | Filter storm tracks over the France domain |
| `tracks_footprints.py` | Generate storm wind footprints |
| `tracks_footprints_varying_radius.py` | Sensitivity analysis of footprints with varying spatial radius |
| `tracks_merge.py` | Merge storm track files into unified datasets |

### Claim Association
Python modules implementing the **association between insurance claims and storm events**, as well as performance evaluation and aggregation utilities.  
These scripts are part of the **core analytical workflow** and are typically executed via pipeline scripts or imported as modules.

| Module | Description |
|------:|------------|
| `__init__.py` | Package initialization file |
| `claims_association_combine_files.py` | Combine association outputs into consolidated datasets |
| `claims_association_performances.py` | Compute performance metrics for claims–storm associations |
| `claims_association_storms.py` | Core logic to associate claims with storm events |
| `claims_association_storms_per_year.py` | Yearly association of claims and storms |
| `claims_association_storms_varying_radius.py` | Sensitivity analysis of associations using varying spatial radii |
| `claims_association_storms_varying_radius_per_year.py` | Yearly sensitivity analysis with varying association radii |
| `gather_anom_sinclim_v2.2.py` | Aggregate and process SINCLIM data (v2.2) |

### Fct 

Python modules providing **shared utilities, preprocessing routines, and domain-specific helpers** used across the storm–claims analysis workflow.  
These functions are typically **imported by higher-level scripts** rather than executed directly.

| Module | Description |
|------:|------------|
| `__init__.py` | Package initialization file |
| `fct_link_storm_claim.py` | Helper functions to link storm events with insurance claims |
| `fct_plot_claims.py` | Plotting utilities for claims and storm-related analyses |
| `paths.py` | Centralized definition of project paths and file locations |
| `preprocess_sinclim.py` | Preprocessing routines for SINCLIM datasets |
| `storm_eu.py` | Core utilities for handling European storm data |
| `storm_eu_cluster.py` | Utilities for clustering European storm events |

### Result Analysis

Jupyter notebooks dedicated to **impact assessment, clustering diagnostics, sensitivity analyses, and result visualization**.  
These notebooks are primarily used for **exploratory analysis, figure generation, and validation**.

| Notebook | Purpose |
|--------:|:--------|
| `association_impact.ipynb` | Analysis of impacts associated with storm–claim linkages |
| `cluster_impact.ipynb` | Evaluation of clustered storm impacts |
| `clusters_footprints.ipynb` | Analysis of storm footprint clustering |
| `flowchart_figure.ipynb` | Generation of workflow and methodology flowchart figures |
| `performances_sensitivity.ipynb` | Sensitivity analysis of association performance metrics |
| `sensitivity_cluster_definition.ipynb` | Sensitivity analysis to cluster definition choices |
| `sensitivity_varying_radius.ipynb` | Sensitivity analysis with varying spatial association radius |
| `summary_insurance_metrics.ipynb` | Summary and aggregation of insurance-related impact metrics |
| `summary_storm_tracks.ipynb` | Summary statistics and diagnostics of storm track datasets |
