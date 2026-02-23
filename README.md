# Unravelling the wind impact of clusters of storms, a case study over the French insurer Generali
Laura Hasbini<sup>1,2\*</sup>, Pascal Yiou<sup>1</sup>, Quentin Hénaff<sup>2</sup>, Laurent Boissier<sup>2</sup>, Arthur Perringaux<sup>2</sup>

<sup>1</sup> Laboratoire des Sciences du Climat et de l’Environnement, UMR 8212CEA-CNRS-UVSQ,
Université Paris-Saclay, Gif-sur-Yvette, France.
<sup>2</sup> Generali France SAS, 93210, Saint Denis, France.

\* corresponding authors: laura.hasbini@lsce.ipsl.fr

## Abstract
Winter windstorms are the most damaging natural hazard in Europe in terms of insured losses, with impacts often arising from clusters of storms rather than isolated events. In reinsurance practice, losses are aggregated over sequences of storms affecting the same region within a limited time window. Yet, attributing individual damages to specific storm events remains challenging. The distribution of costs between insurance and reinsurance companies critically depends on this attribution, making robust and transparent criteria essential for a fair allocation of losses. This study introduces a method to systematically link individual insurance claims to extra-tropical cyclones, enabling event-based attribution of damages. The method is applied to the Generali France loss portfolio to build a catalogue linking individual claims to storm events. The resulting catalogue provides a foundation for risk assessment, loss modelling, and reinsurance applications. We focus on storm clusters, defined as successive storms affecting the same region within a 96-hour period. We show that damaging storms within clusters are more intense than isolated events, with lower minimum sea-level pressure and higher vorticity. Losses within clusters are dominated by a single storm, accounting on average for about 70% of total cluster losses, while the remaining storms collectively contribute the residual losses. Overall, 85% of windstorm-related losses over the 1998-2024 period are associated with clustered events, and damaging storms occur in clusters more frequently than expected from the full population of extratropical cyclones. These results highlight the importance of explicitly accounting for storm clustering in insurance and reinsurance risk management.

---
## Data references
### Input data
|       Dataset       |               Description                    |               Reference/DOI          |
|:-------------------:|:--------------------------------------------:|:--------------------------------:|
|TBD|TBD|[![DOI]()]()|

## Output data
|       Dataset       |              Description                    |           Repository Link        |                   DOI                   |
|:-------------------:|:-------------------------------------------:|:--------------------------------:|:---------------------------------------:|
|TBD|TBD|[Link]()|[![DOI]()]()|

## Reproduce our experiment

### Requirements

| Requirement | Notes |
|------------|-------|
| Python | Version 3.11 |
| Conda | Used for environment management |
| Internet connection | Required for models and external data |

---
### Core pipeline execution
The core pipeline consists of three main steps.

|Script Name | Description |
|Fpreproces_reports.py| Pre process raw IFRC reports, formatting and text selection|
|llm_extraction.py| Extract hazards and impacts using LLMs |
|postprocess_results.py| Reclassify, standardize, and geocode extracted impacts|


## Repository content

### Bash Scripts 

### Tracks Processes

### Claim Asssociation

### Fct 

### Result Analysis

Jupyter notebooks used for data inspection, validation, and analysis.

| Notebook | Purpose |
|--------:|:--------|
| `download_external_sources.ipynb` | Download IFRC Monty and IFRCGo data via APIs |
| `labelled_extracted_row_matching.ipynb` | Match manually labelled data with LLM extracted results |
| `open_data.ipynb` | User guidelines to open the database from different formats |
| `result_data_overview.ipynb` | Overview plots and summary statistics of extracted impacts |
| `validation_accuracy.ipynb` | Accuracy evaluation of extracted impacts |
| `validation_coverage.ipynb` | Coverage assessment across regions and hazards |
| `validation_external_sources.ipynb` | Comparison with external impact databases |
| `validation_sensitivity_analysis.ipynb` | Sensitivity analysis across different LLM models |

---

### src

Source code implementing the ROUGE extraction and post processing pipeline.

| Module | Description |
|------:|:------------|
| `accuracy.py` | Functions to compute accuracy and validation metrics |
| `classOutput.py` | Classes defining standardized LLM extraction outputs |
| `client.py` | API client setup for OpenAI, Groq, and OpenStreetMap |
| `data.py` | Centralized path and directory definitions |
| `external_comparaison.py` | Aggregation and comparison with external datasets |
| `geocoding.py` | End to end geocoding pipeline |
| `geocoding_utils.py` | Utility functions supporting geocoding operations |
| `hazard_def.py` | Definitions of hazard classes and categories |
| `impact_def.py` | Definitions of impact classes and categories |
| `ImpactRegistry.py` | Registry and mapping of impact types |
| `labelling_helpers.py` | Helper functions for manual labelling workflows |
| `LLM_functions/` | Prompt templates and LLM query functions |
| `logger_setup.py` | Logging configuration and utilities |
| `post_processing_functions.py` | Impact post processing and harmonization functions |
| `prompt_examples.py` | Example prompts used for LLM extraction |
| `prompt_hazards.py` | Prompt functions for hazard extraction |
| `prompt_impact.py` | Prompt functions for impact extraction |
| `sanity_checks.py` | Consistency and sanity checks on extracted data |
| `text_processing_functions.py` | Text cleaning and pre processing utilities |
| `units.py` | Unit definitions per impact class and category |
| `utils.py` | General utility and helper functions |