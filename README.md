# Bachelor-Thesis: "Analysis and prediction of urban mobility patterns: Comparing Machine Learning algorithms on New York City taxi trip data"
This repository contains the source code and supplementary materials for the bachelor's thesis written by Florian Ivan Hinrichsen during the summer semester of 2026.
## Repository Structure
- Data Pipeline <br> This folder contains the complete source code used for the initial data preprocessing phase as well as the subsequent model training. The dataset creation files are provided to establish methodological transparency, but are not reproducable. The model training scripts are designed so that the results can be seamlessly reproduced.
- Evaluation <br> This folder provides the complete codebase utilized to generate the evaluation graphs and assess model performance, designed specifically so that the entire evaluation process is completely reproducible.
- Plot Generation <br> This folder contains the scripts used to create the figures for the exploratory data analysis, alongside specific evaluation plots. It provides the scripts used to plot the general evaluation results, visualize the training loss curves, and map out the 6-fold cross-validation methodology, with all scripts explicitly designed to be completely reproducible.
> **Note on Code Documentation:** 
> Because some of the scripts for data preprocessing, evaluation, and plot generation share a highly similar structure across different datasets and models, redundant comments have been omitted. The detailed comments provided in the first scripts apply analogously to the rest of the codebase.
## Prerequisites & Software
To reproduce the model training, evaluation, and visual figures, ensure your environment meets the following specifications:
### 1. Hardware Requirements
* **RAM:** Minimum 16 GB (32 GB recommended). Processing large tabular batches and fitting ensemble models like Random Forest requires substantial memory headroom.
* **Storage:** At least 10–20 GB of free disk space for processed datasets, saved model checkpoints, and evaluation figures. (Note: reproducing the raw data ingestion pipeline requires significantly more space for raw Parquet files).
* **Compute / GPU:** A CUDA-compatible GPU is recommended to accelerate Feedforward Neural Network (FFNN) training, though CPU execution is supported.
### 2. Software & Python Environment
* **Python Version:** Python 3.10 or higher.
* It is strongly recommended to set up an isolated virtual environment.
