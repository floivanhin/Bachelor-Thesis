# Bachelor Thesis
This repository contains the source code and supplementary materials for the bachelor's thesis with the title <br>"Analysis and prediction of urban mobility patterns: Comparing Machine Learning algorithms on New York City taxi trip data" <br> written by Florian Ivan Hinrichsen during the summer semester of 2026.
## Repository Overview
- **Data Pipeline** <br> This folder contains the complete source code used for the initial data preprocessing phase as well as the subsequent model training. The dataset creation files are provided to establish methodological transparency, but are not reproducable. The model training scripts are designed so that the results can be seamlessly reproduced.
- **Evaluation** <br> This folder provides the complete codebase utilized to generate the evaluation graphs and assess model performance, designed specifically so that the entire evaluation process is completely reproducible.
- **Plot Generation** <br> This folder contains the scripts used to create the figures for the exploratory data analysis. It also provides the script used to map out the 6-fold cross-validation methodology, with all scripts explicitly designed to be completely reproducible.
> **Note on Code Documentation:** 
> Because some of the scripts for data preprocessing, evaluation, and plot generation share a highly similar structure across different datasets and models, redundant comments have been omitted. The detailed comments provided in the first scripts apply analogously to the rest of the codebase.
## Prerequisites & Software
To reproduce the model training, evaluation, and visual figures, ensure your environment meets the following specifications:
### 1. Hardware Requirements
- **RAM:** Minimum 16 GB (32 GB recommended). Processing large tabular batches and fitting ensemble models like Random Forest requires substantial memory headroom.
- **Storage:** At least 10–20 GB of free disk space for processed datasets, saved model checkpoints, and evaluation figures. (Note: reproducing the raw data ingestion pipeline requires significantly more space for raw Parquet files).
- **Compute / GPU:** A CUDA-compatible NVIDIA GPU is strongly recommended. While the Feedforward Neural Network (FFNN) training supports CPU execution, reproducing the Random Forest models strictly requires an NVIDIA GPU due to the RAPIDS (cuml) dependency. (Executing the training notebook in a Google Colab instance with a free T4 GPU is fully sufficient).
### 2. Software & Python Environment
- **Python Version:** Python 3.10 or higher.
- It is strongly recommended to set up an isolated virtual environment.
## Suggested Workflow
The recommended workflow for this project looks like this:
### 1. Environment Setup
Clone this repository:
```bash
git clone https://github.com/floivanhin/Bachelor-Thesis.git
cd Bachelor-Thesis
```
Install all required dependencies:
```bash
pip install -r requirements.txt
```
> **Note on Data Preprocessing:**  
> The complete preprocessing pipeline is included for full transparency and code audit purposes in the `Data Pipeline` folder. This includes the `DataPreProcessing.ipynb` notebook and the associated `DataPreparation*.py` scripts. However, **preprocessing is not required to reproduce the results**. The finalized, preprocessed datasets are made publicly available through Kaggle (`https://www.kaggle.com/datasets/florianhinrichsen/nyc-taxi-data-2023-24`), allowing you to proceed directly to the data analysis, model training and evaluation.
### 2. Data Aquisition
The relevant datasets and pre-trained models are hosted on Kaggle. At the top of each notebook, a dedicated cell utilizes the Kagglehub API to automatically download the necessary files, requiring no manual download.
### 3. Data Analysis
To reproduce the exploratory data analysis, navigate to the `Plot Generation` folder and execute the following notebooks:
- `CorrelationHeatmap.ipynb`
- `FeatureCorrelations.ipynb`
- `Distributions.ipynb`
- `Timeseries.ipynb`
### 4. Model Training
To reproduce the model training process, execute the `ModelTraining.ipynb` notebook located in the `Data Pipeline` folder. Please be aware that training the Feedforward Neural Network may take multiple days to complete due to the large size of the dataset.
>**Note on Model Training**
> You can safely skip this step and proceed directly to the evaluation as the trained models are provided through dedicated Kaggle datasets
### 5. Evaluation
To reproduce the model evaluation, execute the notebooks located within the `Evaluation` folder. These files automatically download the pre-trained models from Kaggle and generate the predictions, requiring no manual setup.
## Author
**Florian Ivan Hinrichsen** <br>
Contact: hinrichsenf@googlemail.com / [github.com/floivanhin](https://github.com/floivanhin)
