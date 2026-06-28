# pharmacy-desert-risk-prediction
Predicting pharmacy desert formation risk using machine learning, spatial-temporal features, and higher-order network models.
# Pharmacy Desert Risk Prediction Using Machine Learning

## Section Content

1. Data Collection
2. Data Aggregation and Merging
3. Feature Engineering
4. Exploratory Data Analysis
5. Model 1 — XGBoost
6. Model 2 — LSTM
7. Model 3 — ST-GNN
8. Model 4 — HGNN
9. Model 5 — Tensor Train Network
10. Baseline Comparison
11. Final Findings and Conclusion

---

## Overview

This project focuses on predicting pharmacy desert formation risk at the county-year level using machine learning and higher-order network modeling. The main goal is to classify counties as **Low Risk** or **High Risk** based on pharmacy access pressure, healthcare access indicators, economic conditions, spatial relationships, and temporal changes.

The project compares traditional machine learning, temporal deep learning, spatial-temporal graph modeling, higher-order network modeling, and tensor-based feature interaction modeling.

---

## Dataset

The project uses multiple public datasets:

* **CMS Medicare Part D**
  Pharmacy utilization data including ZIP, claims, drug cost, fills, beneficiaries, and pharmacy-related records.

* **County Health Rankings**
  County-level health outcome ranks and health factor ranks.

* **USDA ERS County-Level Data**
  Economic variables such as employment, labor force, and unemployment rate.

* **ZIP–County Crosswalk**
  Used to map ZIP-level CMS records to county-level FIPS codes.

Large raw data files are not included in this repository due to file size. The notebooks include the workflow used for data collection, aggregation, merging, and preprocessing.

---

## Data Preprocessing

The preprocessing workflow included:

* Downloading and preparing CMS Medicare Part D data.
* Using DuckDB to process large CMS datasets efficiently.
* Mapping ZIP-level pharmacy records to county FIPS codes.
* Aggregating pharmacy data to the county-year level.
* Cleaning County Health Rankings and USDA ERS datasets.
* Converting missing or unavailable values such as `NR` into proper missing values.
* Removing fully missing columns.
* Imputing missing USDA economic values.
* Creating a final county-year dataset for modeling.

The final dataset was structured in county-year format and used for feature engineering, EDA, and model training.

---

## Feature Engineering

Key features were engineered to measure pharmacy desert formation risk.

### Pharmacy Count

`pharmacy_per_county_year`

This feature represents the number of unique pharmacies in each county-year.

### Pharmacy Accessibility Pressure

`bene_per_pharmacy_PAP`

PAP measures beneficiary load per pharmacy.

Formula:

`PAP = Total Beneficiaries / Number of Pharmacies`

A higher PAP value means each pharmacy is serving more beneficiaries, which indicates higher access pressure.

### Temporal Desert Formation Feature

`access_pressure_change_TDF`

TDF captures yearly percentage change in PAP.

Interpretation:

* Negative value = access improved
* Near zero = stable access
* Positive value = access worsened
* 25% or higher PAP increase = High Risk

### Target Variable

`Desert_Formation_Risk`

The target variable classifies each county-year as:

* **Low Risk**
* **High Risk**

A county-year was labeled High Risk when PAP increased by 25% or more compared with the previous year.

---

## Exploratory Data Analysis

Exploratory Data Analysis was performed to understand pharmacy access pressure patterns across states, counties, and years.

Main EDA findings:

* Pharmacy access pressure varied across counties.
* Some counties showed sharp increases in PAP over time.
* Lowndes, GA showed one of the largest increases in pharmacy access pressure.
* Correlation analysis showed relationships between pharmacy utilization, health rankings, unemployment, and pharmacy access pressure.
* EDA confirmed that county-level risk patterns exist and supported the need for machine learning classification.

---

## Models Used

## 1. XGBoost Model

XGBoost was used as the structured tabular baseline model.

In this project:

* County-year features were used as input.
* The model used pharmacy, health, economic, PAP, and TDF features.
* The objective was binary classification.
* Log loss was used as the evaluation metric.
* Feature importance was used to interpret the model.

XGBoost performed strongly on structured county-year data and remained the best practical baseline because of its balanced performance and interpretability.

---

## 2. LSTM Model

Long Short-Term Memory was used to capture temporal patterns.

In this project:

* 2-year county sequences were used as input.
* The model used LSTM layers with 64 hidden units.
* Adam optimizer was used.
* BCEWithLogitsLoss was used as the loss function.
* The output classified counties as Low Risk or High Risk.

LSTM learned temporal patterns, but it struggled to detect High-Risk cases because of class imbalance.

---

## 3. ST-GNN Model

Spatial-Temporal Graph Neural Network was used to capture both time-based and neighboring-county relationships.

In this project:

* Spatial features were created using Queen contiguity.
* Counties were considered neighbors if they shared a border or corner.
* Spatial lag features were created from nearby counties.
* 3-year county sequences were used.
* The model was evaluated using accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC.

ST-GNN added neighboring-county context and detected more High-Risk cases than LSTM, but class separation remained weak.

---

## 4. HGNN Model

Hypergraph Neural Network was used to capture higher-order relationships among counties.

Unlike ST-GNN, which focuses mainly on direct neighboring counties, HGNN can learn group-level patterns among counties that share similar pharmacy, health, economic, and temporal characteristics.

In this project:

* 2-year county-level sequences were used.
* Higher-order county relationships were modeled.
* Hypergraph convolution, temporal learning, dropout, and fully connected layers were used.
* ROC-AUC was used to measure class separation.

HGNN captured higher-order county patterns, but High-Risk recall remained weak.

---

## 5. Tensor Train Network

Tensor Train Network was used to learn high-dimensional feature interactions.

In this project:

* Pharmacy, health, economic, spatial, and temporal engineered features were used.
* A Tensor Train layer was used to learn compact feature interactions.
* ReLU activation, fully connected layers, and sigmoid output were used.
* The final output was a probability for High-Risk classification.

Tensor Train achieved the highest accuracy, but it had very weak High-Risk recall. This means many actual High-Risk counties were predicted as Low Risk.

---

## Model Evaluation

| Model        | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| ------------ | -------: | --------: | -----: | -------: | ------: |
| XGBoost      |     0.82 |      0.75 |   0.82 |     0.77 |     N/A |
| LSTM         |     0.83 |      0.64 |   0.83 |     0.75 |     N/A |
| ST-GNN       |     0.72 |      0.26 |   0.25 |     0.26 |    0.58 |
| HGNN         |     0.72 |      0.23 |   0.26 |     0.29 |    0.58 |
| Tensor Train |     0.84 |      0.35 |   0.06 |     0.08 |     N/A |

---

## Baseline Comparison

XGBoost was used as the baseline model because it performs well on structured county-year tabular data.

Advanced models were compared against XGBoost:

* LSTM captured temporal patterns.
* ST-GNN captured spatial-temporal neighboring-county patterns.
* HGNN captured higher-order county relationships.
* Tensor Train captured high-dimensional feature interactions.

Tensor Train achieved the highest accuracy at 0.84, but its High-Risk recall was very weak. Accuracy alone was not enough because the main goal of this project was to identify High-Risk counties early.

XGBoost remained the best practical baseline because it provided balanced performance and interpretable feature importance.

---

## Summary of Findings

| Finding                                  | Detail                                                                      |
| ---------------------------------------- | --------------------------------------------------------------------------- |
| PAP measured access pressure             | PAP measured beneficiary load per pharmacy.                                 |
| TDF captured yearly change               | TDF measured yearly percentage change in PAP.                               |
| XGBoost was strongest practical baseline | It provided balanced performance and interpretable feature importance.      |
| LSTM learned temporal patterns           | However, it missed High-Risk cases.                                         |
| ST-GNN added spatial context             | It included neighboring-county relationships but had weak class separation. |
| HGNN captured higher-order relationships | It modeled group-level county patterns but recall remained limited.         |
| Tensor Train had highest accuracy        | It achieved 0.84 accuracy but weak High-Risk recall.                        |
| Accuracy alone was not enough            | High-Risk recall was important because the goal was early risk detection.   |

---

## Conclusion

This project built a county-year pharmacy desert risk prediction pipeline using pharmacy utilization, health, economic, spatial, and temporal features. PAP and TDF were created to measure access pressure and yearly change in pharmacy accessibility. Multiple models were compared, including XGBoost, LSTM, ST-GNN, HGNN, and Tensor Train.

The results showed that Tensor Train achieved the highest accuracy, but it had weak High-Risk recall. Therefore, Tensor Train was not the best practical model despite its high accuracy. XGBoost remained the best practical baseline because it provided balanced performance, reliable results, and interpretable feature importance.

Overall, this project shows that machine learning can support early pharmacy desert risk monitoring and proactive pharmacy access planning.

---

## Repository Files

* `1_WEEK_2_DATA_COLLECTION_AND_CLEANING.ipynb`
* `2_Week_3_Data_Aggregation_and_Merging.ipynb`
* `3_WEEK_3_EDA_and_Feature_Engineering.ipynb`
* `4_XGBOOST_LSTM_TEMPORAL_MODELING.ipynb`
* `5_STGNN_SPATIAL_TEMPORAL_MODELING.ipynb`
* `6_HGNN_HIGHER_ORDER_NETWORK_MODELING.ipynb`
* `7_TENSOR_TRAIN_FEATURE_INTERACTION_MODELING.ipynb`
* `PHARMACY_DESERT_RISK_PREDICTION_USING_MACHINE_LEARNING.pdf`
* `Pharmacy Desert Risk Prediction Using Machine Learning.pptx`

---

## How to Run

1. Clone or download the repository.
2. Install the required Python libraries.
3. Run the notebooks in order from 1 to 7.
4. Start with data collection and cleaning.
5. Continue with aggregation, merging, feature engineering, and EDA.
6. Run the modeling notebooks for XGBoost, LSTM, ST-GNN, HGNN, and Tensor Train.
7. Review the final report and presentation for project results.

---

## Required Libraries

The project uses the following Python libraries:

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* xgboost
* torch
* duckdb
* geopandas
* libpysal
* pyarrow
* openpyxl

---

## Author

Sriram Pranay Sharma Devaraju
Regis University
MSDS692 Practicum
