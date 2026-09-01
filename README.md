# 📱 Predicting Smartphone Addiction (Big Data ML Pipeline)

## 📌 Project Overview
This repository contains a scalable Machine Learning pipeline designed to predict smartphone addiction. Working with a massive dataset of nearly **700,000 rows**, this project shifts away from traditional ML algorithms (like KNN or SVM) which bottleneck at this scale, and instead utilizes highly optimized Gradient Boosting frameworks (XGBoost, CatBoost).

## 🚀 The Three-Phase Architecture

### Phase 1: The XGBoost Baseline (`01_xgboost_baseline.ipynb`)
A mathematically strict, custom-built pipeline to establish a strong baseline.
* **The Dummy Variable Trap:** Safely handled Nominal data (Gender) with One-Hot Encoding, while applying Ordinal Mapping to ordered data (Stress Level).
* **Leakage-Free Imputation:** Utilized `IterativeImputer` within a rigorous `fit_transform` (train) and `transform` (test) protocol to prevent data leakage.
* **Selective Scaling:** Applied `StandardScaler` exclusively to continuous variables, protecting the binary and ordinal columns from distortion.
* **Result:** `86.9%` Accuracy via GridSearchCV.

### Phase 2: The CatBoost Grandmaster Architecture (`02_catboost_advanced.ipynb`)
To push past the 90% threshold, I adapted an Apache 2.0 open-source CatBoost architecture from Kaggle 4x Grandmaster Chris Deotte, and injected my own custom behavioral feature engineering.
* **Algorithmic Native Handling:** Leveraged CatBoost's native ability to process raw categorical text and handle numerical `NaNs` via tree-routing without manual imputation.
* **Advanced Feature Engineering (Residuals & Ratios):** Calculated `time_accounting_residual` to find hidden screen time, and fractioned daily behaviors.
* **My Custom Injections:** Added `is_sleep_deprived` (biologically thresholding sleep < 6 hours) and `spam_intensity` (notifications per screen hour).
* **OOF 5-Fold Stratified CV:** Trained 5 independent CatBoost models using early stopping, blending their outputs for a highly stable prediction.
* **Result:** `96.36%` AUC.

### Phase 3: Ensembling (`03_submission_ensembling.ipynb`)
A demonstration of Kaggle-style "Blending," reading the probability outputs of multiple models and merging them using weighted averages to capture edge cases.

## ⚖️ Legal & Acknowledgements
Phase 2 of this project utilizes the baseline CatBoost configuration, Stratified K-Fold loop, and residual feature logic from Chris Deotte's S6E8 Starter Notebook, released under the **Apache 2.0 Open Source License**. I extended this framework by restructuring the data loaders for local environment execution and injecting custom behavioral flags.