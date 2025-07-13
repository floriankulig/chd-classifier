# CHD Classification Project

## Overview

This project applies machine learning techniques to predict Coronary Heart Disease (CHD) risk using medical data. The goal is to identify the most accurate classification model and discover the most predictive factors for CHD diagnosis.

## Key Findings

### 🏆 Best Performing Models

1. **Support Vector Machine (SVM)** - F1: 0.858, Accuracy: 83.9%
2. **K-Nearest Neighbors (KNN)** - F1: 0.851, Accuracy: 83.0%
3. **Logistic Regression** - F1: 0.837, Accuracy: 82.6%

### 📊 Most Important Risk Factors

The analysis revealed the following features as most predictive of CHD:

1. **AP (Angina Pectoris)** - Most significant predictor
2. **RZ (ST Depression)** - Cardiac stress indicator
3. **BloodSugar** - Diabetes-related risk factor
4. **HFmax (Maximum Heart Rate)** - Cardiovascular fitness indicator
5. **ECG_ST** - Specific ECG abnormality pattern

### 🔍 Clinical Insights

- **Angina Pectoris** emerged as the strongest single predictor, confirming its clinical importance as a CHD symptom
- Traditional risk factors like cholesterol and blood pressure were less discriminative than symptomatic indicators
- The **ECG "ST" pattern** was the most relevant among ECG classifications
- Age, while important, was not among the top 5 predictors when combined with other factors

## Dataset Information

- **Size**: 270 patients (after removing 1 invalid entry)
- **Features**: 11 medical parameters
- **Target**: Binary CHD classification (55% positive, 45% negative)
- **Data Quality**: Minimal missing values, primarily cholesterol (handled via imputation)

### Features Analyzed

- Age, Blood Pressure, Cholesterol, Blood Sugar
- Maximum Heart Rate (HFmax), ST Depression (RZ)
- Gender, ECG patterns (Normal, LVH, ST), Angina Pectoris (AP)

## Methodology

### 1. Data Preprocessing

- **Missing Value Treatment**: Cholesterol zeros imputed using similar patient profiles
- **Categorical Encoding**: Label encoding for binary variables, one-hot encoding for ECG
- **Data Splitting**: 75% training, 25% testing with stratified sampling

### 2. Model Comparison

Evaluated 7 classification algorithms:

- K-Nearest Neighbors, Support Vector Machine, Logistic Regression
- Decision Tree, Random Forest, Gradient Boosting, Neural Network

### 3. Optimization Techniques

- **Hyperparameter Tuning**: Grid search for top 3 models
- **Feature Engineering**: Tested polynomial features, interactions, dimensionality reduction
- **Cross-Validation**: 5-fold CV for robust performance estimation

### 4. Evaluation Metrics

- Accuracy, Precision, Recall, F1-Score
- ROC-AUC and Precision-Recall curves
- Confusion matrices for detailed error analysis

## Model Performance Details

| Model               | Accuracy | F1-Score | Precision | Recall | AUC   |
| ------------------- | -------- | -------- | --------- | ------ | ----- |
| SVM (Optimized)     | 83.9%    | 0.858    | 83.5%     | 88.2%  | 0.882 |
| KNN (Optimized)     | 83.0%    | 0.851    | 84.6%     | 84.3%  | 0.845 |
| Logistic Regression | 82.6%    | 0.837    | 85.1%     | 83.5%  | 0.894 |

### Model Characteristics

- **SVM**: Highest sensitivity (88.2%) - best for screening applications
- **KNN**: Most balanced performance across all metrics
- **Logistic Regression**: Best overall discriminative ability (AUC: 0.894)

## Technical Implementation

### Requirements

```bash
conda install scikit-learn conda-forge::umap-learn seaborn numpy=1.26.4
```

### Key Libraries Used

- **scikit-learn**: Core ML algorithms and evaluation
- **pandas/numpy**: Data manipulation and analysis
- **seaborn/matplotlib**: Visualization
- **umap-learn**: Advanced dimensionality reduction

## Visualization Insights

### Dimensionality Reduction

- **PCA**: Explained 34% variance in first 2 components
- **t-SNE**: Revealed cluster patterns in patient data
- **UMAP**: Best separation of CHD classes

### Decision Boundaries

Models showed different decision-making patterns:

- SVM: Complex non-linear boundaries
- KNN: Local neighborhood-based decisions
- Logistic Regression: Linear separation with clear probabilistic interpretation

## Clinical Implications

### For Medical Practice

1. **Screening Priority**: Patients with angina pectoris require immediate attention
2. **Risk Assessment**: Combine ST depression, blood sugar, and max heart rate for comprehensive evaluation
3. **ECG Interpretation**: ST pattern abnormalities are highly indicative of CHD risk

### For Further Research

- Traditional risk factors (cholesterol, blood pressure) may need contextual interpretation
- Symptom-based indicators outperform demographic factors in this dataset
- Model could support clinical decision-making but requires validation on larger, diverse populations

## Limitations & Future Work

### Current Limitations

- Small dataset size (270 patients)
- Single-center data (potential selection bias)
- Limited external validation

### Future Enhancements

1. **Data Expansion**: Include additional clinical parameters and larger patient cohorts
2. **Advanced Models**: Explore ensemble methods and deep learning approaches
3. **Temporal Analysis**: Incorporate patient history and disease progression
4. **External Validation**: Test on independent datasets from different medical centers
5. **Feature Interactions**: Deeper analysis of risk factor combinations

## Usage

The complete analysis is available in `chd_classifier.ipynb`. The notebook runs end-to-end in seconds and includes:

- Comprehensive data exploration
- Model training and optimization
- Detailed performance analysis
- Clinical interpretation of results

## Project Structure

```
chd-classifier/
├── chd_classifier.ipynb    # Main analysis notebook
├── CHD_Classification.csv  # Dataset
├── helpers/                # Utility functions
│   ├── eval_model.py      # Model evaluation
│   ├── optimizer.py       # Hyperparameter optimization
│   ├── plotters.py        # Visualization functions
│   └── feature_engineering.py  # Feature engineering utilities
└── README.md              # This file
```
