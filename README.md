# CIND820 - Advance Data Analytics Project

# Credit Card Fraud Detection

A comprehensive machine learning project comparing batch processing and online learning approaches for detecting fraudulent credit card transactions.

## 📋 Overview

This project implements and evaluates multiple machine learning models for credit card fraud detection using the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle. The analysis compares traditional batch learning algorithms with modern online (incremental) learning techniques.

## 🗂️ Project Structure

```
├── Gangwani_Gunjan_ProjectDesign    # Outlines the Project Design
├── Gangwani_Gunjan_LitReview.pdf     # Explains the background and Literature Review 
├── CIND820_EDA_Analysis.pdf          # Exploratory Data Analysis
├── CIND820_Batch_Processing.pdf      # Batch learning models implementation
├── CIND820_Online_Learning_Models.pdf # Online learning models implementation
├── requirements.txt                   # Python dependencies
└── README.md                         # Project documentation
```

## 🔍 Analysis Components

### 1. Exploratory Data Analysis (EDA)
- Dataset overview and statistical analysis
- Comprehensive profiling report generation
- Feature distribution visualization by class
- Correlation analysis with heatmaps
- Data preprocessing and scaling techniques (RobustScaler, MinMaxScaler)

### 2. Batch Processing Models
Implemented and evaluated traditional machine learning models:
- **Logistic Regression** (with and without scaling)
- **Random Forest Classifier** (with scaling and SMOTE resampling)
- **Linear Support Vector Classifier (LinearSVC)**
- **XGBoost Classifier** (with class weight balancing)
- **Feature Selection** techniques for model optimization

**Key Techniques:**
- Standard scaling for feature normalization
- SMOTE (Synthetic Minority Over-sampling Technique)
- Class weight balancing
- Feature importance-based selection

### 3. Online Learning Models
Implemented incremental learning algorithms using the River library:
- **Logistic Regression** (with and without StandardScaler)
- **Adaptive Random Forest (ARF)**
- **Passive Aggressive Classifier**
- **AdaBoost Classifier** (with Hoeffding Tree)
- **ARFClassifier** (with Hard Sampling)
- **ADWIN Boosting** (drift detection enabled)

**Advanced Techniques:**
- Hard Sampling for imbalanced data
- Concept drift detection with ADWIN
- Real-time model updates with `learn_one()`

## 📊 Evaluation Metrics

All models are evaluated using:
- **F1 Score** - Harmonic mean of precision and recall
- **MCC (Matthews Correlation Coefficient)** - Balanced measure for imbalanced datasets
- **ROC AUC Score** - Area under the receiver operating characteristic curve

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Jupyter Notebook or Google Colab

### Installation

1. Clone the repository:
```bash
git clone https://github.com/GunjanGangwani/CIND820-Credit-Card-Fraud-Detection.git
cd credit-card-fraud-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset:
The notebooks automatically download the dataset using `kagglehub`:
```python
import kagglehub
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
```

## 📈 Key Results

### Batch Processing Models
- **Best F1 Score:** XGBoost with Balanced Weights (0.8889)
- **Best MCC Score:** XGBoost with Balanced Weights (0.8889)
- **Best ROC AUC:** Random Forest with SMOTE (0.9848)

### Online Learning Models
- **Best F1 Score:** Passive Aggressive Classifier (0.8000)
- **Best MCC Score:** Passive Aggressive Classifier (0.8020)
- **Best ROC AUC:** Passive Aggressive Classifier (0.9398)

## 🛠️ Technologies Used

- **Data Processing:** pandas, numpy
- **Visualization:** matplotlib, seaborn, ydata-profiling
- **Batch Learning:** scikit-learn, xgboost, imblearn
- **Online Learning:** river
- **Dataset Management:** kagglehub

## 📝 Dataset Information

- **Total Transactions:** 284,807
- **Features:** 30 (V1-V28 PCA transformed, Time, Amount)
- **Target Variable:** Class (0 = legitimate, 1 = fraudulent)
- **Class Distribution:** Highly imbalanced (~0.17% fraud cases)

## 🔬 Methodology

1. **Data Preprocessing:** Feature scaling and normalization
2. **Class Imbalance Handling:** SMOTE, class weights, hard sampling
3. **Model Training:** Batch (70/30 split) and incremental learning (test then train strategy)
4. **Performance Evaluation:** Multiple metrics for comprehensive assessment
5. **Comparison Analysis:** Batch vs. online learning trade-offs

## F1 Score Comparison for Batch and Online Learning
<img width="846" height="814" alt="Batch Processing" src="https://github.com/user-attachments/assets/ea2f1fce-7323-4e3b-9990-633b4bedab4a" />


<img width="846" height="746" alt="Online Learning" src="https://github.com/user-attachments/assets/e9646659-6593-476a-94db-26d95511cd20" />


## MCC Score Comparison for Batch and Online Learning
<img width="846" height="814" alt="Batch Processing MCC" src="https://github.com/user-attachments/assets/316d5ca6-059d-4c39-a5f7-8334e79bcc7b" />

<img width="846" height="717" alt="Online Learning MCC" src="https://github.com/user-attachments/assets/bf25fc92-1a50-4a0a-8fd0-a59e7c3153a4" />


## 📚 References

- Dataset: [Credit Card Fraud Detection - Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- River Library: [Online Machine Learning in Python](https://riverml.xyz/)

## 👤 Author

Gunjan Gangwani - [Github Profile](https://github.com/GunjanGangwani)

---
