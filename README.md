# CIND820 - Advance Data Analytics Project

# Credit Card Fraud Detection

This repository presents a comprehensive fraud-detection pipeline using both batch machine learning models (scikit-learn, XGBoost) and online learning models (River).
The goal is to evaluate model effectiveness, efficiency, and stability when detecting rare fraudulent transactions in a highly imbalanced dataset.

The project uses the Kaggle [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), consisting of 284,807 anonymized transactions with only 0.17% fraud cases, making evaluation metrics and sampling methods critical.

## Video Presentation of the Initial Code
[5 Mins Video Presentation of Code](https://drive.google.com/file/d/11fJ6qkySEMg6J06W2y3ASwboDq483VVd/view?usp=sharing)

## 📌 Key Objectives

- Build and compare multiple batch models: Logistic Regression, Linear SVC, Random Forest, XGBoost, with variations such as scaling, class weighting, SMOTE, and feature selection.

- Build and assess online models capable of incremental learning on large, streaming datasets: Logistic Regression, Passive Aggressive, Adaptive Random Forest, Hoeffding Tree, AdaBoost, ADWINBoosting.

- Evaluate model effectiveness (F1, MCC), efficiency (runtime), and stability (behavior under imbalance, sampling, and scaling).

- Provide reproducible notebook code for both paradigms.



## 🗂️ Project Structure

```
├── Dataset                              # EDA Report by YProfiling 
├── Gangwani_Gunjan_ProjectDesign.pdf    # Outlines the Project Design
├── Gangwani_Gunjan_LitReview.pdf        # Explains the background and Literature Review 
├── CIND820_EDA_Analysis.pdf             # Exploratory Data Analysis
├── CIND820_Batch_Processing.ipynb       # Batch learning models notebook
├── CIND820_Online_Learning_Models.ipynb   # Online learning models notebook
├── CIND820_Batch_Processing.pdf         # Batch learning models implementation
├── CIND820_Online_Learning_Models.pdf   # Online learning models implementation
├── Gunjan_Gangwani_FinalReport.pdf      # Final Report with interpretations
├── Gunjan_Gangwani_FinalPresentation    # Final Project Presentation 
├── requirements.txt                     # Python dependencies
└── README.md                            # Project documentation
```
## 🛠️ Technologies Used

- **Data Processing:** pandas, numpy
- **Visualization:** matplotlib, seaborn, ydata-profiling
- **Batch Learning:** scikit-learn, xgboost, imblearn
- **Online Learning:** river
- **Dataset Management:** kagglehub

  
## 🔍 Analysis Components

**1. Exploratory Data Analysis (EDA)**

- Dataset summary and class imbalance profiling
- Distribution visualizations
- Correlation heatmaps
- PCA-transformed features (V1–V28)
- Scaling methods: StandardScaler

**2. Batch Processing Models**

Batch learning trains a model using the entire dataset in one go.
It is highly accurate when enough memory and compute are available.

**Models Implemented**

- Logistic Regression 
- Random Forest Classifier 
- Linear SVC 
- XGBoost

**Key Methods**

- Standard scaling
- Class weight balancing
- SMOTE oversampling
- Feature importance and selection

**Batch learning is ideal for:**

- Offline fraud detection systems
- Periodic retraining
- Scenarios where full historical data is accessible

**3. Online Learning Models**

Online learning updates the model one transaction at a time, making it suitable for real-time fraud detection and streaming data.

**Models Implemented**
- Logistic Regression
- Passive Aggressive Classifier
- Adaptive Random Forest (ARF)
- AdaBoost + Hoeffding Tree
- ADWINBoosting (drift-adaptive)

**Key Methods**

- learn_one() incremental updates
- StandardScaling Pipeline
- Hard Sampling for imbalanced streams
- ADWIN drift detection for changing fraud patterns

**Online learning is ideal for:**

- Lightweight, memory-efficient learning
- Live transaction monitoring
- High-velocity data streams
- Systems requiring instant model adaptation


**📈 Visual Comparison**

## F1 Score Comparison for Batch and Online Learning
Batch Processing
<img width="806" height="565" alt="Batch Processing" src="https://github.com/user-attachments/assets/df33f583-8cbe-4b12-a72e-35605655ca48" />

Online Processing
<img width="806" height="565" alt="Online Processing" src="https://github.com/user-attachments/assets/c51d6eba-e50f-4ce0-b043-49c883d729b0" />


## Runtime Comparison for Batch and Online Learning
Batch Processing
<img width="806" height="565" alt="Batch Processing" src="https://github.com/user-attachments/assets/fe8a0140-0328-49db-9181-5f8687dc1845" />

Online Processing
<img width="806" height="565" alt="Online Processing" src="https://github.com/user-attachments/assets/b3902bfa-1d08-40ad-958f-b52b499058ea" />


**🚀 Getting Started**
Install Dependencies
```bash
pip install -r requirements.txt
```

Dataset Download
```bash
import kagglehub
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
```

Run Notebooks

Open the batch or online notebook in Jupyter or Colab.



**📌 Conclusion**

This project demonstrates that:

- **Batch learning models** such as XGBoost achieve the highest predictive performance (F1 ~ 89%), benefiting from full-dataset access and strong ensemble capabilities.

- **Online learning models**, while slightly less accurate, provide significant advantages in speed, memory efficiency, and adaptability, making them suitable for real-time fraud detection pipelines.

- **Passive Aggressive** performs remarkably well for an online model, achieving 80% F1 and MCC, while running extremely fast.

- **Random Forest and XGBoost** remain strong choices in batch environments but are computationally heavier.

The optimal system often uses a hybrid approach:

- an online model monitors real-time streams,
- while a batch model is periodically retrained offline with accumulated data.

This analysis highlights how the choice between batch and online learning depends strongly on the operational requirements—accuracy vs. speed, memory vs. adaptability, and offline vs. real-time deployment.




## 👤 Author

Gunjan Gangwani - [Github Profile](https://github.com/GunjanGangwani)

---
