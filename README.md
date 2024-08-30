# Heart Attack Prediction Using Machine Learning

![Heart Attack Prediction](https://github.com/user-attachments/assets/cc2f7d2a-71ed-4419-b283-89dd698b1b5e)


## Overview
This repository contains a machine learning project aimed at predicting the likelihood of a heart attack based on a set of medical attributes. The dataset includes various patient features such as age, cholesterol levels, and exercise-induced angina, among others. The goal of this project is to develop a robust predictive model that can assist in early diagnosis and prevention of heart-related conditions.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Data Preprocessing](#data-preprocessing)
- [Model Building](#model-building)
- [Model Evaluation](#model-evaluation)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Dataset
The dataset used in this project is publicly available and contains patient information with various medical indicators. The dataset can be accessed via [Kaggle](https://www.kaggle.com/datasets) or other public repositories.

### Data Source
- **Source**: [Kaggle - Heart Attack Analysis & Prediction Dataset](https://www.kaggle.com/nareshbhat/heart-attack-analysis-prediction-dataset)
- **Size**: 303 rows, 14 columns
- **Attributes**: Age, Gender, Chest Pain Type, Cholesterol, Resting Blood Pressure, Max Heart Rate, etc.

## Features
The dataset includes the following features:

- **Age**: Age of the patient (years).
- **Sex**: Gender of the patient (1 = Male, 0 = Female).
- **cp (Chest Pain Type)**:
  - 0: Typical angina
  - 1: Atypical angina
  - 2: Non-anginal pain
  - 3: Asymptomatic
- **trestbps**: Resting blood pressure (in mm Hg).
- **chol**: Serum cholesterol in mg/dl.
- **fbs**: Fasting blood sugar > 120 mg/dl (1 = True; 0 = False).
- **restecg**: Resting electrocardiographic results.
- **thalach**: Maximum heart rate achieved.
- **exang**: Exercise-induced angina (1 = Yes; 0 = No).
- **oldpeak**: ST depression induced by exercise relative to rest.
- **slope**: Slope of the peak exercise ST segment.
- **ca**: Number of major vessels (0-3) colored by fluoroscopy.
- **thal**: Thalassemia (0 = Normal; 1 = Fixed defect; 2 = Reversible defect).
- **target**: Heart attack occurrence (1 = Yes, 0 = No).

## Exploratory Data Analysis (EDA)
Extensive EDA was performed to understand the data distribution, identify correlations, and uncover hidden patterns. Key steps included:

- **Univariate Analysis**: Histograms, box plots, and density plots were created to inspect the distribution of individual features.
- **Bivariate Analysis**: Pair plots and correlation heatmaps were used to explore relationships between features and the target variable.
- **Outlier Detection**: Outliers were identified and analyzed using statistical methods.

## Data Preprocessing
To ensure the data was suitable for model building, the following preprocessing steps were taken:

- **Handling Missing Values**: No missing values were found in the dataset.
- **Feature Scaling**: Continuous features were standardized using z-score normalization.
- **Encoding Categorical Variables**: Categorical features were converted into numerical values using one-hot encoding.

## Model Building
Multiple machine learning models were tested to find the best-performing one. The models included:

- **Logistic Regression**
- **Random Forest**
- **XGBoost**
- **Neural Networks (Keras)**

### Model Architecture for Neural Networks:
```python
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=10)
```

## Model Evaluation
The models were evaluated based on the following metrics:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **ROC-AUC Score**

### Results:
- **Training Accuracy**: 99.59%
- **Test Accuracy**: 83.61%
- **ROC-AUC Score**: 0.91 (for the best model)

## Hyperparameter Tuning
Hyperparameter tuning was performed using `GridSearchCV` and `RandomizedSearchCV` to optimize the model's performance.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
```

## Results
The final model demonstrates strong predictive capability with the following key results:

- **Best Model**: XGBoost Classifier
- **Training Accuracy**: 99.59%
- **Test Accuracy**: 83.61%
- **Key Insights**:
  - Higher cholesterol levels and exercise-induced angina are significant predictors of heart attacks.
  - The model's predictions are reliable with a high ROC-AUC score, indicating a strong ability to distinguish between patients with and without heart attacks.

## Installation
Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/heart-attack-prediction.git
cd heart-attack-prediction
pip install -r requirements.txt
```

## Usage
To run the model and reproduce the results, follow these steps:

1. **Prepare the Data**: Ensure the dataset is available in the correct directory.
2. **Run the Jupyter Notebook**: Open the provided Jupyter notebook and execute the cells.
3. **Model Evaluation**: Evaluate the model's performance on your data.

```bash
jupyter notebook heart-attack-eda-prediction.ipynb
```

Happy Coding!
