# **Dry Bean Seed Classification**

![image alt](https://github.com/AswathyD31/Capstone_Project_Dry_Bean_Classification_Prediction/blob/69dad0f7896fc4cfacbddef3da54bda7612b21fc/Dry_Bean.jpg)

# **Project Overview**

This project is focused on developing a machine learning pipeline to classify dry bean seeds into distinct categories based on their physical properties. By utilizing the Dry Bean Dataset, the project addresses challenges like data preprocessing, class imbalance, feature scaling, and model evaluation. It provides a robust and efficient solution for agricultural applications and seed classification.

# **Problem Statement**

Dry beans are among the most consumed legumes globally, valued for their nutritional content. However, classifying different types of dry beans can be a complex task. This project leverages machine learning to develop a classification system, assisting agricultural processes and research.

# Objectives

* Clean and preprocess the dataset to ensure accuracy and consistency.
* Address class imbalances for fair and robust model predictions.
* Build and evaluate machine learning models to classify dry bean types effectively.
* Handle unseen data for practical, real-world applications.
* Suggest improvements and future work to enhance the pipeline further.

# **Dataset**

The Dry Bean Dataset contains physical characteristics of seeds, including:

* **Area:**  Total pixel count inside the seed boundary.
* **Perimeter:**  Circumference of the seed.
* **Compactness:**  Shape descriptor (perimeter² / area).
* **Major and Minor Axes:**  Length and width of the seed.
* **Shape Coefficient:**  Ratio of length to width.
* Several other features representing geometric properties.

# **Methodology**

**1. Exploratory Data Analysis (EDA)**

* Analyzed feature distributions using visualizations (e.g., histograms, boxplots).
* Checked for correlations and relationships between features.
*  Identified and handled outliers to ensure clean data.
  
**2. Data Preprocessing**

* **Outlier Removal:**  Used the IQR method to remove extreme values while preserving feature distributions.
* **Feature Scaling:**  Applied normalization/standardization to ensure numerical stability.
* **Class Balancing:**  Addressed imbalanced classes using oversampling techniques like SMOTE and SMOTEEN.
  
**3. Feature Engineering**

* Evaluated feature importance using correlation analysis and model-based importance scores.
* Selected key features to enhance model performance and interpretability.
  
**4. Model Development**

Experimented with multiple machine learning models:

* **'Support Vector Classifier'**
* **'Decision Tree Classifier'**
* **'Random Forest Classifier'**
* **'Gradient Boosting Classifier'**
* **'Gaussian Naive Bayes'**
* **'K-Nearest Neighbors'**
*  Performed hyperparameter tuning using **GridSearchCV.**
  
**5. Evaluation**

* Used metrics like accuracy, precision, recall, F1-score, and ROC-AUC.
* Visualized results with confusion matrices.
  
**6. Handling Unseen Data**

The steps to preprocess and classify unseen data. This ensures that the model generalizes well to real-world inputs:

* Preprocessing: Scaled unseen data using the same scaler fitted during training.
* Load the saved model (best_model.joblib) and use it to make predictions on new unseen data.
* Predictions: Applied the trained model to classify unseen data.

# **Results**

* **Accuracy:** Achieved 95% accuracy on the test dataset.
* **Class Imbalance Handling:** Improved F1-scores for minority classes through oversampling.
* **Key Features Identified:** The most critical features for classification were Area, Compactness, and Major Axis Length.
