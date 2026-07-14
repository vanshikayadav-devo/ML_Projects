# 🚢 Titanic Survival Prediction using Random Forest Classifier

A Machine Learning classification project that predicts whether a passenger survived the Titanic disaster using the **Random Forest Classifier** algorithm from Scikit-learn.

---

## 📌 Project Overview

The Titanic disaster is one of the most well-known datasets in Machine Learning. In this project, a **Random Forest Classifier** is trained to predict passenger survival based on various features such as passenger class, age, sex, fare, and embarkation point.

This project demonstrates the complete machine learning workflow, including:

* Data loading
* Data preprocessing
* Handling missing values
* Encoding categorical variables
* Feature selection
* Train-test split
* Model training
* Model evaluation

---

## 📂 Dataset

**Dataset Used:** Titanic Survival Dataset

The dataset contains information about Titanic passengers, including:

* PassengerId
* Survived (Target Variable)
* Pclass
* Name
* Sex
* Age
* SibSp
* Parch
* Ticket
* Fare
* Cabin
* Embarked

---

## 🎯 Problem Statement

Predict whether a passenger survived the Titanic disaster.

### Target Variable

* **0** → Did Not Survive
* **1** → Survived

---

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn

---

## 📚 Machine Learning Algorithm

### Random Forest Classifier

Random Forest is a **Supervised Ensemble Learning Algorithm** used for classification and regression problems.

Instead of relying on a single Decision Tree, Random Forest creates multiple Decision Trees using random subsets of the training data and random feature selection. The final prediction is made using **Majority Voting** among all the trees.

### Why Random Forest?

* Reduces overfitting compared to a single Decision Tree
* Provides higher prediction accuracy
* Handles large datasets efficiently
* Robust against noise and outliers
* Works well with both numerical and categorical features

---

## 🔄 Project Workflow

1. Import required libraries
2. Load the Titanic dataset
3. Explore the dataset
4. Check for missing values
5. Handle missing values
6. Encode categorical features using `LabelEncoder`
7. Select input features and target variable
8. Split the dataset into training and testing sets
9. Train the Random Forest Classifier
10. Predict survival on the test dataset
11. Evaluate the model using multiple performance metrics

---

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

* Missing values in the **Age** column were filled using the column mean.
* Missing values in the **Embarked** column were filled using the most frequent value (Mode).
* Categorical features were converted into numerical values using **Label Encoding**.
* Relevant features were selected for model training.

---

## 📊 Model Evaluation Metrics

The trained model was evaluated using:

* Accuracy Score
* Precision Score
* Recall Score
* F1 Score
* Confusion Matrix
* Classification Report


## 📈 Sample Output

```text
Accuracy : 0.83

Precision : 0.81

Recall : 0.78

F1 Score : 0.79

Confusion Matrix

[[90 15]
 [18 56]]
```


---

## 💡 Concepts Covered

* Supervised Learning
* Classification
* Random Forest
* Ensemble Learning
* Bootstrap Sampling
* Majority Voting
* Label Encoding
* Handling Missing Values
* Train-Test Split
* Model Evaluation

---

## 🚀 Future Improvements

* Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
* Feature Importance visualization
* Cross-validation
* Compare Random Forest with Decision Tree, Logistic Regression, and SVM
* Deploy the model using Flask, FastAPI, or Streamlit

---

## 👩‍💻 Author

**Vanshika Yadav**

B.Tech Information Technology
Dr. B.R. Ambedkar National Institute of Technology (NIT) Jalandhar

---

⭐ If you found this project helpful, consider giving the repository a star!
