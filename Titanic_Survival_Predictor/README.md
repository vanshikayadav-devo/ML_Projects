# 🚢 Titanic Survival Prediction using Random Forest & Extra Trees Classifier

A Machine Learning classification project that predicts whether a passenger survived the Titanic disaster by implementing and comparing two powerful ensemble learning algorithms:

* 🌳 Random Forest Classifier
* 🌲 Extra Trees Classifier (Extremely Randomized Trees)

The project demonstrates the complete machine learning pipeline while comparing the performance of both models on the same dataset.

---

# 📌 Project Overview

The Titanic dataset is one of the most popular beginner-friendly datasets in Machine Learning.

In this project, both **Random Forest** and **Extra Trees** classifiers are trained to predict passenger survival based on features such as:

* Passenger Class
* Age
* Sex
* Fare
* Number of Siblings/Spouses
* Number of Parents/Children
* Embarkation Port

The goal is not only to build accurate classification models but also to understand how two ensemble algorithms differ in terms of training strategy and prediction performance.

---

# 📂 Dataset

**Dataset:** Titanic Survival Dataset

The dataset contains:

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

# 🎯 Problem Statement

Predict whether a passenger survived the Titanic disaster.

### Target Variable

* **0 → Did Not Survive**
* **1 → Survived**

---

# 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn

---

# 📚 Machine Learning Algorithms

## 🌳 Random Forest Classifier

Random Forest is an ensemble learning algorithm that builds multiple Decision Trees using **bootstrap sampling** and **random feature selection**.

The final prediction is obtained using **Majority Voting** across all trees.

### Advantages

* Reduces overfitting
* High prediction accuracy
* Robust to noisy data
* Handles large datasets efficiently

---

## 🌲 Extra Trees Classifier

Extra Trees (Extremely Randomized Trees) is another ensemble learning algorithm similar to Random Forest.

Unlike Random Forest, Extra Trees introduces **additional randomness** by selecting split thresholds randomly instead of searching for the best split.

This often makes the model train faster while maintaining competitive accuracy.

### Advantages

* Faster training
* Less variance
* Reduces overfitting
* High accuracy
* Provides feature importance

---

# 🔄 Project Workflow

1. Import required libraries
2. Load the Titanic dataset
3. Explore the dataset
4. Handle missing values
5. Encode categorical variables
6. Select input features and target variable
7. Split the dataset into training and testing sets
8. Train both models
9. Predict passenger survival
10. Evaluate model performance
11. Compare the results
12. Visualize model performance

---

# 🧹 Data Preprocessing

The following preprocessing steps were performed for both models:

* Filled missing values in **Age** using the mean.
* Filled missing values in **Fare** using the mean.
* Filled missing values in **Embarked** using the mode.
* Removed unnecessary columns:

  * PassengerId
  * Name
  * Ticket
  * Cabin
* Converted categorical variables into numerical values using **Label Encoding**.

---

# 📊 Evaluation Metrics

Both models were evaluated using:

* Accuracy Score
* Precision Score
* Recall Score
* F1 Score
* Confusion Matrix
* Feature Importance

---

# 📈 Model Performance Comparison

| Metric    | 🌳 Random Forest | 🌲 Extra Trees |
| --------- | ---------------: | -------------: |
| Accuracy  |         **0.83** |     **0.8436** |
| Precision |         **0.81** |     **0.8594** |
| Recall    |         **0.78** |     **0.7432** |
| F1 Score  |         **0.79** |     **0.7971** |

---

## Confusion Matrix Comparison

### 🌳 Random Forest

```text
[[90 15]
 [18 56]]
```

### 🌲 Extra Trees

```text
[[96  9]
 [19 55]]
```

---

# 📊 Comparison Summary

| Feature             | Random Forest | Extra Trees  |
| ------------------- | ------------- | ------------ |
| Bootstrap Sampling  | ✅ Yes         | ❌ No         |
| Split Selection     | Best Split    | Random Split |
| Training Speed      | Moderate      | Faster       |
| Randomness          | Moderate      | High         |
| Risk of Overfitting | Low           | Very Low     |
| Feature Importance  | ✅             | ✅            |

---

# 📈 Visualizations

The project includes several visualizations using Matplotlib:

* Confusion Matrix
* Feature Importance Plot
* Actual vs Predicted Comparison
* Evaluation Metrics Bar Chart
* Top Important Features

---

# 💡 Key Learnings

Through this project, the following concepts were explored:

* Supervised Learning
* Classification
* Ensemble Learning
* Decision Trees
* Random Forest
* Extra Trees
* Majority Voting
* Random Feature Selection
* Label Encoding
* Handling Missing Values
* Train-Test Split
* Confusion Matrix
* Feature Importance
* Model Evaluation

---

# 🏆 Conclusion

Both ensemble models achieved strong performance on the Titanic dataset.

* **Random Forest** produced stable and reliable predictions by selecting the best split at each node using bootstrap sampling.
* **Extra Trees** introduced additional randomness by selecting random split thresholds, resulting in **slightly higher accuracy (84.36%)** and **better precision** while also training faster.

This comparison demonstrates how different ensemble strategies can affect model performance, helping in selecting the most suitable algorithm for a classification problem.

---

# 🚀 Future Improvements

* Hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
* Perform K-Fold Cross Validation.
* Compare with Decision Tree, Logistic Regression, SVM, XGBoost, and AdaBoost.
* Apply Feature Engineering.
* Deploy the models using Flask, FastAPI, or Streamlit.

---

# 👩‍💻 Author

**Vanshika Yadav**

B.Tech Information Technology
Dr. B.R. Ambedkar National Institute of Technology (NIT) Jalandhar

---

⭐ If you found this project helpful, consider giving the repository a star!
