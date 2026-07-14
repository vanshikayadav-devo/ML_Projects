# 🏠 California Housing Price Prediction using K-Nearest Neighbors (KNN) and Random Forest Regression

This project implements two popular **Machine Learning Regression algorithms**—**K-Nearest Neighbors (KNN) Regressor** and **Random Forest Regressor**—to predict the **Median House Value** of houses in California.

The project demonstrates the complete machine learning workflow, including data preprocessing, feature engineering, feature scaling (for KNN), model training, prediction, evaluation, and performance comparison between the two algorithms.

---

# 📌 Project Objective

The objective of this project is to predict the **Median House Value** based on various housing characteristics such as location, median income, population, number of rooms, households, and proximity to the ocean.

The project also compares the performance of **KNN Regression** and **Random Forest Regression** on the same dataset.

---

# 📂 Dataset

## California Housing Dataset

### Features

| Feature            | Description                                                |
| ------------------ | ---------------------------------------------------------- |
| Longitude          | Longitude coordinate of the house                          |
| Latitude           | Latitude coordinate of the house                           |
| Housing Median Age | Median age of houses in the block                          |
| Total Rooms        | Total number of rooms                                      |
| Total Bedrooms     | Total number of bedrooms                                   |
| Population         | Population in the block                                    |
| Households         | Number of households                                       |
| Median Income      | Median income of households                                |
| Ocean Proximity    | Distance of the house from the ocean (Categorical Feature) |

### Target Variable

* **Median House Value**

---

# ⚙️ Project Workflow

* Load the California Housing dataset
* Perform data exploration
* Check missing values
* Handle missing values
* Encode categorical features (`ocean_proximity`)
* Select input features and target variable
* Split the dataset into training and testing sets
* Apply **StandardScaler** (KNN only)
* Train the KNN Regressor
* Train the Random Forest Regressor
* Predict house prices
* Evaluate both models
* Compare model performance

---

# 🧠 Machine Learning Algorithms

## 1️⃣ K-Nearest Neighbors (KNN) Regressor

KNN predicts the price of a house by finding the **K nearest houses** based on a chosen distance metric and averaging their prices.

### Characteristics

* Distance-based algorithm
* Lazy Learning Algorithm
* Requires Feature Scaling
* Simple and easy to understand

---

## 2️⃣ Random Forest Regressor

Random Forest is an **Ensemble Learning Algorithm** that builds multiple Decision Trees using random subsets of data and features.

Each tree predicts a house price, and the final prediction is obtained by averaging the predictions of all trees.

### Characteristics

* Ensemble Learning Algorithm
* High prediction accuracy
* Less prone to overfitting
* Handles complex relationships effectively
* Does not require feature scaling

---

# 🧹 Data Preprocessing

The following preprocessing steps were performed:

* Filled missing values in **total_bedrooms** using the column mean.
* Converted the categorical feature **ocean_proximity** into numerical values using **LabelEncoder**.
* Applied **StandardScaler** before training the KNN model.
* Selected relevant features for training.

---

# 📊 Model Evaluation Metrics

Both models were evaluated using:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score

---

# 📈 Model Performance

| Metric                         |        KNN Regressor | Random Forest Regressor |
| ------------------------------ | -------------------: | ----------------------: |
| Mean Absolute Error (MAE)      |        **40,696.11** |           **32,193.21** |
| Mean Squared Error (MSE)       | **3,742,964,884.89** |    **2,524,664,227.62** |
| Root Mean Squared Error (RMSE) |        **61,179.78** |           **50,246.04** |
| R² Score                       |           **0.7144** |              **0.8073** |

---

# 📊 Performance Comparison

### KNN Regressor

**Advantages**

* Simple and easy to implement
* Works well on smaller datasets
* No training phase

**Disadvantages**

* Requires feature scaling
* Slower predictions on large datasets
* Sensitive to the choice of **K**
* Performance decreases with high-dimensional data

---

### Random Forest Regressor

**Advantages**

* Higher prediction accuracy
* Handles non-linear relationships effectively
* Less prone to overfitting
* Robust against noise and outliers
* Does not require feature scaling

**Disadvantages**

* Longer training time
* Higher memory usage
* More complex than KNN

---

# 🏆 Conclusion

Based on the evaluation metrics, the **Random Forest Regressor outperformed the KNN Regressor** on the California Housing dataset.

* Lower prediction errors (MAE, MSE, RMSE)
* Higher R² Score
* Better generalization on unseen data

This demonstrates the effectiveness of ensemble learning techniques for complex regression problems.

---

# 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn

---

# 📦 Python Libraries

```python
pandas
numpy
matplotlib
scikit-learn
```

---

# 📁 Project Structure

```text
California_Housing_Predictor/
│
├── housing.csv
├── knn_regression.py
├── random_forest_regression.py
├── README.md
└── requirements.txt
```

---

# 📚 Learning Outcomes

Through this project, I learned:

* Regression using Scikit-learn
* K-Nearest Neighbors Regression
* Random Forest Regression
* Ensemble Learning
* Feature Scaling using StandardScaler
* Label Encoding
* Handling Missing Values
* Data Preprocessing
* Train-Test Split
* Regression Evaluation Metrics (MAE, MSE, RMSE, R² Score)
* Comparing multiple machine learning algorithms

---

# 🔮 Future Improvements

* Hyperparameter tuning using GridSearchCV and RandomizedSearchCV
* K-Fold Cross Validation
* Feature Importance Visualization
* Actual vs Predicted Values graph
* Compare different values of **K** in KNN
* Tune Random Forest hyperparameters (`n_estimators`, `max_depth`, etc.)
* Compare with Linear Regression and Decision Tree Regression
* Deploy the best-performing model using Flask, FastAPI, or Streamlit

---

# 👩‍💻 Author

**Vanshika Yadav**

B.Tech in Information Technology
Dr. B.R. Ambedkar National Institute of Technology (NIT) Jalandhar

---

⭐ If you found this project helpful, consider giving the repository a **Star**!
