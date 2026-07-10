# 🏠 California Housing Price Prediction using K-Nearest Neighbors (KNN) Regression

This project implements the **K-Nearest Neighbors (KNN) Regression** algorithm to predict the **median house value** of houses in California using various housing-related features.

The project demonstrates the complete machine learning workflow including data preprocessing, feature engineering, feature scaling, model training, prediction, and evaluation.

---

## 📌 Project Objective

The objective of this project is to predict the **Median House Value** based on different housing characteristics such as location, income, population, number of rooms, and proximity to the ocean.

---

## 📂 Dataset

### Features

| Feature | Description |
|----------|-------------|
| Longitude | Longitude coordinate of the house |
| Latitude | Latitude coordinate of the house |
| Housing Median Age | Median age of houses in the block |
| Total Rooms | Total number of rooms |
| Total Bedrooms | Total number of bedrooms |
| Population | Population in the block |
| Households | Number of households |
| Median Income | Median income of households |
| Ocean Proximity | Distance of the house from the ocean (Categorical Feature) |

### Target Variable

- **Median House Value**

---

## ⚙️ Project Workflow

- Load the California Housing dataset
- Perform data exploration
- Handle categorical features (`ocean_proximity`)
- Separate features and target variable
- Split the dataset into training and testing sets
- Apply StandardScaler for feature scaling
- Train the KNN Regressor model
- Predict house prices
- Evaluate model performance

---

## 🧠 Machine Learning Algorithm

**K-Nearest Neighbors Regression (KNN Regressor)**

The model predicts the value of a house by averaging the prices of the **K nearest houses** based on the selected distance metric.

---

## 📊 Model Evaluation Metrics

The model is evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn

---

## 📦 Python Libraries

```python
pandas
numpy
scikit-learn
```

---

## 📁 Project Structure

```
California_Housing_Predictor/
│
├── housing.csv
├── stage.py
├── README.md

```

---



## 📈 Results

The trained KNN Regressor predicts California house prices using the nearest neighboring houses and is evaluated using multiple regression metrics to measure prediction performance.
|---40696.10814645626--->mean_absolute_error
|---3742964884.8899155--->mean_squared_error
|---61179.775129448746--->root_mean_squared_error
|---0.7143666636110436--->r2_score
---

## 📚 Learning Outcomes

Through this project, I learned:

- K-Nearest Neighbors Regression
- Regression workflow in Scikit-learn
- Feature scaling using StandardScaler
- Encoding categorical variables
- Data preprocessing
- Train-Test Split
- Regression evaluation metrics
- Effect of different distance metrics in KNN

---

## 🔮 Future Improvements

- Hyperparameter tuning using GridSearchCV
- K-Fold Cross Validation
- Compare different values of K
- Compare different distance metrics
- Feature engineering
- Data visualization
- Compare KNN with Linear Regression, Decision Tree, and Random Forest Regression

---

## ⭐ If you found this project helpful, consider giving it a star!