# 📧 SMS Spam Detection using Multinomial Naive Bayes



# 📌 Project Overview

This project implements an **SMS Spam Detection System** using the **Multinomial Naive Bayes** algorithm. The objective is to classify SMS messages into two categories:

* 📩 **Ham (Legitimate Message)**
* 🚫 **Spam (Unwanted Message)**

Since machine learning algorithms cannot understand raw text, the SMS messages are first converted into numerical feature vectors using **CountVectorizer**. These vectors are then used to train a **Multinomial Naive Bayes** classifier.

This project demonstrates a complete **Natural Language Processing (NLP)** pipeline, from loading the dataset to evaluating the model's performance.

---

# 🎯 Objectives

* Build an SMS spam classification model.
* Learn text preprocessing techniques used in NLP.
* Convert text into numerical vectors using **CountVectorizer**.
* Train a **Multinomial Naive Bayes** classifier.
* Evaluate the model using multiple performance metrics.
* Understand the complete workflow of text classification.

---

# 📂 Dataset

**Dataset Name:** SMS Spam Dataset

The dataset consists of SMS messages labeled as:

* **Ham** → Normal SMS
* **Spam** → Promotional or fraudulent SMS

### Dataset Columns

| Column    | Description                                      |
| --------- | ------------------------------------------------ |
| id        | Unique identifier (removed during preprocessing) |
| spamORham | Target column containing the message label       |
| Message   | SMS text message                                 |

---

# 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn

---

# 📚 Machine Learning Concepts Used

* Label Encoding
* Train-Test Split
* Natural Language Processing (NLP)
* CountVectorizer
* Document-Term Matrix
* Vocabulary Creation
* Multinomial Naive Bayes
* Model Evaluation

---

# ⚙️ Project Workflow

```text
Load Dataset
      │
      ▼
Data Exploration
      │
      ▼
Remove Unnecessary Columns
      │
      ▼
Label Encoding
      │
      ▼
Train-Test Split
      │
      ▼
CountVectorizer
      │
      ▼
Convert Text into Numerical Features
      │
      ▼
Train Multinomial Naive Bayes Model
      │
      ▼
Prediction
      │
      ▼
Performance Evaluation
```

---

# 🧹 Data Preprocessing

The following preprocessing steps were performed:

* Loaded the dataset using Pandas.
* Checked for missing values.
* Removed the unnecessary **id** column.
* Encoded the target labels using **LabelEncoder**:

  * Ham → 0
  * Spam → 1
* Split the dataset into training and testing sets.

---

# 📝 Text Vectorization

The SMS messages were converted into numerical vectors using **CountVectorizer**.

CountVectorizer performs the following tasks:

* Creates a vocabulary from the training dataset.
* Counts the occurrence of each word.
* Converts every SMS into a numerical vector.
* Uses the same vocabulary to transform the test data, preventing data leakage.

---

# 🤖 Model Used

## Multinomial Naive Bayes

Multinomial Naive Bayes is a probabilistic machine learning algorithm designed for text classification tasks.

It calculates the probability of each class based on the frequency of words appearing in the messages and predicts whether a message is **Spam** or **Ham**.

---

# 📊 Model Performance

| Metric        | Score      |
| ------------- | ---------- |
| **Accuracy**  | **98.49%** |
| **Precision** | **97.02%** |
| **Recall**    | **91.06%** |
| **F1-Score**  | **93.95%** |

---

# 📈 Confusion Matrix

```text
                Predicted

              Ham     Spam

Actual Ham   1209       5

Actual Spam    16     163
```

### Interpretation

* ✅ **1209** Ham messages correctly classified.
* ✅ **163** Spam messages correctly detected.
* ❌ **5** Ham messages incorrectly classified as Spam.
* ❌ **16** Spam messages were missed and classified as Ham.

The model demonstrates excellent overall performance with high precision and accuracy while maintaining a strong recall.

---

# 💻 Sample Console Output

```text
accuracy: 0.9849246231155779

precision: 0.9702380952380952

recall: 0.9106145251396648

f1-score: 0.9394812680115274

confusion-matrix:
[[1209    5]
 [  16  163]]
```

---

# 📁 Project Structure

```text
Email_spam_detector/
│
├── SMS_spam_dataset.csv
├── stage.py
├── README.md

```

---

# 🚀 Future Improvements

* Apply text preprocessing techniques such as:

  * Tokenization
  * Stop Word Removal
  * Stemming
  * Lemmatization

* Compare CountVectorizer with TF-IDF Vectorizer.

* Experiment with different classifiers:

  * Logistic Regression
  * Support Vector Machine (SVM)
  * Random Forest
  * Decision Tree

* Build a web application using Flask or Streamlit.

* Deploy the model for real-time spam prediction.

---

# 🎓 Key Learnings

Through this project, I learned:

* Natural Language Processing (NLP) workflow
* Text preprocessing fundamentals
* Label Encoding
* CountVectorizer and vocabulary creation
* Document-Term Matrix (DTM)
* Multinomial Naive Bayes
* Text classification
* Model evaluation using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix
* End-to-end implementation of an NLP classification project using Scikit-learn

---

# 📜 Conclusion

This project demonstrates a complete implementation of an **SMS Spam Detection System** using **Multinomial Naive Bayes**. By combining **CountVectorizer** for text feature extraction with a probabilistic classifier, the model achieves an **accuracy of 98.49%**, making it highly effective at distinguishing between spam and legitimate SMS messages.

This project serves as a strong introduction to **Natural Language Processing (NLP)** and text classification using machine learning.

---

## 👩‍💻 Author

**Vanshika Yadav**

*B.Tech Information Technology*
*Dr. B. R. Ambedkar National Institute of Technology, Jalandhar*

Learning AI/ML by building practical projects and documenting the complete journey on GitHub.
