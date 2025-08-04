# 🧠 Text Classification using Multinomial Naive Bayes

This project demonstrates a complete pipeline for building and evaluating a text classification model using **Multinomial Naive Bayes** and **Bag-of-Words (CountVectorizer)**. It includes data preprocessing, model training, prediction, and performance evaluation using key metrics like accuracy, confusion matrix, and ROC-AUC.

---

## 📂 Project Structure

```

├── data/                 # (Optional) Contains the dataset
├── notebooks/            # Jupyter notebooks or scripts
├── models/               # Saved model files (optional)
├── README.md             # Project documentation
└── main.py / notebook.ipynb   # Main implementation

````

---

## 🚀 Getting Started

### 📦 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
````

Typical libraries used:

* `scikit-learn`
* `pandas`
* `numpy`
* `matplotlib`

---

## 📊 Dataset

The dataset used contains labeled text data, where each sample is classified into one of two categories (e.g., spam vs. ham, positive vs. negative, etc.). Features are extracted using a **Bag-of-Words model**, and English stop words are removed for better generalization.

---

## 🧪 Model and Evaluation

### ✅ Model Used

* **Multinomial Naive Bayes** (`sklearn.naive_bayes.MultinomialNB`)
* **Text Vectorization** via `CountVectorizer(stop_words='english')`

### ⚙️ Key Code Steps

```python
# Vectorization
vect = CountVectorizer(stop_words='english')
vect.fit(X_train)
X_train_transformed = vect.transform(X_train)
X_test_transformed = vect.transform(X_test)

# Model Training
mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_train_transformed, y_train)

# Prediction
y_pred_class = mnb.predict(X_test_transformed)
y_pred_proba = mnb.predict_proba(X_test_transformed)
```

### 📈 Evaluation Results

| Metric               | Value                                  |
| -------------------- | -------------------------------------- |
| **Accuracy**         | 98.78%                                 |
| **ROC-AUC Score**    | 0.9921                                 |
| **Confusion Matrix** | Very low false positives and negatives |

---

## 📉 ROC Curve

The ROC Curve shows the trade-off between sensitivity and specificity. An AUC score of **0.9921** indicates **near-perfect classification**.

---

## 🧠 Conclusion

The Multinomial Naive Bayes classifier performed exceptionally well, achieving high accuracy and AUC. The approach is efficient, interpretable, and well-suited for discrete, text-based features. This serves as a strong baseline for more advanced NLP models.

---

## 🔮 Future Work

* Use **TF-IDF Vectorizer** instead of raw counts
* Apply **Grid Search** for hyperparameter tuning
* Try **ensemble methods** or **deep learning approaches**
* Extend to multi-class classification

---

## 📌 License

This project is open-source and available under the [MIT License].

---

## 🤝 Contributing

Pull requests and suggestions are welcome. For major changes, please open an issue first to discuss your ideas.

---

## 🙌 Acknowledgments

Thanks to the creators of **scikit-learn** and the open datasets that made this project possible.
