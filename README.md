# 🇪🇬 Arabic Sentiment Analysis – Egyptian Dialect (ML + Deep Learning)

![Arabic Sentiment Analysis](https://raw.githubusercontent.com/username/repo/main/images/تحليل%20المشاعر%20باللهجة%20المصرية.png)

This project presents a comprehensive Arabic sentiment analysis pipeline for classifying **user-generated reviews written in Egyptian dialect** into three sentiment categories: **positive**, **neutral**, and **negative**. It combines both **traditional machine learning techniques** and **deep learning (DNN)** to explore accuracy, performance, and production readiness.


---

## 🔍 Project Objective

The goal is to evaluate the performance of multiple models for Arabic sentiment analysis, compare their effectiveness, and ultimately develop a robust model ready for real-world deployment — particularly for social platforms or customer feedback systems using the Arabic language.

---

## 📊 Model Comparison – Machine Learning (TF-IDF based)

To build a strong baseline, a series of classical ML models were trained on the same preprocessed dataset using **TF-IDF vectorization**. All models were evaluated on the same train-test split for fair comparison.

| Model                   | Accuracy Score |
|------------------------|----------------|
| Logistic Regression    | 82.33%         |
| Naive Bayes            | 81.45%         |
| Support Vector Machine | 82.08%         |
| Random Forest          | 81.53%         |
| LightGBM               | 82.17%         |
| CatBoost               | 81.81%         |

### 🔎 Key Insights

- **Logistic Regression** provided the highest accuracy (82.33%) and is an excellent baseline model for Arabic text.
- **LightGBM** and **SVM** performed competitively, showing the value of both linear and ensemble approaches.
- **Naive Bayes** remains a lightweight and efficient choice, ideal for quick iterations or real-time applications with limited compute.
- **CatBoost** and **Random Forest** demonstrated stable performance and strong generalization with minimal tuning.

---

## 🧠 Deep Learning Model (DNN)

After validating classical ML results, a **Deep Neural Network (DNN)** was implemented using TensorFlow/Keras to push performance further and ensure compatibility with modern production environments.

### 📌 DNN Architecture Overview

- Input Layer: Embedded Arabic tokens with 64-dimensional vectors
- Core Layers: GlobalAveragePooling and ReLU-based Dense layers
- Regularization: Dropout layer to prevent overfitting
- Output Layer: 3-unit softmax layer for multi-class prediction (positive, neutral, negative)

### 🛠️ Training Configuration

- Loss Function: Sparse Categorical Crossentropy
- Optimizer: Adam
- Epochs: 100 (with Early Stopping)
- Batch Size: 1024
- Validation Monitoring: Based on validation accuracy

---

## 📈 Final DNN Performance

The DNN model significantly outperformed traditional ML models, achieving **high accuracy and balanced performance across all sentiment classes**.

### 📋 Classification Report

| Class     | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Positive  | 0.92      | 0.95   | 0.93     | 8,558   |
| Neutral   | 0.99      | 0.89   | 0.93     | 3,369   |
| Negative  | 0.91      | 0.91   | 0.91     | 5,894   |

- **Accuracy**: 93.0%
- **Macro Average F1-Score**: 93%
- **Weighted Average F1-Score**: 93%

### 📉 Training Summary

- **Final Training Accuracy**: 94.47%
- **Final Validation Accuracy**: 92.30%
- **Training Loss**: 0.1585
- **Validation Loss**: 0.2425

These results demonstrate the DNN’s ability to learn semantic differences between sentiment labels, generalize well to unseen data, and maintain high reliability for production tasks.

---

## 🚀 Model Deployment Readiness

To integrate the model into real-world applications, the following components were prepared:

- ✅ **Model File (`sentiment_model.h5`)** – the trained deep learning model ready for inference.
- ✅ **Tokenizer (`tokenizer.pickle`)** – for replicating preprocessing in production pipelines.

These artifacts are fully portable and can be used in Python-based APIs (Flask, FastAPI) or converted for use in browser/mobile environments using TensorFlow.js or TFLite.

---

## 📚 Conclusion

This project validates that **Arabic sentiment analysis**, especially in the Egyptian dialect, is highly feasible using both traditional and deep learning approaches. While classical models offer simplicity and quick inference, the **DNN delivers superior accuracy and flexibility**, making it ideal for scalable applications.

The next step could involve exploring **transformer-based models (e.g., AraBERT)** or building a full-stack system with **real-time sentiment tagging** for customer service, social media monitoring, or e-commerce review engines.

---

## 👨‍💻 Author

**Ibrahim Abdelsattar**  
AI & Data Science Enthusiast | Specialized in NLP  
[LinkedIn Profile](https://www.linkedin.com/in/ibrahim-abdelsattar/)
