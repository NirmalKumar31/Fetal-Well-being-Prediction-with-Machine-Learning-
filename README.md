# 🍼 Fetal Health Classification using XGBoost and BorutaSHAP 🚀

## 👨‍💻 Authors : **Nirmalkumaar T K**, Pratosh Karthikeyan, Jeevanandh R, Pranav Manikandan Sundaresan, Pranav A
📅 Date: August 2023 - October 2023  
📧 Contact: thirupallikrishnan.n@northeastern.edu  
🔗 [LinkedIn](https://www.linkedin.com/in/nirmalkumartk/) | [GitHub](https://github.com/NirmalKumar31)

## 🖥️ Tech Stack
- 🐍 Python
- 📊 Pandas, NumPy
- 📈 Matplotlib, Seaborn
- 🤖 XGBoost, Scikit-Learn
- 🔬 BorutaSHAP, SHAP, Machine Learning
- 📦 imbalanced-learn

## 📌 Overview
This repository contains the implementation of a **machine learning model** for fetal health classification using **cardiotocography (CTG) data**. The goal is to enhance **early risk detection** in fetal health assessment by leveraging the **XGBoost** algorithm and **BorutaSHAP** for feature selection.

## 🌟 Key Contributions
✅ Developed an **XGBoost-based classification model** for fetal health prediction using CTG data.
✅ Improved recall for **minority classes by 72%** using **BorutaSHAP** feature selection.
✅ Addressed class imbalance with **SMOTE (Synthetic Minority Over-sampling Technique)** to enhance diagnostic accuracy.
✅ Identified **key predictors** through **Exploratory Data Analysis (EDA)** and statistical correlation analysis.
✅ Achieved an **accuracy of 94%** on the test dataset, demonstrating significant improvement over baseline models.

## 📊 Dataset
The dataset used for this research is the **Cardiotocography (CTG) dataset** from the **UCI Machine Learning Repository**:
🔗 [UCI CTG Dataset](https://archive.ics.uci.edu/dataset/193/cardiotocography)

### 📋 Data Description
The dataset contains **2126 records** categorized into three fetal health classes:
- 🟢 **Normal (Class 1)**
- 🟡 **Suspect (Class 2)**
- 🔴 **Pathological (Class 3)**

### 🔢 Feature List
The dataset consists of **22 features**, including:
- **Baseline Value**: Fetal heart rate baseline (beats per minute)
- **Accelerations**: Number of accelerations per second
- **Fetal Movement**: Number of fetal movements per second
- **Uterine Contractions**: Number of uterine contractions per second
- **Light Decelerations**: Number of light decelerations per second
- **Severe Decelerations**: Number of severe decelerations per second
- **Prolonged Decelerations**: Number of prolonged decelerations per second
- **Abnormal Short-Term Variability**: Percentage of time with abnormal short-term variability
- **Mean Short-Term Variability**: Mean value of short-term variability
- **Percentage of Time with Abnormal Long-Term Variability**
- **Mean Long-Term Variability**: Mean value of long-term variability
- **Histogram Width**: Width of FHR histogram
- **Histogram Min**: Minimum (low frequency) of FHR histogram
- **Histogram Max**: Maximum (high frequency) of FHR histogram
- **Histogram Number of Peaks**: Number of histogram peaks
- **Histogram Number of Zeroes**: Number of histogram zeros
- **Histogram Mode**
- **Histogram Mean**
- **Histogram Variance**

## 🛠️ Methodology
### 1️⃣ Exploratory Data Analysis (EDA)
📊 Visualized class distribution (showing class imbalance)
📌 Identified key correlated features using a **heatmap**
📈 Analyzed the impact of individual features on fetal health classification

### 2️⃣ Data Preprocessing
🔄 Handled class imbalance using **SMOTE**
📏 Standardized numerical features for better model performance
🔬 Selected most relevant features using **BorutaSHAP**

### 3️⃣ Model Selection
Several machine learning models were evaluated:
- 🔹 **Logistic Regression**
- 🔸 **Support Vector Classifier (SVC)**
- 🔹 **Random Forest Classifier**
- 🔸 **Gaussian Naive Bayes**
- ⭐ **XGBoost Classifier** (Final Model)

### 4️⃣ Feature Selection with BorutaSHAP
To improve the model's interpretability and performance, we used **BorutaSHAP**, which combines:
- **Boruta Algorithm** (identifies relevant features by comparing against shuffled versions)
- **SHAP Values** (quantifies the importance of each feature in the model)

### 5️⃣ Model Performance
| 🏆 Model                        | 🎯 Train Accuracy | 🎯 Test Accuracy |
|---------------------------------|------------------|-----------------|
| 🌲 Random Forest Classifier     | 99.91%          | 89.63%         |
| 📊 Logistic Regression          | 78.75%          | 76.74%         |
| 🤖 Support Vector Classifier    | 67.46%          | 66.18%         |
| 🧪 Gaussian Naive Bayes        | 70.45%          | 70.09%         |
| ⭐ **XGBoost Classifier**      | **99.91%**      | **91.33%**     |
| 🔥 **XGBoost + BorutaSHAP**    | **99.92%**      | **94.00%**     |

### 📌 Classification Report (XGBoost + BorutaSHAP)
- ✅ **Precision:** 95% (Class 0), 89% (Class 1), 86% (Class 2)
- 🎯 **Recall:** 98% (Class 0), 72% (Class 1), 92% (Class 2)
- 📈 **F1-Score:** 96% (Class 0), 79% (Class 1), 89% (Class 2)
- 🏅 **Overall Accuracy:** 94%

## 🔮 Future Work
🚑 **Integration into real-world clinical settings** for fetal monitoring
🧠 **Enhancing interpretability** using explainable AI techniques
📱 **Developing a lightweight mobile application** for remote monitoring

🔗 **[Full Research Paper - Machine Learning in Fetal Well-Being Prediction](https://www.ijaresm.com/uploaded_files/document_file/Pratosh_KarthikeyangS61.pdf)**
🔗 **[Code File](https://github.com/NirmalKumar31/Fetal-Well-being-Prediction-with-Machine-Learning-/blob/4f4725852ce440fd1977307824df0fe8ea08bfdb/Fetal_Classification_Final.ipynb)**

## 📚 References

📖 Abdel-Raouf, Heba; Elhoseny, Mohamed; Taha, Tarek (2020). "Fetal Health Classification Using XGBoost Algorithm."
📖 Aygun, Ibrahim; Olmez, Tugba; et al. (2019). "Fetal Health Classification Using Machine Learning Algorithms: A Comparative Study."

---
🎯 This project aims to contribute to **reducing child mortality (SDG Goal 3)** by leveraging machine learning for **early risk detection in fetal health**. 🚼💡

