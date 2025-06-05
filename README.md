# 🧠 Breast Cancer Detection using SVM (Support Vector Machines)

This project performs binary classification on the Breast Cancer dataset using Support Vector Machines with both **Linear** and **RBF** kernels. It includes preprocessing, dimensionality reduction for visualization, hyperparameter tuning, and performance evaluation using cross-validation.

---

## 📂 Dataset

- **Source**: UCI Breast Cancer Wisconsin dataset
- **Target Variable**: `diagnosis` — Malignant (M) or Benign (B)
- **Features**: 30 numerical features related to tumor characteristics
- **Binary Labels**:
  - Malignant → `1`
  - Benign → `0`

---

## 🧪 Steps Performed

### 1. Data Loading & Preprocessing
- Dropped unnecessary columns like `id` and missing values (`Unnamed: 32`).
- Encoded the `diagnosis` column (Malignant → 1, Benign → 0).
- Scaled features using `StandardScaler`.
- Split data into train and test sets (70/30).

### 2. SVM Model Training
- Trained two SVM models:
  - **Linear kernel** (good for linearly separable data)
  - **RBF kernel** (handles non-linearity)
- Evaluated both on test set using accuracy.

### 3. Dimensionality Reduction for Visualization
- Applied **PCA (2D)** to project high-dimensional data into two dimensions.
- Visualized the **decision boundary** of SVM with RBF kernel.

### 4. Hyperparameter Tuning
- Used `GridSearchCV` to find the best combination of:
  - **C**: Regularization parameter
  - **gamma**: Kernel coefficient for RBF
- Tuned model achieved better performance on cross-validation.

### 5. Cross-validation
- Evaluated the final model using 5-fold cross-validation.
- Reported individual fold scores and overall average accuracy.

---

## 📊 Results

| Model         | Accuracy (Test Set) | Cross-Validation (Mean Accuracy) |
|---------------|---------------------|----------------------------------|
| Linear SVM    | ~0.96               | ~0.96                            |
| RBF SVM       | ~0.98               | ~0.98                            |
| Tuned RBF SVM | ↑ Improved          | ↑ Improved                       |

> 💡 RBF kernel performed better than linear, especially after tuning hyperparameters.

---

## 📈 Visualization

- PCA was used to project features to 2D for plotting.
- The RBF SVM decision boundary was visualized showing clear separation between classes.

---

## 🛠️ Tools & Libraries

- Python
- pandas, numpy
- scikit-learn
- matplotlib
- seaborn (optional)

---
