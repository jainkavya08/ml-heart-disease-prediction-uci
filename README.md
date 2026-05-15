# 🫀 Heart Disease Prediction - ML Pipeline

##  VAUTECH IT SOLUTIONS - TASK 2

| | |
|---|---|
| **Intern** | Kavya jain |
| **Intern ID** | VT26ML005 |
| **Domain** | Machine Learning |
| **Company** | VAUTECH IT SOLUTIONS |
| **Mentor** | Vishal Rajbhar |

---

## 📂 Dataset

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **Instances:** 303 (297 after removing missing values)
- **Features:** 13 + 1 target column
- **Task Type:** Binary Classification
- **Target:** `0` = No Heart Disease / `1` = Heart Disease Present

---

## 🔁 Project Pipeline

### Step 1: Load Dataset
- Loaded Cleveland Heart Disease dataset from UCI ML Repository
- Assigned correct column names from documentation

### Step 2: Explore Data
- Checked shape, dtypes and info
- Identified missing values hidden as `"?"`

### Step 3: Handle Missing Values
- Replaced `"?"` with `NaN`
- Found 4 missing values in `ca` and 2 in `thal`
- Dropped 6 rows with missing values

### Step 4: Simplify Target Variable
- Original target had values 0 to 4
- Converted to binary: 0 = No Disease, 1 = Disease Present

### Step 5: Convert Data Types
- Converted `ca` and `thal` columns from object to numeric type

### Step 6: Exploratory Data Analysis (EDA)
- Visualized feature distributions using histograms
- Analyzed feature vs target relationships using boxplots
- Identified correlations using heatmap

### Step 7: Feature Scaling
- Separated features (X) and target (y)
- Applied StandardScaler to all feature columns
- Target column left untouched

### Step 8: Train-Test Split
- Split data into 80% training and 20% testing
- Training size: 237 rows
- Testing size: 60 rows

### Step 9: Model Building
- Built Logistic Regression classification model
- Trained on X_train and y_train

### Step 10: Model Evaluation
- Predicted on X_test
- Evaluated using accuracy score and confusion matrix

---

## 📊 Results

| Metric | Score |
|--------|-------|
| Accuracy | 86.67% |

> Note: Research papers on this dataset report accuracy in the range of 80% - 90% — our result is well aligned with published research!

---

## 🛠️ Tools & Libraries

| Tool | Purpose |
|------|---------|
| Python | Programming language |
| pandas | Data loading and manipulation |
| numpy | Numerical operations |
| matplotlib | Data visualization |
| seaborn | Advanced visualization |
| scikit-learn | Scaling, splitting, model building |

---

## 📁 Files

| File | Description |
|---|---|
| `Heart_Disease_ML_Internship.ipynb` | Main notebook with complete pipeline |
| `processed.cleveland.data` | Raw dataset file |
| `heart-disease.names` | Dataset documentation from UCI |
| `notes.md` | Detailed step-by-step project notes |
| `requirements.txt` | Required Python libraries |
| `.gitignore` | Git ignore rules |

---

## 🚀 How to Run

1. Clone the repository
```bash
git clone https://github.com/yourusername/heart-disease-prediction.git
```

2. Install required libraries
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

3. Open the notebook
```bash
jupyter notebook Heart_Disease_ML_Internship.ipynb
```

---

## 📜 Citation

Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). Heart Disease [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X

---

## 👨‍💻 Author

Kavya jain
