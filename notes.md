# 🫀 Detailed Notes — Heart Disease ML Project

## 🏢 VAUTECH IT SOLUTIONS — TASK 1

| | |
|---|---|
| **Intern** | Kavya jain |
| **Intern ID** | VT26ML005 |
| **Domain** | Machine Learning |
| **Company** | VAUTECH IT SOLUTIONS |
| **Mentor** | Vishal Rajbhar |

---

## 📌 Project Overview

This project performs complete **Dataset Selection, Exploratory Data Analysis (EDA), Data Preprocessing, and Classification Model Building** on the UCI Cleveland Heart Disease Dataset. The goal is to predict whether a patient has heart disease or not, based on 13 medical features.

---

## 📂 Dataset Information

| Detail | Info |
|---|---|
| Source | UCI Machine Learning Repository |
| URL | https://archive.ics.uci.edu/dataset/45/heart+disease |
| Instances | 303 (297 after cleaning) |
| Features | 13 + 1 target column |
| Task Type | Binary Classification |
| Target | `0` = No Heart Disease / `1` = Heart Disease Present |
| Missing Values | Yes — denoted by `"?"` |

### Column Description

| Column | Type | Description | Missing? |
|---|---|---|---|
| age | Integer | Age of patient in years | No |
| sex | Binary | Gender (1 = Male, 0 = Female) | No |
| cp | Categorical | Chest pain type (1=typical angina, 2=atypical angina, 3=non-anginal, 4=asymptomatic) | No |
| trestbps | Integer | Resting blood pressure in mm Hg | No |
| chol | Integer | Serum cholesterol in mg/dl | No |
| fbs | Binary | Fasting blood sugar > 120 mg/dl (1=True, 0=False) | No |
| restecg | Categorical | Resting ECG results (0=normal, 1=ST-T abnormality, 2=left ventricular hypertrophy) | No |
| thalach | Integer | Maximum heart rate achieved | No |
| exang | Binary | Exercise induced angina (1=Yes, 0=No) | No |
| oldpeak | Float | ST depression induced by exercise relative to rest | No |
| slope | Categorical | Slope of peak exercise ST segment (1=upsloping, 2=flat, 3=downsloping) | No |
| ca | Integer | Number of major vessels colored by fluoroscopy (0-3) | **Yes (4)** |
| thal | Categorical | Thalassemia (3=normal, 6=fixed defect, 7=reversable defect) | **Yes (2)** |
| target | Integer (Target) | Diagnosis of heart disease (0=No Disease, 1-4=Disease) | No |

---

## 🔬 Step-by-Step Explanation

---

### ✅ Step 1 — Import Libraries

```python
import pandas as pd
import matplotlib.pyplot as plt
```

**What & Why:**
- `pandas` — the primary library for loading, manipulating, and analyzing tabular data (DataFrames).
- `matplotlib.pyplot` — used for creating visualizations like charts and plots.
- Other libraries like `numpy`, `seaborn`, and `sklearn` are imported later as needed. This is a clean practice — import only what you need, when you need it.

---

### ✅ Step 2 — Load the Dataset

```python
df = pd.read_csv('processed.cleveland.data', header=None)
```

**What & Why:**
- `pd.read_csv()` reads the `.data` file (which is a comma-separated file) into a pandas DataFrame.
- `header=None` tells pandas that the file has **no header row** — the first row is actual data, not column names.
- Without `header=None`, pandas would treat the first patient's data as column names — which is wrong!
- We use the **Cleveland version** specifically because it is the most studied, most clean, and most commonly used version of the Heart Disease dataset.

---

### ✅ Step 3 — Preview the Data

```python
df.head()
```

**What & Why:**
- `df.head()` displays the **first 5 rows** of the DataFrame.
- It's a quick sanity check to confirm the data loaded correctly.
- You can pass a number like `df.head(10)` to see the first 10 rows.
- At this stage, columns show as `0, 1, 2...13` because no column names were assigned yet.

---

### ✅ Step 4 — Assign Column Names

```python
df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
              'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
```

**What & Why:**
- Column names were found in the UCI documentation page for the Heart Disease dataset.
- Assigning proper names makes the data readable and easier to work with.
- Without proper names, we would have to reference columns as `df[0]`, `df[1]` etc. — which is confusing.
- The names are assigned **in the exact order** they appear in the dataset — order matters!

---

### ✅ Step 5 — Check Dataset Shape

```python
df.shape
# Output: (303, 14)
```

**What & Why:**
- `df.shape` returns a tuple `(rows, columns)`.
- Output `(303, 14)` means **303 patients** and **14 columns** (13 features + 1 target).
- This matches the UCI documentation — confirming the dataset loaded correctly.

---

### ✅ Step 6 — Dataset Info

```python
df.info()
```

**What & Why:**
- `df.info()` gives a summary of the DataFrame including:
  - Number of rows and columns
  - Column names
  - Non-null count (how many values are not missing)
  - Data type of each column (`float64` = decimal number, `int64` = integer, `object` = text)
- **Important observation:** `ca` and `thal` show `object` dtype even though they should be numbers. This is because the `"?"` missing values inside them made pandas treat them as text columns.
- All columns show `303 non-null` — but this is misleading! Missing values stored as `"?"` are not detected yet.

---

### ✅ Step 7 — Check for Missing Values (First Check)

```python
df.isnull().sum()
```

**What & Why:**
- `df.isnull()` creates a boolean DataFrame — `True` where values are missing, `False` otherwise.
- `.sum()` counts the `True` values per column. In Python, `True = 1` and `False = 0` — so summing gives the total count of missing values.
- **Result shows all zeros** — because missing values are stored as `"?"` strings, not as `NaN`.
- This is a classic real-world data problem — missing values disguised as strings!

---

### ✅ Step 8 — Replace "?" with NaN

```python
import numpy as np
df.replace('?', np.nan, inplace=True)
```

**What & Why:**
- `np.nan` is the standard Python representation of a missing/null value. It stands for **Not a Number**.
- `df.replace('?', np.nan)` finds all `"?"` strings and replaces them with actual `NaN`.
- `inplace=True` modifies the original DataFrame directly instead of creating a new one. Without it, the changes would be lost.
- After this step, `df.isnull().sum()` will correctly detect the missing values.

---

### ✅ Step 9 — Check Missing Values (Second Check)

```python
df.isnull().sum()
```

**Output:**
```
ca      4
thal    2
(all others)   0
```

**What & Why:**
- Now we can see the **real** missing values:
  - `ca` — 4 missing values
  - `thal` — 2 missing values
- This matches exactly what the UCI documentation stated — **"Has Missing Values? Yes"**
- Total of only **6 missing rows** out of 303 — less than 2% of the data.

---

### ✅ Step 10 — Handle Missing Values (Drop Rows)

```python
df.dropna(inplace=True)
df.shape
# Output: (297, 14)
```

**What & Why:**
- Since only **6 rows** out of 303 have missing values — that's less than 2% of the data.
- The safest approach here is to simply **drop those rows** entirely.
- `dropna()` removes any row that contains at least one `NaN` value.
- `inplace=True` applies the change directly to `df`.
- **Result:** 303 - 6 = **297 clean rows** remaining.
- **Why not fill with mean/mode?** Because `ca` and `thal` are still `object` type at this point — we can't calculate mean/mode on text columns. Dropping was the cleanest solution here.

---

### ✅ Step 11 — Simplify Target Variable

```python
df['target'] = (df['target'] > 0).astype(int)
df['target'].value_counts()
```

**What & Why:**
- The original `target` column has values **0 to 4** representing different severity levels of heart disease.
- For basic binary classification, we simplify this to just **0 or 1**:
  - `0` → No heart disease
  - `1` → Heart disease present (any severity)
- `df['target'] > 0` checks each value — returns `True` for 1,2,3,4 and `False` for 0.
- `.astype(int)` converts `True → 1` and `False → 0`.
- This is called **Binary Classification** — the most common type of ML problem.
- **Result:** Dataset now has only 0s and 1s in the target column.

---

### ✅ Step 12 — Check Basic Statistics

```python
df.describe()
```

**What & Why:**
- `df.describe()` shows statistical summary for every numerical column:
  - `count` — number of values
  - `mean` — average
  - `std` — standard deviation (how spread out values are)
  - `min` / `max` — smallest and largest values
  - `25%`, `50%`, `75%` — quartile values
- Helps identify **outliers** — values that are abnormally high or low.
- Example finding: `chol` has a max of 564 — very high compared to the average of ~246. This is an outlier!

---

### ✅ Step 13 — Exploratory Data Analysis (EDA): Feature Distributions

```python
df.hist(figsize=(12,10))
plt.tight_layout()
plt.show()
```

**What & Why:**
- `df.hist()` creates a **histogram** for every column — showing how values are distributed.
- `figsize=(12,10)` sets the size of the entire figure — 12 inches wide, 10 inches tall.
- `plt.tight_layout()` automatically adjusts spacing between subplots so they don't overlap.
- **Key observations from histograms:**
  - `age` — most patients are between 40-65 years old
  - `sex` — dataset is heavily male-dominated
  - `thalach` — follows a nice bell curve (normal distribution)
  - `chol` — has some extreme outlier values near 500+
  - `fbs` — majority of patients have normal fasting blood sugar

---

### ✅ Step 14 — EDA: Feature vs Target (Boxplots)

```python
df.boxplot(column='age', by='target', figsize=(6,4))
plt.title('Age vs Heart Disease')
plt.suptitle('')
plt.xlabel('0 = No Disease, 1 = Disease')
plt.ylabel('Age')
plt.show()
```

**What & Why:**
- A **boxplot** shows the distribution of a feature split by the target variable.
- It helps us see if a feature actually **influences** the target.
- `by='target'` splits the box plot into two groups — disease vs no disease.
- `plt.suptitle('')` removes the auto-generated pandas title to avoid having two titles.
- **Key observations:**
  - **Age** — patients with disease tend to be older (median ~58 vs ~52). Age IS important! ✅
  - **Thalach** — patients with disease have LOWER max heart rate. Counter-intuitive but medically valid! ✅
  - **Chol** — both groups have almost identical cholesterol levels. Chol is NOT a strong predictor! ❌

---

### ✅ Step 15 — EDA: Correlation Heatmap

```python
import seaborn as sns

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
```

**What & Why:**
- **Correlation** measures how strongly two features are related to each other.
  - Value close to `+1` → Strong positive relationship
  - Value close to `-1` → Strong negative relationship
  - Value close to `0` → No relationship
- `df.corr()` calculates correlation between all pairs of columns.
- `sns.heatmap()` visualizes this as a color-coded grid — red = positive, blue = negative.
- `annot=True` shows the actual numbers inside each cell.
- `fmt='.2f'` formats numbers to 2 decimal places.
- **Key findings from the target row:**
  - `thal` (0.53) — strongest positive correlation with disease ✅
  - `ca` (0.46) — strong positive correlation ✅
  - `thalach` (-0.42) — strong negative correlation ✅
  - `chol` (0.08) — very weak correlation — confirms boxplot finding ❌
  - `fbs` (0.00) — almost zero relationship with disease ❌

---

### ✅ Step 16 — Convert Object Columns to Numeric

```python
df['ca'] = pd.to_numeric(df['ca'])
df['thal'] = pd.to_numeric(df['thal'])
```

**What & Why:**
- Even after replacing `"?"` with `NaN` and dropping those rows — `ca` and `thal` are still stored as `object` (text) type.
- This is because pandas assigned the `object` dtype when it first saw the `"?"` characters — and doesn't automatically change it back.
- `pd.to_numeric()` explicitly converts a column to numeric (float or int) type.
- ML models cannot work with `object` type columns — everything must be numeric.
- After this step, `df.info()` should show `float64` for both `ca` and `thal`.

---

### ✅ Step 17 — Feature Scaling (StandardScaler)

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = df.drop('target', axis=1)
y = df['target']

X_scaled = scaler.fit_transform(X)
print(X_scaled[:5])
```

**What & Why:**
- Different features have very different value ranges:
  - `chol` goes up to 564
  - `oldpeak` only goes up to 6.2
- Without scaling, the model thinks `chol` is more important just because its numbers are bigger — which is unfair!
- **StandardScaler** transforms each feature so that it has **mean = 0** and **standard deviation = 1**.
- `X = df.drop('target', axis=1)` — separates all features. `axis=1` means drop a column (not a row).
- `y = df['target']` — stores the target variable separately.
- **Why not scale `y`?** Because `y` contains class labels (0 and 1). Scaling them would turn them into decimals and break classification.
- `fit_transform()` learns the mean and std of each feature, then applies scaling — all in one step.
- After scaling, all values will be mostly between -3 and +3.

---

### ✅ Step 18 — Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("Training size:", X_train.shape)   # (237, 13)
print("Testing size:", X_test.shape)     # (60, 13)
```

**What & Why:**
- We split the data into two parts:
  - **Training set (80%)** — used to train/teach the model.
  - **Testing set (20%)** — used to evaluate how well the model performs on completely unseen data.
- `test_size=0.2` means 20% of data goes to testing.
- `random_state=42` ensures the split is the same every time you run the code (reproducibility). Without it, you'd get a different split each time — making results inconsistent.
- **Why 80-20 split?** It's the most commonly used ratio in ML — gives enough data for learning while keeping enough for fair evaluation.
- **Result:** 237 rows for training, 60 rows for testing.

---

### ✅ Step 19 — Build the Model (Logistic Regression)

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

print("Model Training Complete!")
```

**What & Why:**
- **Logistic Regression** is a classification algorithm used to predict binary outcomes (0 or 1).
- Despite having "Regression" in its name — it is actually a **classification** algorithm.
- It works by calculating the **probability** that a patient has heart disease:
  - Probability > 0.5 → Predicts Disease (1)
  - Probability < 0.5 → Predicts No Disease (0)
- `model = LogisticRegression()` creates the model — like hiring a student ready to learn.
- `model.fit(X_train, y_train)` trains the model — it studies patterns from 237 patients.
- It's the best starting algorithm for binary classification — simple, fast, and interpretable.

---

### ✅ Step 20 — Make Predictions

```python
y_pred = model.predict(X_test)
print(y_pred)
```

**What & Why:**
- `model.predict(X_test)` uses the trained model to predict class labels for the 60 test patients.
- It returns an array of `0`s and `1`s — where `0 = No Disease` and `1 = Disease`.
- The model has **never seen these 60 patients before** — this is a true test of what it learned!
- These predictions are then compared against the actual labels (`y_test`) to evaluate performance.

---

### ✅ Step 21 — Evaluate with Accuracy Score

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy * 100, "%")
# Output: 86.67%
```

**What & Why:**
- `accuracy_score` compares predicted labels (`y_pred`) vs actual labels (`y_test`).
- It calculates the percentage of correct predictions.
- **86.67% accuracy** means the model correctly predicted **52 out of 60** test patients.
- This is an excellent result for a first simple model!
- Published research papers on this dataset report accuracy in the range of **80% - 90%** — our result is well within that range! ✅

---

### ✅ Step 22 — Confusion Matrix

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

**Output:**
```
True Negative  = 32   False Positive = 4
False Negative = 4    True Positive  = 20
```

**What & Why:**
- A **Confusion Matrix** gives a detailed breakdown of correct and incorrect predictions:
  - **True Negative (32)** — Patient had no disease, model correctly predicted no disease ✅
  - **True Positive (20)** — Patient had disease, model correctly predicted disease ✅
  - **False Positive (4)** — Patient was healthy, but model predicted disease ❌ (False Alarm)
  - **False Negative (4)** — Patient had disease, but model predicted healthy ❌ (Missed Case)
- **False Negative is the most dangerous** in medical scenarios — a sick patient being told they are healthy could be life-threatening!
- `annot=True` shows the actual numbers inside each cell.
- `fmt='d'` formats numbers as integers (not decimals).
- Total correct = 32 + 20 = **52** ✅
- Total wrong = 4 + 4 = **8** ❌

---

## 📊 Final Results

| Metric | Value |
|---|---|
| Training Size | 237 rows |
| Testing Size | 60 rows |
| Model Used | Logistic Regression |
| Accuracy | **86.67%** |
| Correct Predictions | 52 out of 60 |
| False Positives | 4 |
| False Negatives | 4 |

> Published research on this dataset reports 80% - 90% accuracy. Our result is well within that range! ✅

---

## 🔑 Key EDA Findings

| Feature | Relationship with Disease | Strength |
|---|---|---|
| `thal` | Higher value = more disease | ✅ Strong (0.53) |
| `ca` | More vessels = more disease | ✅ Strong (0.46) |
| `exang` | Exercise angina = more disease | ✅ Moderate (0.42) |
| `oldpeak` | Higher ST depression = more disease | ✅ Moderate (0.42) |
| `thalach` | Lower max heart rate = more disease | ✅ Strong (-0.42) |
| `cp` | Chest pain type matters | ✅ Moderate (0.41) |
| `chol` | Almost no relationship | ❌ Weak (0.08) |
| `fbs` | Almost zero relationship | ❌ None (0.00) |

---

## 🛠️ Libraries Used

| Library | Purpose |
|---|---|
| `pandas` | Data loading, cleaning, manipulation |
| `numpy` | Replacing `"?"` with `NaN` |
| `matplotlib` | Visualization (histograms, boxplots) |
| `seaborn` | Enhanced visualization (heatmap, confusion matrix) |
| `sklearn.preprocessing.StandardScaler` | Feature scaling |
| `sklearn.model_selection.train_test_split` | Splitting data |
| `sklearn.linear_model.LogisticRegression` | Classification model |
| `sklearn.metrics.accuracy_score` | Evaluating accuracy |
| `sklearn.metrics.confusion_matrix` | Detailed prediction breakdown |

---

## 📜 Citation

Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). Heart Disease [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X

---

*Notes prepared as part of ML Internship Task 2 — VAUTECH IT SOLUTIONS*
