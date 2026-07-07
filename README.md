# 📊 Projects

A collection of data analysis and machine learning coursework, spanning regression, classification, clustering, and exploratory analysis in both Python and R.

---

## ✈️ [Data Expo 2002-2003 Airline Time Data.ipynb](Data%20Expo%202002-2003%20Airline%20Time%20Data.ipynb)

An end-to-end analysis of **11.7 million US domestic flight records** (2002–2003), structured around five investigative questions.

**Approach:**
- 🧹 Merged the 2002 and 2003 raw CSVs, verified there were no cancelled flights to exclude, and engineered a single `delay` variable (`ArrDelay + DepDelay`) as the basis for all five questions
- 📅 **Q1 — Timing:** Grouped delay by day-of-week and scheduled time to find the lowest-delay windows
- ✈️ **Q2 — Aircraft age:** Cross-referenced plane manufacture year against delay to test whether older aircraft fly worse
- 📈 **Q3 — Traffic trends:** Tracked flight volume between locations over time
- 🔗 **Q4 — Cascading delays:** Correlated departure delay against arrival delay across the network
- 🤖 **Q5 — Predictive modeling:** Built and compared regression models (Linear Regression, Lasso, Ridge, Random Forest) to predict arrival delay, evaluated on **MSE** and **R²**

**Key findings:** Saturday and ~3:30 AM departures see the lowest average delay; departure delay strongly predicts arrival delay (evidence of cascading effects); Linear Regression matches the more complex models on accuracy.

<table>
<tr>
<td><img src="assets/q1_delay_by_day.png" width="380"/></td>
<td><img src="assets/q4_correlation_matrix.png" width="380"/></td>
</tr>
<tr>
<td><img src="assets/q5_mse_comparison.png" width="380"/></td>
<td><img src="assets/q5_rsq_comparison.png" width="380"/></td>
</tr>
</table>

---

## 🚗 [Coursework Regression.R](Coursework%20Regression.R)

Predicts car price from the Kaggle **CarPrice** dataset using three regression techniques of increasing complexity.

**Approach:**
- 🧹 Dropped identifier columns, removed duplicates, and used the **IQR method** to strip outliers from `price`
- 📊 Explored feature distributions and correlations before modeling
- ✂️ 80/20 train-test split (`caTools`, seed-fixed for reproducibility)
- 📐 **Multiple Linear Regression** — fit, then refined with **backward stepwise elimination** to drop non-significant predictors
- 🌳 **CART (Decision Tree)** — grown to full depth, then **cost-complexity pruned** using the 1-SE rule on cross-validated error
- 🌲 **Random Forest** (500 trees) — with permutation-based variable importance

**Evaluation:** All three models scored on **RMSE** and **R²**, ranked in a summary table. `enginesize`, `curbweight`, and `horsepower` consistently emerged as the strongest predictors across CART and Random Forest importance rankings.

---

## 📞 [Coursework Classification Unsupervised.R](Coursework%20Classification%20Unsupervised.R)

A two-part analysis of **telecom customer churn** — first unsupervised (who are the customer segments?), then supervised (can we predict who churns?).

**Approach — Unsupervised:**
- 🧹 Cleaned and recoded status labels, imputed missing values with the median, removed outliers from `Total_Revenue`
- 📉 **PCA** to understand variance structure — the first 6 components capture 76% of variance, driven by tenure/charges, age, and long-distance usage patterns
- 🎯 **K-means** (k=2, chosen via the silhouette method) — validated with **ANOVA**, confirming the two clusters differ significantly in tenure and revenue
- 🌿 **Hierarchical clustering** (Ward linkage) as a second, independent segmentation to cross-check the K-means result

**Approach — Supervised classification:**
- ✂️ 80/20 train-test split, features scaled for the linear model
- 📐 **Logistic Regression** as the interpretable baseline
- 🌳 **Decision Tree** with a confusion-matrix heatmap for visual diagnostics
- 🌲 **Random Forest** — importance ranking flagged `Contract`, `Monthly_Charge`, and `Tenure` as the top churn drivers

**Evaluation:** Accuracy, Precision, Recall, and macro-averaged **F1** across all three classifiers — each landing around **85% accuracy**, ranked in a final comparison table.

---

## 🛒 [Kaggle Superstore Data.ipynb](Kaggle%20Superstore%20Data.ipynb)

Exploratory analysis of retail superstore sales, asking where sales come from and where profit actually leaks.

**Approach:**
- 🧹 Cleaned and processed raw sales records, checked for missing values
- 📈 Trend-lined sales and profit over time
- 🗂️ Broke down performance by **category**, **region**, and **customer segment**
- 📉 Correlation matrix to sanity-check relationships between sales, profit, and discount
- 📐 A simple **Linear Regression** (`Profit ~ Sales + Quantity + Discount`) to quantify the discount-profit relationship directly

**Key findings:** Office Supplies is the most-purchased category, but Technology drives the most profit — suggesting a possible rebalancing opportunity away from Furniture. Discounting measurably hurts profit margins rather than growing volume enough to offset it.

<table>
<tr>
<td><img src="assets/category_profit_pie.png" width="380"/></td>
<td><img src="assets/superstore_correlation.png" width="380"/></td>
</tr>
</table>

---

## 🎓 [Dibimbing B14 Final Project.pdf](Dibimbing%20B14%20Final%20Project.pdf)

Final capstone project writeup from the **Dibimbing** data analytics bootcamp (Batch 14).

---

## 🧰 Tech Stack

![Python](https://img.shields.io/badge/Python-pandas%20%7C%20scikit--learn-blue)
![R](https://img.shields.io/badge/R-tidyverse%20%7C%20caret%20%7C%20randomForest-276DC3)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626)
