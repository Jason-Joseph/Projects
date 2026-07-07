# 📊 Projects

A collection of data analysis and machine learning coursework: regression, classification, clustering, and exploratory analysis in Python and R.

---

## ✈️ [Data Expo 2002-2003 Airline Time Data.ipynb](Data%20Expo%202002-2003%20Airline%20Time%20Data.ipynb)

Analyzes 11.7 million US domestic flight records from 2002 and 2003, split into five questions.

**Approach:**
- 🧹 Merged the 2002 and 2003 raw CSVs, verified there were no cancelled flights to exclude, and engineered a single `delay` variable (`ArrDelay + DepDelay`) as the basis for all five questions
- 📅 **Q1: Timing.** Grouped delay by day of week and scheduled time to find the lowest delay windows
- ✈️ **Q2: Aircraft age.** Cross referenced plane manufacture year against delay to test whether older aircraft fly worse
- 📈 **Q3: Traffic trends.** Tracked flight volume between locations over time
- 🔗 **Q4: Cascading delays.** Correlated departure delay against arrival delay across the network
- 🤖 **Q5: Predictive modeling.** Built and compared four regression models to predict arrival delay from `Month`, `DayOfWeek`, `CRSDepTime`, `CRSArrTime`, `ActualElapsedTime`, `DepDelay`, and `Distance`

The features split into one near linear driver, `DepDelay`, given Q4's finding that it correlates closely with `ArrDelay`, and several weaker, noisier schedule and time features. Linear Regression tests how far a plain linear fit gets. Lasso and Ridge test whether regularizing away the noisier features helps. Random Forest tests whether those features hide non-linear interactions a linear model can't reach.

**Result of the assessment:**

| Model | R² | MSE |
|---|---|---|
| Random Forest | **0.975** | **37.98** |
| Linear Regression | 0.939 | 93.84 |
| Ridge Regression | 0.939 | 93.86 |
| Lasso Regression | 0.842 | 242.82 |

Random Forest wins outright. It captures non-linear interactions between the schedule and time features that the linear models miss, so the relationship isn't purely linear. Lasso lags behind Ridge and Linear too, which means its regularization penalty cut real signal, not just noise.

**Key findings:** Saturday and 3:30 AM departures see the lowest average delay. Departure delay predicts arrival delay closely, which supports the cascading delay theory from Q4. Random Forest is the strongest predictor of the four models tested.

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

## 🚗 [Vehicle Price Regression Analysis.R](Vehicle%20Price%20Regression%20Analysis.R)

Predicts car price from the Kaggle CarPrice dataset using three regression models.

**Approach:**
- 🧹 Dropped identifier columns, removed duplicates, and used the **IQR method** to strip outliers from `price`
- 📊 Explored feature distributions and correlations before modeling
- ✂️ 80/20 train-test split (`caTools`, seed-fixed for reproducibility)
- 📐 **Multiple Linear Regression**, fit then refined with **backward stepwise elimination** to drop non-significant predictors
- 🌳 **CART (Decision Tree)**, grown to full depth then **cost-complexity pruned** using the 1-SE rule on cross-validated error
- 🌲 **Random Forest** (500 trees), with permutation-based variable importance

**Evaluation:** All three models are scored on **RMSE** and **R²** in a summary table. CART and Random Forest importance rankings both point to `enginesize`, `curbweight`, and `horsepower` as the strongest predictors.

---

## 📞 [Customer Churn Clustering and Classification.R](Customer%20Churn%20Clustering%20and%20Classification.R)

A two-part analysis of telecom customer churn. The unsupervised half segments customers into groups. The supervised half predicts who is about to churn.

**Approach, unsupervised:**
- 🧹 Cleaned and recoded status labels, imputed missing values with the median, removed outliers from `Total_Revenue`
- 📉 **PCA** to understand variance structure. The first 6 components capture 76% of variance, driven by tenure and charges, age, and long-distance usage
- 🎯 **K-means** with k=2, chosen via the silhouette method and validated with **ANOVA**. The two clusters differ in tenure and revenue
- 🌿 **Hierarchical clustering** (Ward linkage) as a second, independent segmentation to cross-check the K-means result

**Approach, supervised classification:**
- ✂️ 80/20 train-test split, features scaled for the linear model
- 📐 **Logistic Regression** as the interpretable baseline
- 🌳 **Decision Tree** with a confusion-matrix heatmap for visual diagnostics
- 🌲 **Random Forest**. Importance ranking flags `Contract`, `Monthly_Charge`, and `Tenure` as the top churn drivers

**Evaluation:** Accuracy, Precision, Recall, and macro-averaged **F1** across all three classifiers, ranked in a final comparison table. Each lands around **85% accuracy**.

---

## 🛒 [Kaggle Superstore Data.ipynb](Kaggle%20Superstore%20Data.ipynb)

Exploratory analysis of retail superstore sales: where sales come from, and where profit leaks.

**Approach:**
- 🧹 Cleaned and processed raw sales records, checked for missing values
- 📈 Trend-lined sales and profit over time
- 🗂️ Broke down performance by **category**, **region**, and **customer segment**
- 📉 Correlation matrix to sanity-check relationships between sales, profit, and discount
- 📐 A **Linear Regression** (`Profit ~ Sales + Quantity + Discount`) to quantify how discount affects profit

**Key findings:** Office Supplies is the most purchased category, but Technology drives the most profit. That gap points to a rebalancing opportunity away from Furniture. Discounting hurts profit margins more than it grows volume.

<table>
<tr>
<td><img src="assets/category_profit_pie.png" width="380"/></td>
<td><img src="assets/superstore_correlation.png" width="380"/></td>
</tr>
</table>

---

## 🛍️ [Online Retail Customer Analytics Dashboard.pdf](Online%20Retail%20Customer%20Analytics%20Dashboard.pdf)

Capstone project from the Dibimbing data analytics bootcamp, Batch 14. A customer analytics and retention study on a UK online retail dataset spanning December 2009 to December 2011, about 1.07 million transactions, built around three questions: how customers behave, which products sell best, and how to retain existing customers.

**Approach:**
- 🧹 **Pre-processing.** Removed rows missing `CustomerID`, about 22% of the data, since customer identity matters for the segmentation work. Filtered out non-positive quantities, leaving 824,364 clean rows
- 📊 **RFM Analysis.** Scored every customer on **Recency**, **Frequency**, and **Monetary** value, then bucketed them into named segments (Best Customer, Loyal Customers, Big Spender, Potential Customers, Lost Cheap, Almost Lost) to prioritize retention effort
- 📈 **Cohort Analysis.** Tracked month by month retention for each signup cohort to see how long customers keep buying
- 📋 **Executive Dashboard.** A single page view combining sales trend, RFM segment mix, geographic revenue contribution, and top selling products for stakeholders

**Key findings:**
- The UK dominates the customer base: £14.7M in revenue against £621K for the next largest market, Ireland. That's a concentration risk as much as a strength
- Sales peaked in **Q4 2010** and show a recurring seasonal spike heading into every Q4, but overall 2011 volume and sales declined 4 to 10% versus 2010
- The **December 2009 cohort** retained well: 20 to 40% of customers were still purchasing 24 months later. Across cohorts generally, **the first 6 to 10 months after first purchase** is the window where most customers churn
- **45.68%** of customers fall into the "Others/Recent Shopper" segment, the single largest RFM bucket, ahead of Loyal Customers (10.02%) and Best Customer (11.61%)

**Recommendations proposed:** a loyalty points program, targeted re-engagement campaigns timed to the 6 to 10 month churn window, seasonal Q4 promotions, and expansion into the EIRE and Netherlands markets to reduce UK concentration risk.

<img src="assets/rfm_dashboard.png" width="700"/>

<table>
<tr>
<td><img src="assets/cohort_analysis.png" width="380"/></td>
<td><img src="assets/best_selling_products.png" width="380"/></td>
</tr>
</table>

---

## 🧰 Tech Stack

![Python](https://img.shields.io/badge/Python-pandas%20%7C%20scikit--learn-blue)
![R](https://img.shields.io/badge/R-tidyverse%20%7C%20caret%20%7C%20randomForest-276DC3)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626)
