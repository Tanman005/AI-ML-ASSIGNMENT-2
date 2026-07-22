# Customer Churn Prediction using Logistic Regression

## Objective
A telecommunications company wants to predict whether a customer is likely to leave (churn)
based on demographic information and service usage. This project builds a Logistic Regression
model to predict customer churn using the Telco Customer Churn dataset.

## Dataset
**Telco Customer Churn Dataset** (Kaggle)
Link: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

> The dataset is not included in this repository. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv`
> from the Kaggle link above and place it in the project folder before running the notebook.

## Libraries Used
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn (`train_test_split`, `LabelEncoder`, `StandardScaler`, `LogisticRegression`,
  evaluation metrics)

## Methodology
1. **Data Understanding** — Loaded the dataset, inspected the first five records, and
   identified numerical features, categorical features, and the target variable (`Churn`).
2. **Data Preprocessing** — Converted `TotalCharges` to numeric, handled missing values
   (filled with 0 for customers with zero tenure), dropped the non-predictive `customerID`
   column, label-encoded the target, one-hot encoded categorical features, and split the
   data into 80% training / 20% testing sets. Numerical features were scaled with
   `StandardScaler`.
3. **Model Development** — Trained a Logistic Regression model (`max_iter=1000`) on the
   processed training data to predict churn.
4. **Model Evaluation** — Evaluated the model on the held-out test set using Accuracy,
   Precision, Recall, F1-Score, and a Confusion Matrix.
5. **Conclusion** — Summarized key drivers of churn and a limitation of Logistic Regression
   for this problem.

## Results
> Fill in with the actual values printed after running `Assignment-2.ipynb`:

| Metric    | Score |
|-----------|-------|
| Accuracy  |       |
| Precision |       |
| Recall    |       |
| F1-Score  |       |

**Confusion Matrix:** see the heatmap generated in the notebook (Task 4).

**Key observations:**
- (Fill in from the notebook's Observations section — e.g. class imbalance affecting recall,
  which features had the strongest coefficients, etc.)

## Conclusion
This Logistic Regression model provides a solid baseline for predicting customer churn using
demographic and service-usage attributes. The key findings show that contract type, tenure,
and internet/service add-ons are strong drivers of churn: customers on month-to-month
contracts with shorter tenure and fewer bundled services are considerably more likely to
leave, while long-tenure customers on annual or two-year contracts churn far less often.
Charges (both monthly and total) also play a role, suggesting price sensitivity contributes
to attrition. One notable limitation of Logistic Regression for this problem is that it
assumes a linear relationship between the log-odds of churn and the input features, so it
can struggle to capture more complex, non-linear interactions between variables that a
tree-based or ensemble model might capture more effectively.
