# Loan Amount Prediction - Random Forest + Streamlit

This version predicts **LoanAmount**, not Loan_Status.

## Model

The target is:

```text
LoanAmount
```

The model is:

```text
RandomForestRegressor
```

The categorical columns are one-hot encoded and missing values are handled with imputers.

RandomizedSearchCV is used to search for better Random Forest parameters.

## Project structure

```text
loan_amount_random_forest_streamlit/
├── app.py
├── requirements.txt
├── README.md
└── preprocessed_data.csv
```

Put your existing `preprocessed_data.csv` in this folder.

## Required dataset columns

```text
Gender
Married
Dependents
Education
Self_Employed
ApplicantIncome
CoapplicantIncome
LoanAmount
Loan_Amount_Term
Credit_History
Property_Area
```

`Loan_Status` is NOT used as the target in this version.

`Loan_ID` can exist; the app ignores it.

## Install

Open the terminal in the project folder:

```bash
pip install -r requirements.txt
```

or:

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

## What the webpage provides

1. Applicant input form
2. Predicted loan amount
3. R² score
4. MAE
5. MSE
6. RMSE
7. Actual vs Predicted graph
8. Feature importance
9. Best Random Forest parameters
10. Dataset preview

The prediction is for educational/demo purposes and is not a real banking lending decision.
