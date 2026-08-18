import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


st.set_page_config(
    page_title="Loan Amount Prediction",
    page_icon="💰",
    layout="wide"
)

st.markdown(
    """
    <style>
    .title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0;
    }
    .subtitle {
        font-size: 18px;
        margin-bottom: 25px;
    }
    .prediction-box {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        border: 2px solid #ddd;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<p class="title">💰 Loan Amount Prediction</p>',
    unsafe_allow_html=True
)
st.markdown(
    '<p class="subtitle">Random Forest Regression + Streamlit</p>',
    unsafe_allow_html=True
)

DATA_FILE = "preprocessed_data.csv"

# These are the same applicant columns used in the uploaded project,
# but LoanAmount is now the TARGET instead of Loan_Status.
FEATURES = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "ApplicantIncome",
    "CoapplicantIncome",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area"
]

CATEGORICAL_COLUMNS = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area"
]

NUMERICAL_COLUMNS = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "Loan_Amount_Term",
    "Credit_History"
]


def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)


@st.cache_resource(show_spinner=False)
def train_model(df):
    data = df.copy()
    data = data.drop(columns=["Loan_ID"], errors="ignore")

    required = FEATURES + ["LoanAmount"]
    missing = [col for col in required if col not in data.columns]

    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )

    # LoanAmount is the regression target.
    data["LoanAmount"] = pd.to_numeric(
        data["LoanAmount"], errors="coerce"
    )
    data = data.dropna(subset=["LoanAmount"])

    X = data[FEATURES].copy()
    y = data["LoanAmount"].copy()

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", make_ohe())
        ]
    )

    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_transformer,
                CATEGORICAL_COLUMNS
            ),
            (
                "numerical",
                numerical_transformer,
                NUMERICAL_COLUMNS
            )
        ]
    )

    rf = RandomForestRegressor(
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("random_forest", rf)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    param_distributions = {
        "random_forest__n_estimators": [100, 200, 300],
        "random_forest__max_depth": [5, 8, 10, 15, None],
        "random_forest__min_samples_split": [2, 5, 10],
        "random_forest__min_samples_leaf": [1, 2, 4],
        "random_forest__max_features": [1.0, "sqrt", "log2"]
    }

    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=20,
        cv=5,
        scoring="neg_mean_squared_error",
        random_state=42,
        n_jobs=-1,
        verbose=0
    )

    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_

    y_pred = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    rf_model = best_model.named_steps["random_forest"]
    fitted_preprocessor = best_model.named_steps["preprocessor"]

    feature_names = fitted_preprocessor.get_feature_names_out()

    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": rf_model.feature_importances_
    }).sort_values(
        "Importance",
        ascending=False
    )

    results = pd.DataFrame({
        "Actual Loan Amount": y_test.values,
        "Predicted Loan Amount": y_pred
    })

    return {
        "model": best_model,
        "features": FEATURES,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "cv_rmse": np.sqrt(-random_search.best_score_),
        "best_params": random_search.best_params_,
        "feature_importance": feature_importance,
        "results": results,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred
    }


# -----------------------------
# Load dataset
# -----------------------------
try:
    df = load_data()
except FileNotFoundError:
    st.error(
        "❌ preprocessed_data.csv was not found. "
        "Put it in the same folder as app.py."
    )
    st.stop()
except Exception as e:
    st.error(f"❌ Dataset loading error: {e}")
    st.stop()


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("⚙️ Model Information")
st.sidebar.write("Target:")
st.sidebar.code("LoanAmount")

st.sidebar.metric("Dataset Rows", len(df))
st.sidebar.metric("Dataset Columns", len(df.columns))

if st.sidebar.button("🔄 Retrain Model"):
    st.cache_resource.clear()
    st.rerun()


# -----------------------------
# Train model
# -----------------------------
with st.spinner("Training Random Forest Regression model..."):
    try:
        output = train_model(df)
    except Exception as e:
        st.error(f"❌ Model training failed: {e}")
        st.stop()


model = output["model"]


# -----------------------------
# Metrics
# -----------------------------
st.subheader("📊 Model Performance")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "R² Score",
        f"{output['r2']:.3f}"
    )

with c2:
    st.metric(
        "MAE",
        f"{output['mae']:.2f}"
    )

with c3:
    st.metric(
        "RMSE",
        f"{output['rmse']:.2f}"
    )

with c4:
    st.metric(
        "CV RMSE",
        f"{output['cv_rmse']:.2f}"
    )


tab1, tab2, tab3, tab4 = st.tabs(
    [
        "💰 Predict Loan Amount",
        "📊 Evaluation",
        "⭐ Feature Importance",
        "📁 Dataset"
    ]
)


# ============================================================
# TAB 1 - PREDICTION
# ============================================================
with tab1:
    st.subheader("Enter Applicant Details")

    c1, c2, c3 = st.columns(3)

    with c1:
        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        married = st.selectbox(
            "Married",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["0", "1", "2", "3+"]
        )

        education = st.selectbox(
            "Education",
            ["Graduate", "Not Graduate"]
        )

    with c2:
        self_employed = st.selectbox(
            "Self Employed",
            ["No", "Yes"]
        )

        applicant_income = st.number_input(
            "Applicant Income",
            min_value=0.0,
            value=5000.0,
            step=500.0
        )

        coapplicant_income = st.number_input(
            "Coapplicant Income",
            min_value=0.0,
            value=1500.0,
            step=500.0
        )

    with c3:
        loan_term = st.selectbox(
            "Loan Amount Term",
            [360, 180, 120, 84, 60, 48, 36, 24, 12],
            index=0
        )

        credit_history = st.selectbox(
            "Credit History",
            [1.0, 0.0],
            format_func=lambda x:
                "Good Credit History (1)" if x == 1.0
                else "No/Bad Credit History (0)"
        )

        property_area = st.selectbox(
            "Property Area",
            ["Urban", "Semiurban", "Rural"]
        )

    predict = st.button(
        "🚀 Predict Loan Amount",
        type="primary",
        use_container_width=True
    )

    if predict:
        input_data = pd.DataFrame({
            "Gender": [gender],
            "Married": [married],
            "Dependents": [dependents],
            "Education": [education],
            "Self_Employed": [self_employed],
            "ApplicantIncome": [applicant_income],
            "CoapplicantIncome": [coapplicant_income],
            "Loan_Amount_Term": [loan_term],
            "Credit_History": [credit_history],
            "Property_Area": [property_area]
        })

        try:
            predicted_amount = float(
                model.predict(input_data)[0]
            )

            predicted_amount = max(0, predicted_amount)

            st.markdown(
                f"""
                <div class="prediction-box">
                    <h2>💰 Predicted Loan Amount</h2>
                    <h1>₹ {predicted_amount:,.2f}</h1>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.success(
                f"The Random Forest model predicts a loan amount of "
                f"₹ {predicted_amount:,.2f}."
            )

            st.info(
                "This is a machine-learning prediction for educational/demo "
                "purposes and is not a real banking approval or lending decision."
            )

        except Exception as e:
            st.error(f"Prediction error: {e}")


# ============================================================
# TAB 2 - EVALUATION
# ============================================================
with tab2:
    st.subheader("Regression Evaluation")

    st.write("### Actual vs Predicted Loan Amount")

    results = output["results"].copy()
    st.dataframe(
        results.head(20),
        use_container_width=True
    )

    fig, ax = plt.subplots()

    ax.scatter(
        output["y_test"],
        output["y_pred"],
        alpha=0.7
    )

    min_value = min(
        output["y_test"].min(),
        output["y_pred"].min()
    )
    max_value = max(
        output["y_test"].max(),
        output["y_pred"].max()
    )

    ax.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--"
    )

    ax.set_xlabel("Actual Loan Amount")
    ax.set_ylabel("Predicted Loan Amount")
    ax.set_title("Actual vs Predicted Loan Amount")

    st.pyplot(fig)
    plt.close(fig)

    st.write("### Regression Metrics")

    metrics_df = pd.DataFrame({
        "Metric": [
            "R² Score",
            "Mean Absolute Error (MAE)",
            "Mean Squared Error (MSE)",
            "Root Mean Squared Error (RMSE)"
        ],
        "Value": [
            output["r2"],
            output["mae"],
            output["mse"],
            output["rmse"]
        ]
    })

    st.dataframe(
        metrics_df,
        use_container_width=True
    )

    st.write("### Best Random Forest Parameters")
    st.json(output["best_params"])


# ============================================================
# TAB 3 - FEATURE IMPORTANCE
# ============================================================
with tab3:
    st.subheader("⭐ Feature Importance")

    top_n = st.slider(
        "Number of features",
        min_value=5,
        max_value=min(
            30,
            len(output["feature_importance"])
        ),
        value=min(
            15,
            len(output["feature_importance"])
        )
    )

    fi = output["feature_importance"].head(top_n).copy()

    fi["Feature"] = (
        fi["Feature"]
        .str.replace("categorical__", "", regex=False)
        .str.replace("numerical__", "", regex=False)
    )

    st.dataframe(
        fi,
        use_container_width=True
    )

    chart_df = fi.sort_values("Importance")

    st.bar_chart(
        chart_df.set_index("Feature")["Importance"]
    )


# ============================================================
# TAB 4 - DATASET
# ============================================================
with tab4:
    st.subheader("📁 Dataset")

    st.write("### First 10 Rows")
    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.write("### Dataset Information")

    info_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [
            str(dtype)
            for dtype in df.dtypes
        ],
        "Missing Values": [
            int(value)
            for value in df.isna().sum()
        ]
    })

    st.dataframe(
        info_df,
        use_container_width=True
    )

    st.write("### Loan Amount Distribution")

    if "LoanAmount" in df.columns:
        st.bar_chart(
            df["LoanAmount"].dropna()
        )


st.divider()

st.caption(
    "Loan Amount Prediction | Random Forest Regression | Streamlit"
)
