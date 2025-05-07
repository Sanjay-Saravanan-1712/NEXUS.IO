import streamlit as st
import pandas as pd
import os
import joblib
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, r2_score, silhouette_score

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans
import xgboost as xgb
import lightgbm as lgb

st.set_page_config(page_title="NEXUS.IO", layout="wide")
st.title("🤖 NEXUS.IO — Custom AutoML Assistant")

# Session state initialization
if "page" not in st.session_state:
    st.session_state.page = None
if "file" not in st.session_state:
    st.session_state.file = None

# Sidebar navigation
with st.sidebar:
    st.title("NEXUS.IO -- AutoML")
    st.info("Custom AutoML using scikit-learn + XGBoost + LightGBM")
    if st.button('Dataset'):
        st.session_state.page = "Dataset"
    if st.button('Dataset Profiling'):
        st.session_state.page = "Profiling"
    if st.button('ML Model Training'):
        st.session_state.page = "Model"
    if st.button('Download Model'):
        st.session_state.page = "Download"

# File upload and display
if st.session_state.page == "Dataset":
    st.title("Upload Your CSV File")
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file, encoding='latin1')
        df.to_csv("data.csv", index=False)
        st.dataframe(df)

# EDA using ydata profiling
if st.session_state.page == "Profiling":
    if os.path.exists("data.csv"):
        df = pd.read_csv("data.csv")
        st.title("Automated Exploratory Data Analysis")
        profile = ProfileReport(df, explorative=True)
        st_profile_report(profile)
    else:
        st.warning("Please upload a dataset first.")

# Model training

if st.session_state.page == "Model":
    if os.path.exists("data.csv"):
        df = pd.read_csv("data.csv")
        st.title("Train Your Model")
        model_type = st.selectbox("Choose the problem type", ["Classification", "Regression", "Clustering"])

        if model_type != "Clustering":
            target = st.selectbox("Select the target column", df.columns)
            X = df.drop(columns=[target])
            y = df[target]
        else:
            X = df.select_dtypes(include=['float64', 'int64'])

        # Clean the dataset by removing unwanted characters
        def clean_numeric_columns(df):
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = df[col].replace({'[^0-9.]': ''}, regex=True)  # Remove non-numeric characters
                df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric, invalid parsing becomes NaN
            return df

        df = clean_numeric_columns(df)

        # Preprocessing pipeline
        num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer([
            ("num", numeric_pipeline, num_features),
            ("cat", categorical_pipeline, cat_features)
        ])

        if model_type == "Classification":
            st.subheader("Running AutoML for Classification...")
            models = {
                "Random Forest": RandomForestClassifier(),
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "SVM": SVC(),
                "XGBoost": xgb.XGBClassifier(),
                "LightGBM": lgb.LGBMClassifier()
            }

            best_model = None
            best_score = 0
            for name, model in models.items():
                pipe = Pipeline([
                    ("pre", preprocessor),
                    ("clf", model)
                ])
                scores = cross_val_score(pipe, X, y, cv=5, scoring='accuracy')
                st.write(f"{name}: Accuracy = {scores.mean():.4f}")
                if scores.mean() > best_score:
                    best_score = scores.mean()
                    best_model = pipe

        elif model_type == "Regression":
            st.subheader("Running AutoML for Regression...")
            models = {
                "Random Forest": RandomForestRegressor(),
                "Linear Regression": LinearRegression(),
                "SVR": SVR(),
                "XGBoost": xgb.XGBRegressor(),
                "LightGBM": lgb.LGBMRegressor()
            }

            best_model = None
            best_score = -float('inf')
            for name, model in models.items():
                pipe = Pipeline([
                    ("pre", preprocessor),
                    ("reg", model)
                ])
                scores = cross_val_score(pipe, X, y, cv=5, scoring='r2')
                st.write(f"{name}: R2 = {scores.mean():.4f}")
                if scores.mean() > best_score:
                    best_score = scores.mean()
                    best_model = pipe

        elif model_type == "Clustering":
            st.subheader("Running Clustering with KMeans...")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model = KMeans(n_clusters=3, random_state=42)
            model.fit(X_scaled)
            labels = model.labels_
            score = silhouette_score(X_scaled, labels)
            df['Cluster'] = labels
            st.write(f"KMeans Silhouette Score: {score:.4f}")
            st.dataframe(df)
            best_model = Pipeline([
                ("scaler", scaler),
                ("cluster", model)
            ])

        st.session_state.model = best_model
        if st.button("Save Trained Model"):
            joblib.dump(best_model, "model.pkl")
            st.success("Model saved as model.pkl!")

    else:
        st.warning("Please upload and select a dataset first.")

# Download section
if st.session_state.page == "Download":
    st.title("Download Trained Model")
    if os.path.exists("model.pkl"):
        with open("model.pkl", 'rb') as f:
            st.download_button("Download Trained Model", f, "model.pkl")
    else:
        st.warning("No model available. Train and save one first.")
