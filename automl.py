import streamlit as st
import pandas as pd
import os
import ydata_profiling as pandas_profiling
from streamlit_pandas_profiling import st_profile_report
from pycaret.classification import setup,compare_models,save_model
from pycaret.regression import setup as r_setup, compare_models as r_compare_models
from pycaret.clustering import setup as c_setup, create_model as c_create_model, predict_model

st.title("Welcome to NEXUS.io --AutoML platform")
st.title("🤖 NEXUS.IO — AutoML Assistant")
st.markdown("""
Welcome to **NEXUS.IO**, a beginner-friendly AutoML platform built using **Streamlit**, **PyCaret**, and **ydata-profiling**. This project is designed as a smart assistant that simplifies the machine learning lifecycle for users with little to no coding experience.

---

## 🎯 Project Objective
The goal is to automate the core ML pipeline:
- Ingest and analyze raw datasets
- Select the best ML model through comparison
- Train, evaluate, and save models — all in a few clicks

This version uses **PyCaret** to handle the modeling layer for **classification**, **regression**, and **clustering**.

---

## 🧠 Features Implemented
- ✅ Upload a CSV dataset  
- ✅ Automatically generate an EDA report using `ydata-profiling`  
- ✅ Choose model type (classification/regression/clustering)  
- ✅ Auto-train and compare models using `PyCaret`  
- ✅ View the best-performing model  
- ✅ Save the model for future use  

---

## 🛠 Tech Stack
| Module         | Tool Used              |
|----------------|------------------------|
| UI Framework   | Streamlit              |
| AutoML Engine  | PyCaret                |
| EDA Tool       | ydata-profiling        |
| Backend Logic  | Python (Pandas, OS)    |

---

## 🚧 Limitations & Future Scope

> ⚠ **Note:** This current version relies heavily on **PyCaret**, which acts as the AutoML backbone. While it's effective for rapid development and prototyping, it abstracts away many low-level details that are important for customization and fine-tuning.

---

### 🔄 What's Next?

I'm currently planning to **replace PyCaret** with a **custom AutoML pipeline** that will:
- ✅ Include manual preprocessing steps (e.g., missing value imputation, encoding)
- ✅ Implement model selection logic using `sklearn`, `xgboost`, and `lightgbm`
- ✅ Add custom metrics tracking and visualizations
- ✅ Possibly integrate experiment tracking using MLflow or Weights & Biases

This will make the platform **more flexible, transparent, and extensible**.

---

## 🧭 Navigation Guide

Use the sidebar to explore the tool:
- **Dataset** → Upload your CSV file  
- **Dataset Profiling** → Automatically explore and summarize your data  
- **ML Model Training** → Let the system pick and train the best models  
- **Download Model** → Save your model as a `.pkl` file  

---

## ✨ Why NEXUS.IO?
- Saves hours of model experimentation time
- Great for quick baselines and teaching ML concepts
- Built to evolve from simple no-code workflows to full-scale pipelines

---
""")
if "page" not in st.session_state:
     st.session_state.page=None
elif "file" not in st.session_state:
     st.session_state.file=None

with st.sidebar:
    #st.image()
    st.title("NEXUS.IO --AutoML")
    st.info("AutoML will automatically select the best models, preprocess the data, and optimize performance with minimal manual tuning.Using Streamlit, Pandas Profiling and Pycaret.")
    if st.button('Dataset'):
        st.session_state.page="Dataset"
    if st.button('Dataset Profiling'):
        st.session_state.page="Profiling"
    if st.button('ML Model Training'):
        st.session_state.page="Model"
    if st.button('Download Model'):
        st.session_state.page="Download"
if os.path.exists("data.csv"):
    df=pd.read_csv("data.csv", index_col=None)
if st.session_state.page=="Dataset":
    st.title("Upload Your CSV File")
    file=st.file_uploader("Upload your Dataset")
    if file:
        df=pd.read_csv(file,index_col=None, encoding='latin1')
        df.to_csv("data.csv",index=None)
        st.dataframe(df)
if st.session_state.page=="Profiling":
    st.title("Automated Exploratory Data Analysis")
    profile_report=df.profile_report()
    st_profile_report(profile_report)
if st.session_state.page=="Model":
    st.title("ML Model Training")
    model_type=st.selectbox("Select The Type Of Model",["Classification","Regression","Clustering"])
    if model_type == "Classification":
        st.subheader("Classification Model Setup")
        target=st.selectbox("Select Target Column", df.columns)
        cla_setup=setup(data=df,target=target)
        if st.button("Compare Models"):
            best_model=compare_models()
            st.write(best_model)
            st.session_state.model=best_model
        if st.button("Save model"):
            save_model(st.session_state.model,'model')
    if model_type == "Regression":
        st.subheader("Regression Model Setup")
        target=st.selectbox("Select Target Column", df.columns)
        reg_setup=r_setup(data=df,target=target)
        if st.button("Compare Models"):
            best_model=r_compare_models()
            st.write(best_model)
            st.session_state.model=best_model
        if st.button("Save model"):
            save_model(st.session_state.model,'model')
    if model_type == "Clustering":
        st.subheader("Clustering Model Setup")
        
        if st.button("Create Clustering Models"):
            clu_model = c_create_model('kmeans')
            best_model=predict_model(clu_model,data=df)
            st.write(best_model)
            st.session_state.model=best_model
        if st.button("Save model"):
            save_model(st.session_state.model,'model')
if st.session_state.page=="Download":
    with open("model.pkl",'rb') as f:
        st.title("Download Your Pretrained Model")
        st.download_button("Download Model", f ,"model.pkl")
