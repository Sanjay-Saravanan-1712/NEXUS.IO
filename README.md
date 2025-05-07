🤖 NEXUS.IO — A Simple Yet Powerful AutoML Assistant
Welcome to NEXUS.IO, a beginner-friendly AutoML platform built using Streamlit, scikit-learn, and pandas profiling. This project was created with one goal in mind:

✅ Make machine learning automation simple, transparent, and educational — without relying on black-box tools like PyCaret.

Whether you're working on classification, regression, or clustering, NEXUS.IO helps you:

Explore your data

Train top-performing models

Save them for deployment

All without writing a line of ML code manually.



🎯 Project Highlights
Feature	Description

📂 Upload Dataset	Drag and drop a .csv file directly into the app

📊 Auto EDA	Generates interactive Exploratory Data Analysis using pandas-profiling

🧠 ML Pipelines	Custom-built using scikit-learn pipelines

🔍 Model Selection	Automatically compares models for classification & regression

🧪 Clustering Support	Uses KMeans for unsupervised learning workflows

💾 Model Export	Save trained models as .pkl files

🌐 Web UI	Built entirely using Streamlit for an interactive and clean UX


🛠️ Tech Stack
Layer	Tool
UI	Streamlit
Data Handling	Pandas
AutoEDA	ydata-profiling (pandas-profiling)
ML Pipeline	scikit-learn
Model Export	joblib

💡 Why I Built This
Most AutoML platforms like PyCaret are powerful, but they abstract too much.

I wanted to create something simple yet extensible, where I could:

Understand each step of the ML lifecycle

Customize things as needed

Share it as a base for others starting out with AutoML

NEXUS.IO aims to be your stepping stone — from beginner-level automation to advanced model experimentation.

🧭 How to Use the App

✅ Step 1: Clone the Repository

git clone https://github.com/Sanjay-Saravanan-1712/NEXUS.IO

cd nexus.io

✅ Step 2: Install Requirements
It's best to use a virtual environment.


pip install -r requirements.txt

✅ Step 3: Launch the App

streamlit run app.py


🧪 What’s Inside the ML Pipeline?

Each pipeline follows the same modular steps:

Data Preprocessing

Missing value imputation

Categorical encoding

Feature scaling

Model Selection

A set of models is compared using cross-validation

Best model is selected based on chosen metric

Export

Final model is saved using joblib for reuse or deployment

📌 Supported Models

Task	Algorithms Used

Classification	Logistic Regression, Random Forest, Gradient Boosting

Regression	Linear Regression, Ridge, Random Forest Regressor

Clustering	KMeans (with auto cluster detection via silhouette score)

📁 Folder Structure


nexus.io/

├── automl.py             # Pycaret Streamlit app

├── automl_og.py          # Main Streamlit app

├── data.csv              # Uploaded dataset

├── model.pkl             # Saved model

├── requirements.txt      # Python dependencies

└── README.md             # This file

🔄 Future Scope

I plan to improve and extend NEXUS.IO with:

📈 Visual performance dashboards (e.g., ROC, residuals, silhouette plots)

🧩 Manual hyperparameter tuning

☁️ Cloud deployment (Streamlit Cloud)

🧪 MLFlow / W&B experiment tracking

🤝 Contribution & Feedback

This project is open for learning and collaboration.

If you:

Spot a bug

Have an idea for a feature

Or want to contribute new models

Feel free to open an issue or submit a PR. Let’s make NEXUS.IO better together 🚀

📬 Contact

Built by SANJAY SARAVANAN — CS Undergrad @ Sri Sairam Engineering College

🧑‍💻 GitHub: github.com/Sanjay-Saravanan-1712

📬 Email: sanjaysaravanan171204@example.com

🏆 Looking for SDE / ML internships & full-time roles

🧠 What I Learned

This project wasn’t just about building an app — it was a learning ground. Here's what I explored and understood under the hood:

✅ First time using scikit-learn Pipelines:

I learned how to chain preprocessing and model training steps together using Pipeline() and ColumnTransformer() for cleaner, more modular ML workflows. It’s more 
maintainable and production-ready than writing step-by-step code manually.

🛠️ Dynamic model selection and evaluation:

Implemented a flexible logic to compare multiple models dynamically based on the selected task (classification, regression, clustering).

📈 Data profiling and automatic EDA:

Integrated ydata-profiling (formerly pandas-profiling) to create on-the-fly visual reports for quick dataset understanding.

🌐 Building a real-world UI with Streamlit:

Understood how to convert backend ML pipelines into user-friendly frontends, enabling others to interact with models without touching the code.

⭐ Final Words

NEXUS.IO shows how far you can go with just:

Streamlit for the UI

Pandas for data

scikit-learn for ML

No fluff. Just a clean pipeline that works.
