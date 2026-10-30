PROJECT TITLE
Design and Implementation of a Machine Learning-Based Malaria Risk Prediction System Using Clinical Symptoms

IMPORTANT
This package is a runnable application scaffold, not a hosted public website and not a clinically validated medical tool. It does not include fabricated patient data or pre-filled performance claims. You must supply an appropriately sourced dataset with documented labels before training a model.

REQUIRED DATA FILE
Place your CSV at:
data/malaria_dataset

Required columns:
Fever, Headache, Chills, Sweating, Vomiting, Nausea, Weakness, Muscle_Pain, Malaria_Status

Feature values must be 0 or 1 (or blank for missing symptoms).
Malaria_Status must be 0 or 1 and should be supported by a credible reference standard. Symptoms alone are not a reliable confirmation of malaria.

SETUP (Windows)
1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Create an environment:
   python -m venv .venv
4. Activate it in Command Prompt:
   .venv\Scripts\activate.bat
   (PowerShell alternative: .\.venv\Scripts\Activate.ps1)
5. Install packages:
   python -m pip install -r requirements.txt
6. Add your documented dataset to data/malaria_dataset.csv.
7. Train and evaluate models:
   python train_model.py
8. Launch the web app:
   python -m streamlit run app.py
9. Open the local address Streamlit displays, usually http://localhost:8501.

OUTPUTS
- models/malaria_risk_model.joblib: selected model bundle
- outputs/model_comparison_results.csv: model evaluation metrics from your dataset

RESEARCH CAUTION
The included training script uses a basic held-out split and selects a baseline model by F1 score. For a stronger dissertation evaluation, select/tune models using cross-validation within training data and evaluate the chosen model on a separate untouched test set. Review sensitivity, specificity, class balance, leakage, repeated-person records, and independent validation.

PRIVACY
Do not put identifiable patient data in this project folder or share a dataset unless you have the appropriate rights and safeguards.

HOSTING
The app runs locally on your computer using Streamlit. It is not automatically published to the internet by this ZIP. Public deployment requires a hosting service and careful review of privacy, security, reliability, and clinical limitations.
