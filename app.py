from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "models" / "malaria_risk_model.joblib"
RESULTS_FILE = ROOT / "outputs" / "model_comparison_results.csv"

st.set_page_config(
    page_title="Malaria Risk Prediction",
    page_icon="🩺",
    layout="centered"
)

st.title("Malaria Risk Prediction System")
st.subheader("Machine-learning prototype using selected symptoms")
st.warning(
    "Academic prototype only. This application does not diagnose malaria. "
    "Symptoms overlap with other illnesses; seek professional assessment and "
    "appropriate testing if malaria is suspected."
)

with st.sidebar:
    st.header("About this system")
    st.write("Input: selected symptom responses")
    st.write("Output: classification from a trained model")
    st.write("Purpose: educational demonstration and research prototype")
    st.caption("Not validated for clinical use.")

if not MODEL_FILE.exists():
    st.error("No trained model found yet.")
    st.write("To set up the application:")
    st.code("python train_model.py", language="bash")
    st.write(
        "First place an appropriately sourced dataset at "
        "`data/malaria_dataset.csv`. See README.txt for the required columns."
    )
    st.stop()

@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_FILE)

bundle = load_bundle()
model = bundle["model"]
features = bundle["features"]

st.caption(f"Loaded model: {bundle.get('model_name', 'Saved classifier')}")

st.header("Symptom information")
st.write("Choose Yes or No for each symptom.")

with st.form("symptom_form"):
    responses = {}
    labels = {
        "Fever": "Fever",
        "Headache": "Headache",
        "Chills": "Chills",
        "Sweating": "Sweating",
        "Vomiting": "Vomiting",
        "Nausea": "Nausea",
        "Weakness": "General weakness",
        "Muscle_Pain": "Muscle pain"
    }
    for feature in features:
        responses[feature] = st.radio(
            labels.get(feature, feature),
            ["No", "Yes"],
            horizontal=True,
            key=feature
        )
    submitted = st.form_submit_button("Generate model output", type="primary")

if submitted:
    values = {name: int(value == "Yes") for name, value in responses.items()}
    input_df = pd.DataFrame([[values[name] for name in features]], columns=features)
    try:
        result = int(model.predict(input_df)[0])
        st.header("Model output")
        if result == 1:
            st.info("The model returned the positive class (label 1).")
        else:
            st.info("The model returned the negative class (label 0).")
        st.write(
            "This is only the model's classification, not a confirmed infection "
            "status or a reliable individual clinical risk estimate."
        )
        st.subheader("Submitted symptom responses")
        st.dataframe(
            pd.DataFrame(
                [{"Symptom": labels.get(k, k), "Response": v}
                 for k, v in responses.items()]
            ),
            hide_index=True,
            use_container_width=True
        )
        st.subheader("Next step")
        st.write(
            "If malaria is suspected, consult a qualified healthcare professional "
            "and obtain appropriate testing. Do not use this output alone to "
            "decide whether treatment is needed."
        )
    except Exception as exc:
        st.error("The application could not process this input.")
        st.exception(exc)

if RESULTS_FILE.exists():
    with st.expander("View model comparison (developer/research output)"):
        try:
            results = pd.read_csv(RESULTS_FILE)
            st.dataframe(results, hide_index=True, use_container_width=True)
        except Exception:
            st.caption("The results file could not be read.")

st.divider()
st.caption("PGD Computer Science project prototype | No clinical validation implied.")
