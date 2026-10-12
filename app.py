"""Streamlit interface for the crop recommendation prototype.

Run from the project root:  streamlit run app.py
"""
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from data_validation import FEATURE_LABELS, FEATURES, INPUT_RANGES, validate_inputs  # noqa: E402
from prediction import (  # noqa: E402
    ModelLoadError,
    ModelNotFoundError,
    load_metadata,
    load_model,
    out_of_training_range,
    predict_row,
)

MODELS_DIR = ROOT / "models"

st.set_page_config(page_title="Crop Recommendation (SML prototype)", page_icon="🌾")
st.title("🌾 Crop Recommendation System")
st.caption(
    "Educational decision-support prototype built for a Statistical Machine Learning project. "
    "It is not a substitute for local agronomic advice."
)


@st.cache_resource
def get_model():
    return load_model(MODELS_DIR)


@st.cache_data
def get_metadata():
    return load_metadata(MODELS_DIR)


try:
    pipeline = get_model()
except (ModelNotFoundError, ModelLoadError) as exc:
    st.error(str(exc))
    st.stop()

metadata = get_metadata()
features = metadata.get("features", FEATURES)

with st.sidebar:
    st.header("About the model")
    if metadata:
        st.write(f"**Model:** {metadata.get('selected_model', 'unknown')}")
        if "test_accuracy" in metadata:
            st.write(f"Held-out test accuracy: {metadata['test_accuracy']:.3f}")
            st.write(f"Held-out test macro-F1: {metadata['test_macro_f1']:.3f}")
        st.write(f"Crop classes: {len(metadata.get('classes', []))}")
        st.caption(
            "Scores come from the project's held-out test split. They describe this dataset, "
            "not performance on real farms."
        )
    else:
        st.info("Training metadata not found; showing predictions only.")

st.subheader("Enter measurements")
st.caption("All fields are required. Hints marked 'e.g.' are examples only, not defaults.")

examples = {"N": "90", "P": "42", "K": "43", "temperature": "21", "humidity": "82", "ph": "6.5", "rainfall": "200"}

with st.form("prediction_form"):
    left, right = st.columns(2)
    raw = {}
    for i, feature in enumerate(features):
        lo, hi = INPUT_RANGES[feature]
        column = left if i % 2 == 0 else right
        raw[feature] = column.text_input(
            FEATURE_LABELS[feature],
            value="",
            placeholder=f"e.g. {examples[feature]}",
            help=f"Accepted range: {lo:g} to {hi:g}",
            key=f"in_{feature}",
        )
    submitted = st.form_submit_button("Recommend a crop", type="primary")

if submitted:
    values, errors = validate_inputs(raw)
    if errors:
        for message in errors:
            st.error(message)
    else:
        label, probabilities = predict_row(pipeline, values, features)
        for message in out_of_training_range(values, metadata):
            st.warning(message)
        st.success(f"Predicted crop: **{label}**")
        if probabilities is not None:
            st.write("Top model probabilities")
            top = probabilities.head(5).rename("probability").to_frame()
            st.bar_chart(top)
            st.caption(
                "These are the model's estimated class probabilities from patterns in the training "
                "data. They are not a guarantee of real-world success."
            )
        st.info(
            "This result reflects patterns in the training dataset only. Verify it locally "
            "(soil tests, local climate, agronomist advice) before any practical use."
        )
