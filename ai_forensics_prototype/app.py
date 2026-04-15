"""Streamlit entry point for the AI forensics prototype."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from feature_extraction import extract_features
from model import explain_prediction, predict_text, train_baseline_model


EXAMPLE_TEXTS = {
    "Phishing Email": (
        "Dear employee, our security team detected unusual login activity on your "
        "company mailbox. To prevent account suspension, confirm your password and "
        "recovery details through the secure verification link below."
    ),
    "Crypto Scam": (
        "Your wallet has been temporarily restricted due to an automated compliance "
        "review. Please submit identity confirmation and validate the attached "
        "transaction form within the next fifteen minutes."
    ),
    "Suspicious Chat": (
        "hey can you send the gift cards now, i told you my phone is broken and i "
        "cannot log in. just send the codes here first and i will pay you back later"
    ),
}


@st.cache_resource
def load_model():
    """Train the baseline model once and cache it for the session."""
    return train_baseline_model()


def format_feature_table(features: dict) -> pd.DataFrame:
    """Convert the feature dictionary into a display-friendly table."""
    rows = [
        ("Average sentence length", round(features["avg_sentence_length"], 3)),
        ("Average word length", round(features["avg_word_length"], 3)),
        ("Lexical diversity", round(features["lexical_diversity"], 3)),
        ("Repetition ratio", round(features["repetition_ratio"], 3)),
        ("Entropy estimate", round(features["entropy"], 3)),
    ]
    return pd.DataFrame(rows, columns=["Feature", "Value"])


def build_indicator_chart(features: dict) -> pd.DataFrame:
    """Create a small normalized chart for the extracted indicators."""
    chart_values = {
        "Sentence Length": min(features["avg_sentence_length"] / 20.0, 1.0),
        "Word Length": min(features["avg_word_length"] / 6.0, 1.0),
        "Lexical Diversity": min(features["lexical_diversity"], 1.0),
        "Repetition": min(features["repetition_ratio"] / 0.35, 1.0),
        "Entropy": min(features["entropy"] / 5.0, 1.0),
    }
    return pd.DataFrame(
        {"Indicator": list(chart_values.keys()), "Score": list(chart_values.values())}
    ).set_index("Indicator")


def inject_styles():
    """Apply lightweight custom styling to make the app look cleaner."""
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #f4f7fb 0%, #eef3f8 100%);
        }
        .block-container {
            max-width: 900px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .hero-card {
            background: white;
            border: 1px solid #d7e2ee;
            border-radius: 18px;
            padding: 1.25rem 1.4rem;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.07);
            margin-bottom: 1rem;
        }
        .section-card {
            background: white;
            border: 1px solid #d7e2ee;
            border-radius: 16px;
            padding: 1rem 1.2rem;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
            margin-top: 1rem;
        }
        .small-note {
            color: #4b5563;
            font-size: 0.95rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    """Render the Streamlit interface."""
    st.set_page_config(
        page_title="AI Content Forensics Prototype",
        page_icon=":mag:",
        layout="centered",
    )
    inject_styles()

    if "input_text" not in st.session_state:
        st.session_state["input_text"] = ""

    st.markdown(
        """
        <div class="hero-card">
            <h1 style="margin-bottom:0.4rem;">Forensic Analysis of AI-Generated Cybercrime Text</h1>
            <p class="small-note" style="margin-bottom:0;">
                A lightweight student prototype for analyzing suspicious phishing emails,
                scam messages, and chat content using simple forensic-style text indicators.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model_bundle = load_model()

    st.write("**Demo examples**")
    col1, col2, col3 = st.columns(3)
    if col1.button("Load Phishing Email", use_container_width=True):
        st.session_state["input_text"] = EXAMPLE_TEXTS["Phishing Email"]
    if col2.button("Load Crypto Scam", use_container_width=True):
        st.session_state["input_text"] = EXAMPLE_TEXTS["Crypto Scam"]
    if col3.button("Load Suspicious Chat", use_container_width=True):
        st.session_state["input_text"] = EXAMPLE_TEXTS["Suspicious Chat"]

    input_text = st.text_area(
        "Enter suspicious text",
        height=220,
        placeholder=(
            "Paste a phishing email, scam message, or suspicious chat content here..."
        ),
        key="input_text",
    )

    if st.button("Analyze", type="primary"):
        if not input_text.strip():
            st.error("No text was provided. Paste suspicious content or load one of the example cases first.")
            return

        features = extract_features(input_text)
        prediction = predict_text(input_text, model_bundle)
        explanation = explain_prediction(features, prediction["label"])

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Results")
        st.write("**Input text:**")
        st.info(input_text)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Prediction", prediction["label"])
        with col2:
            st.metric("Confidence", f"{prediction['confidence']:.1%}")
        with col3:
            st.metric("AI Probability", f"{prediction['ai_probability']:.1%}")

        chart_col, table_col = st.columns([1.05, 1])
        with chart_col:
            st.subheader("Indicator Overview")
            st.bar_chart(build_indicator_chart(features), height=280)
        with table_col:
            st.subheader("Extracted Features")
            st.table(format_feature_table(features))

        st.subheader("Interpretation")
        st.write(explanation)
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
