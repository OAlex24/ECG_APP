
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="ECG Classification System",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# COLOUR THEME
# ============================================================

NAVY = "#12355B"
TEAL = "#176B87"
LIGHT_BLUE = "#EAF4F6"
LIGHT_GREY = "#F5F8FA"
BORDER = "#DCE6EA"
TEXT = "#425466"


# ============================================================
# SIMPLE CSS
# No HTML content is used for the application sections.
# CSS is only used for visual styling.
# ============================================================

st.markdown(
    f"""
    <style>

    .stApp {{
        background-color: {LIGHT_GREY};
    }}

    /* Main headings */
    h1 {{
        color: {NAVY};
    }}

    h2 {{
        color: {NAVY};
    }}

    h3 {{
        color: {NAVY};
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {NAVY};
    }}

    [data-testid="stSidebar"] * {{
        color: white;
    }}

    /* Buttons */
    .stButton > button {{
        background-color: {TEAL};
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 10px 20px;
    }}

    .stButton > button:hover {{
        background-color: {NAVY};
        color: white;
    }}

    /* Upload box */
    [data-testid="stFileUploader"] {{
        background-color: white;
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 12px;
    }}

    /* Metric cards */
    [data-testid="stMetric"] {{
        background-color: white;
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 18px;
    }}

    /* Expander */
    [data-testid="stExpander"] {{
        border: 1px solid {BORDER};
        border-radius: 10px;
        background-color: white;
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

MODEL_PATH = Path(__file__).parent / "best_cnn_lstm_v2.keras"


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()


# ============================================================
# HEADER
# ============================================================

st.title("❤️ ECG CLASSIFICATION SYSTEM")

st.subheader(
    "AI-Based ECG Signal Classification Using an Optimized CNN-LSTM Model"
)

st.info(
    "MSc Computer Science Research Project"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("ECG Research System")

    st.caption(
        "Optimized CNN-LSTM Classification"
    )

    st.divider()

    st.subheader("System Information")

    st.write("**Model:** Optimized CNN-LSTM")
    st.write("**Input:** 187 ECG signal values")
    st.write("**Classes:** 5")
    st.write("**Framework:** TensorFlow / Keras")

    st.divider()

    st.subheader("Researcher")

    st.write("**OLAFEMIWA Alex Omoniyi**")
    st.write("**Matric:** PG/CSC/2024/214")

    st.divider()

    st.caption(
        "Research prototype • Academic demonstration only"
    )


# ============================================================
# RESEARCH PROJECT
# ============================================================

st.header("Research Project")

st.subheader(
    "Development of an Hybrid CNN-LSTM Models for "
    "ECG-Based Heart Disease Diagnosis in Ondo State, Nigeria"
)

research_col1, research_col2 = st.columns(2)

with research_col1:

    st.write("**Researcher**")
    st.write("OLAFEMIWA Alex Omoniyi")

    st.write("**Matric Number**")
    st.write("PG/CSC/2024/214")

    st.write("**Programme**")
    st.write("Master of Science (MSc) in Computer Science")

    st.write("**Department**")
    st.write("Department of Computer Science")


with research_col2:

    st.write("**Faculty**")
    st.write("Faculty of Science")

    st.write("**Institution**")
    st.write("Westley University, Ondo, Nigeria")

    st.write("**Supervisor**")
    st.write("Dr. Makinde")


st.divider()


# ============================================================
# ABOUT THE SYSTEM
# ============================================================

st.header("About the System")

st.write(
    "This research prototype uses an optimized "
    "CNN-LSTM deep learning model to classify ECG "
    "signals into five classes."
)

st.write(
    "The CNN component extracts important patterns "
    "from the ECG waveform, while the LSTM component "
    "captures sequential information within the signal."
)

st.write(
    "The system accepts one ECG signal containing "
    "exactly 187 numerical values and returns a "
    "predicted class together with the model's "
    "confidence and class probabilities."
)


# ============================================================
# HOW THE SYSTEM WORKS
# ============================================================

st.header("How the System Works")

step1, step2, step3, step4 = st.columns(4)

with step1:

    st.subheader("01 • Upload")

    st.write(
        "Upload a CSV file containing one ECG signal "
        "with 187 numerical values."
    )


with step2:

    st.subheader("02 • Preprocess")

    st.write(
        "The signal is validated and reshaped into "
        "the required model input format."
    )


with step3:

    st.subheader("03 • Classify")

    st.write(
        "The optimized CNN-LSTM model processes "
        "the ECG signal."
    )


with step4:

    st.subheader("04 • Display")

    st.write(
        "The predicted class, confidence and "
        "probability distribution are displayed."
    )


# ============================================================
# ECG CLASSIFICATION
# ============================================================

st.header("ECG Signal Classification")

st.subheader("📁 Upload ECG Signal")

st.write(
    "Upload a CSV file containing one ECG signal "
    "with exactly 187 numerical values."
)


uploaded_file = st.file_uploader(
    "Choose an ECG CSV file",
    type=["csv"],
    help=(
        "The uploaded file must contain exactly "
        "187 numerical ECG signal values."
    )
)


# ============================================================
# PROCESS UPLOADED ECG
# ============================================================

if uploaded_file is not None:

    try:

        # Read CSV
        df = pd.read_csv(
            uploaded_file,
            header=None
        )

        # Flatten all values
        values = df.values.flatten()

        # Convert to numeric
        values = pd.to_numeric(
            pd.Series(values),
            errors="coerce"
        ).dropna().values

        # ====================================================
        # VALIDATE
        # ====================================================

        if len(values) != 187:

            st.error(
                f"Invalid ECG signal. The system requires "
                f"exactly 187 numerical values, but "
                f"{len(values)} values were detected."
            )

        else:

            st.success(
                "✓ ECG signal successfully loaded and validated."
            )

            # =================================================
            # ECG WAVEFORM
            # =================================================

            st.header("ECG Waveform")

            fig, ax = plt.subplots(
                figsize=(12, 4)
            )

            ax.plot(
                values,
                linewidth=1.5
            )

            ax.set_xlabel("Sample")
            ax.set_ylabel("Amplitude")

            ax.grid(
                alpha=0.25
            )

            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)


            # =================================================
            # PREDICT
            # =================================================

            if st.button(
                "🔍 Predict ECG",
                use_container_width=True
            ):

                # Reshape input
                input_signal = values.astype(
                    np.float32
                ).reshape(
                    1,
                    187,
                    1
                )

                # Model prediction
                prediction = model.predict(
                    input_signal,
                    verbose=0
                )[0]

                # Predicted class
                predicted_class = int(
                    np.argmax(prediction)
                )

                # Confidence
                confidence = (
                    float(
                        prediction[predicted_class]
                    ) * 100
                )


                # =============================================
                # RESULT
                # =============================================

                st.header("Prediction Result")

                result1, result2 = st.columns(2)

                with result1:

                    st.metric(
                        "Predicted Class",
                        f"Class {predicted_class}"
                    )

                with result2:

                    st.metric(
                        "Model Confidence",
                        f"{confidence:.2f}%"
                    )


                # =============================================
                # PROBABILITY DISTRIBUTION
                # =============================================

                st.header(
                    "Class Probability Distribution"
                )

                probability_df = pd.DataFrame(
                    {
                        "Class": [
                            f"Class {i}"
                            for i in range(5)
                        ],
                        "Probability (%)":
                            prediction * 100
                    }
                )


                chart_col, table_col = st.columns(
                    [2, 1]
                )


                with chart_col:

                    chart_data = probability_df.set_index(
                        "Class"
                    )

                    st.bar_chart(
                        chart_data
                    )


                with table_col:

                    display_df = probability_df.copy()

                    display_df[
                        "Probability (%)"
                    ] = display_df[
                        "Probability (%)"
                    ].round(2)

                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        hide_index=True
                    )


                # =============================================
                # INTERPRETATION
                # =============================================

                st.header(
                    "System Interpretation"
                )

                st.info(
                    f"The optimized CNN-LSTM model classified "
                    f"the uploaded ECG signal as **Class "
                    f"{predicted_class}** with a confidence "
                    f"of **{confidence:.2f}%**."
                )

                st.caption(
                    "Class labels are presented as Class 0–Class 4 "
                    "in this research prototype."
                )


    except Exception as error:

        st.error(
            "Unable to process the uploaded file. "
            "Please ensure that the CSV contains exactly "
            "187 numerical ECG signal values."
        )


# ============================================================
# ABOUT THE RESEARCH
# ============================================================

st.divider()

with st.expander("🎓 About This Research"):

    st.subheader("Thesis Information")

    st.write(
        "**Title:** Development of an Hybrid CNN-LSTM Models "
        "for ECG-Based Heart Disease Diagnosis in Ondo State, Nigeria"
    )

    st.write(
        "**Researcher:** OLAFEMIWA Alex Omoniyi"
    )

    st.write(
        "**Matric Number:** PG/CSC/2024/214"
    )

    st.write(
        "**Programme:** Master of Science (MSc) in Computer Science"
    )

    st.write(
        "**Department:** Department of Computer Science"
    )

    st.write(
        "**Faculty:** Faculty of Science"
    )

    st.write(
        "**Institution:** Westley University, Ondo, Nigeria"
    )

    st.write(
        "**Supervisor:** Dr. Makinde"
    )

    st.subheader("Model")

    st.write(
        "The system uses an optimized hybrid CNN-LSTM architecture."
    )

    st.write(
        "The CNN layers extract spatial patterns from ECG signals, "
        "while the Bidirectional LSTM captures sequential "
        "relationships in the extracted features."
    )

    st.subheader("Model Performance")

    performance_col1, performance_col2 = st.columns(2)

    with performance_col1:

        st.write("Accuracy: **97.83%**")
        st.write("Precision: **98.09%**")
        st.write("Recall: **97.83%**")

    with performance_col2:

        st.write("F1-Score: **97.93%**")
        st.write(
            "5-Fold CV Mean Accuracy: **98.45%**"
        )

    st.subheader("Important Notice")

    st.warning(
        "This application is a research prototype for "
        "academic demonstration. It is not intended to "
        "replace professional medical examination, "
        "clinical diagnosis, or medical decision-making."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "ECG Classification System | "
    "OLAFEMIWA Alex Omoniyi | "
    "PG/CSC/2024/214 | "
    "MSc Computer Science | "
    "Wesley University, Ondo, Nigeria"
)
