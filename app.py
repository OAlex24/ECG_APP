
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ECG Classification System",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* ============================================================
   GENERAL
   ============================================================ */

.stApp {
    background-color: #F5F8FA;
}

.main {
    color: #172B4D;
}


/* ============================================================
   HEADER
   ============================================================ */

.main-header {
    background: linear-gradient(135deg, #12355B, #176B87);
    padding: 30px 35px;
    border-radius: 14px;
    margin-bottom: 25px;
    color: white;
    box-shadow: 0 4px 12px rgba(18, 53, 91, 0.15);
}

.main-header-title {
    color: white;
    font-size: 34px;
    font-weight: 700;
    margin-bottom: 8px;
}

.main-header-subtitle {
    color: #E8F4F7;
    font-size: 17px;
    margin-bottom: 14px;
}

.research-badge {
    display: inline-block;
    background-color: rgba(255,255,255,0.15);
    color: white;
    padding: 7px 14px;
    border-radius: 20px;
    font-size: 13px;
    border: 1px solid rgba(255,255,255,0.25);
}


/* ============================================================
   SECTION TITLES
   ============================================================ */

.section-title {
    color: #12355B;
    font-size: 23px;
    font-weight: 700;
    margin-top: 28px;
    margin-bottom: 15px;
    border-left: 5px solid #1B8798;
    padding-left: 12px;
}


/* ============================================================
   RESEARCH CARD
   ============================================================ */

.research-card {
    background-color: white;
    padding: 24px;
    border-radius: 12px;
    border: 1px solid #DCE6EA;
    border-top: 4px solid #176B87;
    box-shadow: 0 2px 8px rgba(18, 53, 91, 0.06);
    margin-bottom: 20px;
}

.research-title {
    color: #12355B;
    font-size: 21px;
    font-weight: 700;
    margin-bottom: 18px;
    line-height: 1.4;
}

.research-item {
    color: #425466;
    font-size: 15px;
    margin-bottom: 9px;
}

.research-label {
    color: #176B87;
    font-weight: 700;
}


/* ============================================================
   ABOUT CARD
   ============================================================ */

.about-card {
    background-color: white;
    padding: 22px;
    border-radius: 12px;
    border: 1px solid #DCE6EA;
    box-shadow: 0 2px 8px rgba(18, 53, 91, 0.06);
    margin-bottom: 20px;
}

.about-text {
    color: #425466;
    font-size: 15px;
    line-height: 1.7;
    margin-bottom: 12px;
}


/* ============================================================
   UPLOAD CARD
   ============================================================ */

.upload-card {
    background-color: #EAF4F6;
    padding: 22px;
    border-radius: 12px;
    border: 1px solid #C9E0E5;
    margin-bottom: 15px;
}

.upload-title {
    color: #12355B;
    font-size: 21px;
    font-weight: 700;
    margin-bottom: 8px;
}

.upload-description {
    color: #425466;
    font-size: 15px;
}


/* ============================================================
   STEP CARDS
   ============================================================ */

.step-card {
    background-color: white;
    padding: 18px;
    border-radius: 10px;
    border: 1px solid #DCE6EA;
    min-height: 135px;
    box-shadow: 0 2px 6px rgba(18, 53, 91, 0.04);
}

.step-number {
    color: #176B87;
    font-size: 20px;
    font-weight: 700;
}

.step-title {
    color: #12355B;
    font-size: 17px;
    font-weight: 700;
    margin: 7px 0;
}

.step-description {
    color: #66788A;
    font-size: 14px;
    line-height: 1.5;
}


/* ============================================================
   RESULT CARDS
   ============================================================ */

.result-card {
    background-color: white;
    padding: 25px;
    border-radius: 12px;
    border: 1px solid #DCE6EA;
    box-shadow: 0 3px 10px rgba(18, 53, 91, 0.07);
    text-align: center;
    min-height: 130px;
}

.result-label {
    color: #66788A;
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 10px;
}

.result-value {
    color: #12355B;
    font-size: 30px;
    font-weight: 700;
}

.confidence-value {
    color: #16837F;
    font-size: 30px;
    font-weight: 700;
}


/* ============================================================
   SIDEBAR
   ============================================================ */

[data-testid="stSidebar"] {
    background-color: #12355B;
}

[data-testid="stSidebar"] * {
    color: white;
}

.sidebar-title {
    color: white;
    font-size: 22px;
    font-weight: 700;
}

.sidebar-subtitle {
    color: #B9DCE2;
    font-size: 13px;
    margin-bottom: 20px;
}

.sidebar-divider {
    border-top: 1px solid rgba(255,255,255,0.20);
    margin: 20px 0;
}


/* ============================================================
   BUTTON
   ============================================================ */

.stButton > button {
    background-color: #176B87;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 25px;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #12355B;
    color: white;
}


/* ============================================================
   FILE UPLOADER
   ============================================================ */

[data-testid="stFileUploader"] {
    background-color: white;
    border-radius: 10px;
    padding: 10px;
    border: 1px solid #DCE6EA;
}


/* ============================================================
   FOOTER
   ============================================================ */

.footer {
    margin-top: 40px;
    padding: 22px;
    text-align: center;
    color: #66788A;
    border-top: 1px solid #DCE6EA;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = Path(__file__).parent / "best_cnn_lstm_v2.keras"


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-header">

        <div class="main-header-title">
            ❤️ ECG CLASSIFICATION SYSTEM
        </div>

        <div class="main-header-subtitle">
            AI-Based ECG Signal Classification Using an Optimized CNN-LSTM Model
        </div>

        <div class="research-badge">
            MSc Computer Science Research Project
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">ECG Research System</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-subtitle">Optimized CNN-LSTM Classification</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-divider"></div>',
        unsafe_allow_html=True
    )

    st.markdown("### System Information")

    st.write("**Model:** Optimized CNN-LSTM")
    st.write("**Input:** 187 ECG signal values")
    st.write("**Classes:** 5")
    st.write("**Framework:** TensorFlow / Keras")

    st.markdown(
        '<div class="sidebar-divider"></div>',
        unsafe_allow_html=True
    )

    st.markdown("### Researcher")

    st.write("**OLAFEMIWA Alex Omoniyi**")
    st.write("Matric: PG/CSC/2024/214")

    st.markdown(
        '<div class="sidebar-divider"></div>',
        unsafe_allow_html=True
    )

    st.caption("Research prototype • Academic demonstration only")


# ============================================================
# RESEARCH PROJECT
# ============================================================

st.markdown(
    '<div class="section-title">Research Project</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="research-card">',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="research-title">
        Development of an Hybrid CNN-LSTM Models for ECG-Based
        Heart Disease Diagnosis in Ondo State, Nigeria
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="research-item">
        <span class="research-label">Researcher:</span>
        OLAFEMIWA Alex Omoniyi
    </div>

    <div class="research-item">
        <span class="research-label">Matric Number:</span>
        PG/CSC/2024/214
    </div>

    <div class="research-item">
        <span class="research-label">Programme:</span>
        Master of Science (MSc) in Computer Science
    </div>

    <div class="research-item">
        <span class="research-label">Department:</span>
        Department of Computer Science
    </div>

    <div class="research-item">
        <span class="research-label">Faculty:</span>
        Faculty of Science
    </div>

    <div class="research-item">
        <span class="research-label">Institution:</span>
        Westley University, Ondo, Nigeria
    </div>

    <div class="research-item">
        <span class="research-label">Supervisor:</span>
        Dr. Makinde
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# ABOUT THE SYSTEM
# ============================================================

st.markdown(
    '<div class="section-title">About the System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="about-card">',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="about-text">
        This research prototype uses an optimized
        <strong>CNN-LSTM deep learning model</strong>
        to classify ECG signals into five classes.
    </div>

    <div class="about-text">
        The CNN component extracts important patterns from
        the ECG waveform, while the LSTM component captures
        sequential information within the signal.
    </div>

    <div class="about-text">
        The system accepts one ECG signal containing exactly
        <strong>187 numerical values</strong> and returns a
        predicted class together with the model's confidence
        and class probabilities.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# HOW THE SYSTEM WORKS
# ============================================================

st.markdown(
    '<div class="section-title">How the System Works</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.markdown(
        """
        <div class="step-card">

            <div class="step-number">01</div>

            <div class="step-title">
                Upload
            </div>

            <div class="step-description">
                Upload a CSV file containing one ECG signal
                with 187 numerical values.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        """
        <div class="step-card">

            <div class="step-number">02</div>

            <div class="step-title">
                Preprocess
            </div>

            <div class="step-description">
                The signal is validated and reshaped into
                the required model input format.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        """
        <div class="step-card">

            <div class="step-number">03</div>

            <div class="step-title">
                Classify
            </div>

            <div class="step-description">
                The optimized CNN-LSTM model processes
                the ECG signal.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with col4:

    st.markdown(
        """
        <div class="step-card">

            <div class="step-number">04</div>

            <div class="step-title">
                Display
            </div>

            <div class="step-description">
                The predicted class, confidence and
                probability distribution are displayed.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# ECG SIGNAL CLASSIFICATION
# ============================================================

st.markdown(
    '<div class="section-title">ECG Signal Classification</div>',
    unsafe_allow_html=True
)


# ============================================================
# UPLOAD INFORMATION
# ============================================================

st.markdown(
    '<div class="upload-card">',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="upload-title">
        📁 Upload ECG Signal
    </div>

    <div class="upload-description">
        Upload a CSV file containing one ECG signal with
        exactly <strong>187 numerical values</strong>.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# FILE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "Choose an ECG CSV file",
    type=["csv"],
    help="The uploaded file must contain exactly 187 numerical ECG signal values."
)


# ============================================================
# ECG PROCESSING
# ============================================================

if uploaded_file is not None:

    try:

        # Read CSV
        df = pd.read_csv(
            uploaded_file,
            header=None
        )

        # Flatten the data
        values = df.values.flatten()

        # Convert to numeric
        values = pd.to_numeric(
            pd.Series(values),
            errors="coerce"
        ).dropna().values

        # Validate number of ECG values
        if len(values) != 187:

            st.error(
                f"Invalid ECG signal. The system requires exactly "
                f"187 numerical values, but {len(values)} values were detected."
            )

        else:

            st.success(
                "✓ ECG signal successfully loaded and validated."
            )


            # =================================================
            # ECG WAVEFORM
            # =================================================

            st.markdown(
                '<div class="section-title">ECG Waveform</div>',
                unsafe_allow_html=True
            )

            fig, ax = plt.subplots(
                figsize=(12, 3.5)
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

            st.pyplot(
                fig
            )


            # =================================================
            # PREDICTION
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

                # Predict
                prediction = model.predict(
                    input_signal,
                    verbose=0
                )[0]

                # Get predicted class
                predicted_class = int(
                    np.argmax(prediction)
                )

                # Get confidence
                confidence = float(
                    prediction[predicted_class]
                ) * 100


                # =============================================
                # PREDICTION RESULT
                # =============================================

                st.markdown(
                    '<div class="section-title">Prediction Result</div>',
                    unsafe_allow_html=True
                )

                result_col1, result_col2 = st.columns(2)


                with result_col1:

                    st.markdown(
                        f"""
                        <div class="result-card">

                            <div class="result-label">
                                PREDICTED CLASS
                            </div>

                            <div class="result-value">
                                Class {predicted_class}
                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                with result_col2:

                    st.markdown(
                        f"""
                        <div class="result-card">

                            <div class="result-label">
                                MODEL CONFIDENCE
                            </div>

                            <div class="confidence-value">
                                {confidence:.2f}%
                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                # =============================================
                # PROBABILITY DISTRIBUTION
                # =============================================

                st.markdown(
                    '<div class="section-title">Class Probability Distribution</div>',
                    unsafe_allow_html=True
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

                    st.bar_chart(
                        probability_df.set_index(
                            "Class"
                        )
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

                st.markdown(
                    '<div class="section-title">System Interpretation</div>',
                    unsafe_allow_html=True
                )

                st.info(
                    f"The optimized CNN-LSTM model classified "
                    f"the uploaded ECG signal as **Class "
                    f"{predicted_class}** with a confidence "
                    f"of **{confidence:.2f}%**."
                )

                st.caption(
                    "Class labels are presented as Class 0–Class 4 "
                    "in this research prototype. This application "
                    "is intended for academic research and "
                    "demonstration only and should not be used "
                    "as a clinical diagnostic tool."
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

st.markdown("---")

with st.expander("🎓 About This Research"):

    st.markdown("### Thesis Information")

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

    st.markdown("### Model")

    st.write(
        "The system uses an optimized hybrid CNN-LSTM architecture."
    )

    st.write(
        "The CNN layers extract spatial patterns from ECG signals, "
        "while the Bidirectional LSTM captures sequential "
        "relationships in the extracted features."
    )

    st.markdown("### Model Performance")

    st.write("• Accuracy: **97.83%**")
    st.write("• Precision: **98.09%**")
    st.write("• Recall: **97.83%**")
    st.write("• F1-Score: **97.93%**")
    st.write("• 5-Fold Cross-Validation Mean Accuracy: **98.45%**")

    st.markdown("### Important Notice")

    st.info(
        "This application is a research prototype for academic "
        "demonstration. It is not intended to replace professional "
        "medical examination, clinical diagnosis, or medical "
        "decision-making."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        <strong>ECG Classification System</strong><br><br>

        OLAFEMIWA Alex Omoniyi • PG/CSC/2024/214<br>

        MSc Computer Science • Westley University, Ondo, Nigeria<br><br>

        Research Prototype • Optimized CNN-LSTM Model

    </div>
    """,
    unsafe_allow_html=True
)
