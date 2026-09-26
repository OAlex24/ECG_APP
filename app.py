
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

    .main-title {
        text-align: center;
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 25px;
    }

    .research-badge {
        text-align: center;
        font-size: 14px;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 600;
        margin-top: 20px;
        margin-bottom: 12px;
    }

    .result-box {
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        border: 1px solid rgba(128,128,128,0.25);
        margin-top: 10px;
        margin-bottom: 20px;
    }

    .result-class {
        font-size: 34px;
        font-weight: 700;
    }

    .confidence {
        font-size: 22px;
        font-weight: 600;
        margin-top: 8px;
    }

    .academic-box {
        padding: 20px;
        border-radius: 10px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-top: 10px;
        margin-bottom: 15px;
    }

    .footer {
        text-align: center;
        font-size: 13px;
        margin-top: 35px;
        padding-top: 15px;
        border-top: 1px solid rgba(128,128,128,0.25);
    }

</style>
""", unsafe_allow_html=True)

# ============================================================
# MODEL
# ============================================================

MODEL_PATH = "best_cnn_lstm_v2.keras"

@st.cache_resource
def load_ecg_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_ecg_model()

# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">❤️ ECG CLASSIFICATION SYSTEM</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-Based ECG Signal Classification Using an Optimized CNN-LSTM Model</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="research-badge">🎓 MSc Computer Science Research Prototype</div>',
    unsafe_allow_html=True
)

# ============================================================
# RESEARCH TITLE
# ============================================================

st.markdown(
    '<div class="section-title">Research Project</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="academic-box">

<strong>DEVELOPMENT OF AN HYBRID CNN-LSTM MODELS FOR ECG-BASED
HEART DISEASE DIAGNOSIS IN ONDO STATE, NIGERIA</strong>

<br><br>

<strong>Researcher:</strong> OLAFEMIWA Alex Omoniyi<br>
<strong>Matric Number:</strong> PG/CSC/2024/214<br>
<strong>Supervisor:</strong> Dr. Makinde

</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ System Information")

    st.write("**Model:** Optimized CNN-LSTM")
    st.write("**Input:** 187 ECG signal values")
    st.write("**Output:** 5 ECG classes")
    st.write("**Framework:** TensorFlow/Keras")

    st.markdown("---")

    st.subheader("📚 Researcher")

    st.write("**OLAFEMIWA Alex Omoniyi**")
    st.write("MSc Computer Science")
    st.write("Matric: PG/CSC/2024/214")

    st.markdown("---")

    st.subheader("🏛️ Academic Information")

    st.write("Department of Computer Science")
    st.write("Faculty of Science")
    st.write("Postgraduate College")
    st.write("Westley University, Ondo, Nigeria")

    st.markdown("---")

    st.caption(
        "This application was developed as part of an MSc "
        "Computer Science thesis research project."
    )

# ============================================================
# ABOUT THE SYSTEM
# ============================================================

st.markdown(
    '<div class="section-title">About the System</div>',
    unsafe_allow_html=True
)

st.write(
    "This research prototype applies a hybrid Convolutional Neural "
    "Network–Long Short-Term Memory (CNN-LSTM) architecture to the "
    "classification of ECG signals. The CNN component learns important "
    "patterns from the ECG waveform, while the LSTM component captures "
    "sequential information within the signal."
)

st.info(
    "⚠️ Research prototype: This application is developed for academic "
    "research and demonstration purposes and is not intended to replace "
    "professional medical diagnosis or clinical decision-making."
)

# ============================================================
# HOW IT WORKS
# ============================================================

st.markdown(
    '<div class="section-title">How the System Works</div>',
    unsafe_allow_html=True
)

step1, step2, step3, step4 = st.columns(4)

with step1:
    st.markdown("### ①")
    st.write("**Upload ECG**")
    st.caption("Upload a CSV containing 187 ECG signal values.")

with step2:
    st.markdown("### ②")
    st.write("**Process Signal**")
    st.caption("The signal is converted into the model input format.")

with step3:
    st.markdown("### ③")
    st.write("**CNN-LSTM Analysis**")
    st.caption("The trained model extracts ECG patterns.")

with step4:
    st.markdown("### ④")
    st.write("**Classification**")
    st.caption("The model produces the predicted class and confidence.")

# ============================================================
# UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">📁 Upload ECG Signal</div>',
    unsafe_allow_html=True
)

st.write(
    "Upload a CSV file containing exactly **187 numerical ECG signal values**."
)

uploaded_file = st.file_uploader(
    "Choose an ECG CSV file",
    type=["csv"],
    help="The CSV must contain exactly 187 numerical ECG signal values."
)

# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_ecg(signal):

    signal = np.asarray(
        signal,
        dtype=np.float32
    )

    if len(signal) != 187:
        raise ValueError(
            f"The ECG signal must contain exactly 187 values. "
            f"Your file contains {len(signal)} values."
        )

    signal_input = signal.reshape(
        1,
        187,
        1
    )

    probabilities = model.predict(
        signal_input,
        verbose=0
    )[0]

    predicted_class = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[predicted_class]
    )

    return predicted_class, confidence, probabilities


# ============================================================
# PROCESS UPLOADED FILE
# ============================================================

if uploaded_file is not None:

    try:

        data = pd.read_csv(
            uploaded_file,
            header=None
        )

        signal = pd.to_numeric(
            data.values.flatten(),
            errors="coerce"
        )

        if np.isnan(signal).any():

            st.error(
                "❌ The uploaded file contains missing or "
                "non-numerical values."
            )

        elif len(signal) != 187:

            st.error(
                f"❌ Invalid ECG signal length. "
                f"The system requires exactly 187 values, "
                f"but {len(signal)} values were uploaded."
            )

        else:

            st.success(
                "✓ ECG file successfully uploaded and validated."
            )

            # ====================================================
            # WAVEFORM
            # ====================================================

            st.markdown(
                '<div class="section-title">📈 ECG Waveform</div>',
                unsafe_allow_html=True
            )

            fig, ax = plt.subplots(
                figsize=(12, 4)
            )

            ax.plot(
                signal,
                linewidth=1.2
            )

            ax.set_xlabel(
                "ECG Sample"
            )

            ax.set_ylabel(
                "Amplitude"
            )

            ax.set_title(
                "Uploaded ECG Signal"
            )

            ax.grid(
                True,
                alpha=0.3
            )

            st.pyplot(
                fig,
                use_container_width=True
            )

            # ====================================================
            # PREDICT
            # ====================================================

            st.markdown(
                '<div class="section-title">🤖 Classification</div>',
                unsafe_allow_html=True
            )

            if st.button(
                "🔍 Predict ECG",
                type="primary",
                use_container_width=True
            ):

                with st.spinner(
                    "The CNN-LSTM model is analysing the ECG signal..."
                ):

                    predicted_class, confidence, probabilities = (
                        predict_ecg(signal)
                    )

                # =================================================
                # RESULT
                # =================================================

                st.markdown(
                    '<div class="section-title">Prediction Result</div>',
                    unsafe_allow_html=True
                )

                result_col1, result_col2 = st.columns(2)

                with result_col1:

                    st.markdown(
                        f"""
                        <div class="result-box">
                            <div>Predicted ECG Class</div>
                            <div class="result-class">
                                Class {predicted_class}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with result_col2:

                    st.markdown(
                        f"""
                        <div class="result-box">
                            <div>Model Confidence</div>
                            <div class="confidence">
                                {confidence * 100:.2f}%
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # =================================================
                # PROBABILITIES
                # =================================================

                st.markdown(
                    '<div class="section-title">📊 Class Probabilities</div>',
                    unsafe_allow_html=True
                )

                probability_df = pd.DataFrame(
                    {
                        "ECG Class": [
                            f"Class {i}"
                            for i in range(5)
                        ],
                        "Probability (%)": [
                            round(float(p) * 100, 4)
                            for p in probabilities
                        ]
                    }
                )

                st.bar_chart(
                    probability_df.set_index(
                        "ECG Class"
                    )
                )

                st.dataframe(
                    probability_df,
                    use_container_width=True,
                    hide_index=True
                )

                # =================================================
                # INTERPRETATION
                # =================================================

                st.markdown(
                    '<div class="section-title">📋 System Interpretation</div>',
                    unsafe_allow_html=True
                )

                st.write(
                    f"The optimized CNN-LSTM model classified the "
                    f"uploaded ECG signal as **Class {predicted_class}**, "
                    f"with a model confidence of **{confidence * 100:.2f}%**."
                )

                st.caption(
                    "The displayed class number corresponds to the "
                    "classification label used by the research dataset. "
                    "It should not be interpreted independently as a "
                    "medical diagnosis."
                )

    except Exception as e:

        st.error(
            f"❌ Unable to process the uploaded ECG file: {str(e)}"
        )

# ============================================================
# ACADEMIC INFORMATION
# ============================================================

st.markdown("---")

with st.expander("🎓 About This Research"):

    st.markdown("""
### Thesis Information

**Title:**  
**DEVELOPMENT OF AN HYBRID CNN-LSTM MODELS FOR ECG-BASED HEART DISEASE DIAGNOSIS IN ONDO STATE, NIGERIA**

**Researcher:** OLAFEMIWA Alex Omoniyi

**Matric Number:** PG/CSC/2024/214

**Department:** Department of Computer Science

**Faculty:** Faculty of Science

**Institution:** Westley University, Ondo, Nigeria

**Programme:** Master of Science (MSc) in Computer Science

**Supervisor:** Dr. Makinde

### Purpose of the Application

This application provides an interactive demonstration of the
optimized CNN-LSTM model developed as part of the research.

The system accepts an ECG signal, processes the signal and passes
it through the trained deep learning model to obtain a classification
prediction.

The application is intended strictly for academic research,
demonstration and thesis defence purposes.
""")

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        <strong>ECG Classification Research Prototype</strong><br>
        Hybrid CNN-LSTM Deep Learning Model<br><br>
        Developed as part of an MSc Computer Science Thesis<br>
        OLAFEMIWA Alex Omoniyi | PG/CSC/2024/214<br>
        Westley University, Ondo, Nigeria
    </div>
    """,
    unsafe_allow_html=True
)
