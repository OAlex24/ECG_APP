
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="ECG Classification System",
    page_icon="❤️",
    layout="wide"
)

# ------------------------------------------------------------
# MODEL PATH
# ------------------------------------------------------------

MODEL_PATH = "best_cnn_lstm_v2.keras"

# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

@st.cache_resource
def load_ecg_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_ecg_model()

# ------------------------------------------------------------
# TITLE
# ------------------------------------------------------------

st.title("❤️ ECG Classification System")

st.write(
    "An AI-based research prototype for ECG signal classification "
    "using an optimized CNN-LSTM deep learning model."
)

st.info(
    "This application is a research/academic demonstration only "
    "and is not intended for clinical diagnosis or medical decision-making."
)

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------

st.sidebar.header("System Information")

st.sidebar.write("**Model:** Optimized CNN-LSTM")
st.sidebar.write("**Input:** 187 ECG signal values")
st.sidebar.write("**Classes:** 5")
st.sidebar.write("**Framework:** TensorFlow/Keras")

st.sidebar.markdown("---")

st.sidebar.write(
    "Upload a CSV containing one ECG signal with exactly "
    "187 numerical values."
)

# ------------------------------------------------------------
# FILE UPLOAD
# ------------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an ECG CSV file",
    type=["csv"]
)

# ------------------------------------------------------------
# PREDICTION FUNCTION
# ------------------------------------------------------------

def predict_ecg(signal):

    # Convert to NumPy array
    signal = np.asarray(signal, dtype=np.float32)

    # Validate number of ECG values
    if len(signal) != 187:
        raise ValueError(
            f"The uploaded ECG must contain exactly 187 values. "
            f"Your file contains {len(signal)} values."
        )

    # Reshape for CNN-LSTM
    signal_input = signal.reshape(1, 187, 1)

    # Model prediction
    probabilities = model.predict(
        signal_input,
        verbose=0
    )[0]

    predicted_class = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_class])

    return predicted_class, confidence, probabilities


# ------------------------------------------------------------
# MAIN APPLICATION
# ------------------------------------------------------------

if uploaded_file is not None:

    try:

        # Read uploaded CSV
        data = pd.read_csv(
            uploaded_file,
            header=None
        )

        # Flatten all values
        signal = pd.to_numeric(
            data.values.flatten(),
            errors="coerce"
        )

        # Check for invalid values
        if np.isnan(signal).any():

            st.error(
                "The uploaded file contains non-numeric or missing values."
            )

        else:

            # Check length
            if len(signal) != 187:

                st.error(
                    f"Invalid ECG length: {len(signal)} values found. "
                    "The system requires exactly 187 values."
                )

            else:

                # ------------------------------------------------
                # DISPLAY WAVEFORM
                # ------------------------------------------------

                st.subheader("ECG Waveform")

                fig, ax = plt.subplots(
                    figsize=(12, 4)
                )

                ax.plot(
                    signal
                )

                ax.set_xlabel(
                    "Sample"
                )

                ax.set_ylabel(
                    "Normalized Amplitude"
                )

                ax.grid(
                    True,
                    alpha=0.3
                )

                st.pyplot(
                    fig
                )

                # ------------------------------------------------
                # PREDICTION BUTTON
                # ------------------------------------------------

                if st.button(
                    "🔍 Predict ECG",
                    type="primary"
                ):

                    predicted_class, confidence, probabilities = (
                        predict_ecg(signal)
                    )

                    st.subheader(
                        "Prediction Result"
                    )

                    col1, col2 = st.columns(2)

                    with col1:

                        st.metric(
                            "Predicted Class",
                            f"Class {predicted_class}"
                        )

                    with col2:

                        st.metric(
                            "Confidence",
                            f"{confidence * 100:.2f}%"
                        )

                    # ------------------------------------------------
                    # CLASS PROBABILITIES
                    # ------------------------------------------------

                    st.subheader(
                        "Model Class Probabilities"
                    )

                    probability_df = pd.DataFrame(
                        {
                            "Class": [
                                f"Class {i}"
                                for i in range(5)
                            ],
                            "Probability": [
                                f"{p * 100:.4f}%"
                                for p in probabilities
                            ]
                        }
                    )

                    st.table(
                        probability_df
                    )

                    # ------------------------------------------------
                    # SIMPLE EXPLANATION
                    # ------------------------------------------------

                    st.subheader(
                        "Interpretation"
                    )

                    st.write(
                        f"The optimized CNN-LSTM model classified "
                        f"the uploaded ECG signal as **Class "
                        f"{predicted_class}** with a confidence of "
                        f"**{confidence * 100:.2f}%**."
                    )

                    st.caption(
                        "The class number represents the classification "
                        "label used in the research dataset. It should "
                        "not be interpreted as a clinical diagnosis."
                    )

    except Exception as e:

        st.error(
            f"Unable to process the ECG file: {str(e)}"
        )

else:

    st.subheader(
        "Getting Started"
    )

    st.write(
        "Upload an ECG CSV file containing exactly 187 signal values "
        "to begin."
    )

    st.write(
        "The system will display the ECG waveform and use the "
        "optimized CNN-LSTM model to generate a classification."
    )

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------

st.markdown("---")

st.caption(
    "ECG Classification Research Prototype | "
    "CNN-LSTM Deep Learning Model"
)
