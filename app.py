
import streamlit as st
import numpy as np
import pickle

# Load trained model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))


# Streamlit page settings
st.set_page_config(page_title="Parkinson's Disease Detector", layout="wide")

# Add a banner image
# banner=Image.open("banner.png")
# banner_resized=banner.resize((1200,300))
# st.image(banner_resized,use_container_width=True)
st.image("banner.png", use_container_width=True)  # Updated parameter

# App Title
st.title("🧠 Parkinson's Disease Prediction")

st.write("This AI model predicts Parkinson’s Disease based on voice pattern analysis. "
         "Enter the patient’s voice parameters in the sidebar to get a prediction.")

# Sidebar for user input
st.sidebar.header("📊 Enter Voice Parameters")

fo = st.sidebar.number_input("MDVP:Fo(Hz) (Average vocal fundamental frequency)", value=120.0)
fhi = st.sidebar.number_input("MDVP:Fhi(Hz) (Maximum vocal fundamental frequency)", value=140.0)
flo = st.sidebar.number_input("MDVP:Flo(Hz) (Minimum vocal fundamental frequency)", value=110.0)
jitter_percent = st.sidebar.number_input("MDVP:Jitter(%)", value=0.01)
jitter_abs = st.sidebar.number_input("MDVP:Jitter(Abs)", value=0.00008)
rap = st.sidebar.number_input("MDVP:RAP", value=0.004)
ppq = st.sidebar.number_input("MDVP:PPQ", value=0.006)
ddp = st.sidebar.number_input("Jitter:DDP", value=0.014)
shimmer = st.sidebar.number_input("MDVP:Shimmer", value=0.06)
shimmer_db = st.sidebar.number_input("MDVP:Shimmer(dB)", value=0.6)
apq3 = st.sidebar.number_input("Shimmer:APQ3", value=0.03)
apq5 = st.sidebar.number_input("Shimmer:APQ5", value=0.04)
apq = st.sidebar.number_input("MDVP:APQ", value=0.04)
dda = st.sidebar.number_input("Shimmer:DDA", value=0.09)
nhr = st.sidebar.number_input("NHR", value=0.02)
hnr = st.sidebar.number_input("HNR", value=20.0)
rpde = st.sidebar.number_input("RPDE", value=0.45)
dfa = st.sidebar.number_input("DFA", value=0.82)
spread1 = st.sidebar.number_input("Spread1", value=-4.0)
spread2 = st.sidebar.number_input("Spread2", value=0.3)
d2 = st.sidebar.number_input("D2", value=2.4)
ppe = st.sidebar.number_input("PPE", value=0.37)

# Collect input data
input_data = np.array([fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp,
                        shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr, 
                        rpde, dfa, spread1, spread2, d2, ppe]).reshape(1, -1)

# Ensure proper input shape
if input_data.shape[1] != 22:
    st.sidebar.error("⚠️ Incorrect number of input features! Please check your inputs.")

# Standardize input
input_data_scaled = scaler.transform(input_data)

# Prediction Button
if st.sidebar.button("🔍 Predict"):
    # Validate input shape before making predictions
    if input_data_scaled.shape[1] == 22:
        # Make prediction
        prediction = model.predict(input_data_scaled)

        st.subheader("🔎 Prediction Result:")

        # Show results with images
        if len(prediction) > 0 and prediction[0] == 1:
            st.error("⚠️ The Person HAS Parkinson’s Disease")
            st.image("afflict.png", use_container_width=True)
        elif len(prediction) > 0:
            st.success("✅ The Person does NOT have Parkinson’s Disease")
            st.image("healthy.png", use_container_width=True)
            # st.image("healthy.png", width=300)


        else:
            st.warning("⚠️ Unable to make a prediction. Please try again.")
    else:
        st.error("⚠️ Invalid input data. Prediction cannot be made.")

