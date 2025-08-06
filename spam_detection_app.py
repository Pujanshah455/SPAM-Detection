import streamlit as st
import joblib

st.title("📩 SMS Spam Detection App")

try:
    # Load model and vectorizer
    model = joblib.load("spam_classifier_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    st.success("✅ Model and vectorizer loaded successfully.")
except Exception as e:
    st.error(f"❌ Error loading model/vectorizer: {e}")

st.markdown("Enter an SMS message below and check if it's **Spam** or **Ham**.")

user_input = st.text_area("✉️ Enter SMS text here:")

if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message first.")
    else:
        try:
            transformed_text = vectorizer.transform([user_input])
            prediction = model.predict(transformed_text)[0]
            prediction_proba = model.predict_proba(transformed_text).max()

            if prediction == 1:
                st.error(f"🚨 This message is **SPAM** with {prediction_proba:.2%} confidence.")
            else:
                st.success(f"✅ This message is **HAM (Not Spam)** with {prediction_proba:.2%} confidence.")
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
