import streamlit as st
import requests
import json

# Title and description
st.title("Fraud Detection App")
st.write("Enter the transaction and identity data below to get a fraud prediction.")

# Form to collect data
with st.form(key="fraud_form"):
    st.subheader("Transaction Data")
    transaction_json = st.text_area(
        "Transaction Data (in JSON format)",
        height=150,
        value='{"TransactionAmt": 100.0, "ProductCD": "W", "card1": 123, "card2": 150, "card3": 200, "card4": "visa"}'
    )
    
    st.subheader("Identity Data")
    identity_json = st.text_area(
        "Identity Data (in JSON format)",
        height=150,
        value='{"DeviceType": "mobile", "DeviceInfo": "Samsung", "id_12": "abc"}'
    )
    
    submit_button = st.form_submit_button(label="Predict Fraud")

# When the form is submitted, send a POST request to your backend API
if submit_button:
    try:
        # Parse the JSON input from the text areas
        transaction_data = json.loads(transaction_json)
        identity_data = json.loads(identity_json)
        
        # Build the payload as expected by your API
        payload = {
            "transaction_table": transaction_data,
            "identity_table": identity_data
        }
        
        # Replace with your API's URL if different (e.g., deployed URL)
        api_url = "/api/predict"
        response = requests.post(api_url, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            st.success(f"Predicted Fraud Probability: {result['prediction']:.4f}")
        else:
            st.error(f"Error: {response.text}")
    except Exception as e:
        st.error(f"Error processing input: {e}")
