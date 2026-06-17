import json
import os
import boto3
import streamlit as st
from botocore.exceptions import ClientError, NoCredentialsError

ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "creditendpoints")
REGION = os.environ.get("AWS_REGION", "us-east-1")


@st.cache_resource
def get_runtime_client():
    return boto3.client("sagemaker-runtime", region_name=REGION)


def invoke_endpoint(features: list[float]) -> dict:
    runtime = get_runtime_client()
    payload = {"instances": [features]}
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload),
    )
    return json.loads(response["Body"].read().decode("utf-8"))


import streamlit as st
import requests

st.title("Credit Score Prediction")

Age = st.number_input("Age", value=35.0)
Occupation = st.text_input("Occupation", value="Engineer")

Annual_Income = st.number_input("Annual Income", value=20000.0)
Monthly_Inhand_Salary = st.number_input("Monthly Salary", value=3000.0)

Num_Bank_Accounts = st.number_input("Bank Accounts", value=5.0)
Num_Credit_Card = st.number_input("Credit Cards", value=5.0)
Interest_Rate = st.number_input("Interest Rate", value=15.0)
Num_of_Loan = st.number_input("Number of Loans", value=3.0)

Delay_from_due_date = st.number_input("Delay From Due Date", value=15.0)
Num_of_Delayed_Payment = st.number_input("Delayed Payments", value=10.0)
Changed_Credit_Limit = st.number_input("Credit Limit Change", value=10.0)
Num_Credit_Inquiries = st.number_input("Credit Inquiries", value=5.0)

Credit_Mix = st.text_input("Credit Mix", value="Standard")
Outstanding_Debt = st.number_input("Debt", value=1500.0)
Credit_Utilization_Ratio = st.number_input("Utilization Ratio", value=30.0)
Credit_History_Age = st.number_input("Credit History Age", value=200.0)

Payment_of_Min_Amount = st.text_input("Min Payment", value="Yes")
Total_EMI_per_month = st.number_input("EMI", value=50.0)
Amount_invested_monthly = st.number_input("Investment", value=100.0)

Payment_Behaviour = st.text_input("Payment Behaviour", value="Low_spent_Small_value_payments")
Monthly_Balance = st.number_input("Monthly Balance", value=300.0)

if st.button("Predict", type="primary"):
    features = [
        Age,
        Occupation,
        Annual_Income,
        Monthly_Inhand_Salary,
        Num_Bank_Accounts,
        Num_Credit_Card,
        Interest_Rate,
        Num_of_Loan,
        Delay_from_due_date,
        Num_of_Delayed_Payment,
        Changed_Credit_Limit,
        Num_Credit_Inquiries,
        Credit_Mix,
        Outstanding_Debt,
        Credit_Utilization_Ratio,
        Credit_History_Age,
        Payment_of_Min_Amount,
        Total_EMI_per_month,
        Amount_invested_monthly,
        Payment_Behaviour,
        Monthly_Balance
    ]
        
    try:
        result = invoke_endpoint(features)
    except NoCredentialsError:
        st.error("No AWS credentials found.")
    except ClientError as e:
        st.error(f"AWS error: {e.response['Error'].get('Message', str(e))}")
    else:
        label = result["predictions_label"][0]
        probs = result["probabilities"][0]
    
        st.success(f"Prediction: {label}")
        st.write("Class probabilities:")
        st.bar_chart({"probability": probs})

