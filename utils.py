import requests
import pandas as pd
from datetime import datetime, timedelta
import os
import streamlit as st

def get_graph_access_token():
    secrets = st.secrets["email_graph"]
    url = f"https://login.microsoftonline.com/{secrets['tenant_id']}/oauth2/v2.0/token"
    data = {
        "client_id": secrets["client_id"],
        "scope": "https://graph.microsoft.com/.default",
        "client_secret": secrets["client_secret"],
        "grant_type": "client_credentials",
    }
    response = requests.post(url, data=data)
    response.raise_for_status()
    return response.json()["access_token"]

def send_verification_email(name, recipient_email, token):
    try:
        access_token = get_graph_access_token()
        secrets = st.secrets["email_graph"]

        url = "https://graph.microsoft.com/v1.0/users/{sender}/sendMail".format(
            sender=secrets["sender_email"]
        )

        email_message = {
            "message": {
                "subject": "Your Verification Token",
                "body": {
                    "contentType": "Text",
                    "content": f"Hello {name},\n\nYour verification token is: {token}\n\nThis token is valid for 24 hours.\n\n- Enrollment App"
                },
                "toRecipients": [{"emailAddress": {"address": recipient_email}}],
            }
        }

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }

        response = requests.post(url, headers=headers, json=email_message)
        response.raise_for_status()

        return True
    except Exception as e:
        print(f"Graph API email failed: {e}")
        return False

def save_token_to_log(email, token, filepath="data/access_log.csv"):
    now = datetime.now()
    df = pd.DataFrame([{"email": email, "token": token, "timestamp": now.isoformat()}])
    if os.path.exists(filepath):
        existing = pd.read_csv(filepath)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(filepath, index=False)

def is_token_valid(email, token, filepath="data/access_log.csv", valid_hours=24):
    if not os.path.exists(filepath):
        return False
    df = pd.read_csv(filepath)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    cutoff = datetime.now() - timedelta(hours=valid_hours)
    valid_tokens = df[(df["email"] == email) & (df["token"] == token) & (df["timestamp"] >= cutoff)]
    return not valid_tokens.empty



