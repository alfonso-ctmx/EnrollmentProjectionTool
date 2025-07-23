import requests
import pandas as pd
from datetime import datetime, timedelta
import os
import streamlit as st

import time
import streamlit as st     # already imported above

TOKEN_TTL_SEC = 24 * 60 * 60      # 24 h  ➟ change if you want a shorter TTL

def store_token_in_session(token: str, ttl_sec: int = TOKEN_TTL_SEC) -> None:
    """Save the OTP and its expiry just for the current browser session."""
    st.session_state["otp"] = str(token)
    st.session_state["otp_exp"] = time.time() + ttl_sec


def is_token_valid_session(user_input: str) -> bool:
    """True if the entered code matches the session copy and is still fresh."""
    return (
        "otp" in st.session_state
        and "otp_exp" in st.session_state
        and time.time() < st.session_state["otp_exp"]
        and str(user_input).strip() == str(st.session_state["otp"])
    )

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



