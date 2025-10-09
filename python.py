# app.py — Minimal & Debug
import os
import uuid
import json
import requests
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Debug Chat → n8n", page_icon="🛰️", layout="centered")

# ==== CONFIG ====
N8N_URL = st.secrets.get("N8N_CHAT_WEBHOOK_URL", os.getenv("N8N_CHAT_WEBHOOK_URL", ""))  # PRODUCTION
N8N_TEST = st.secrets.get("N8N_CHAT_WEBHOOK_TEST_URL", os.getenv("N8N_CHAT_WEBHOOK_TEST_URL", ""))  # optional
AUTH_HEADER = st.secrets.get("N8N_AUTH_HEADER", os.getenv("N8N_AUTH_HEADER", ""))  # optional

st.title("🔧 Debug gửi chat → n8n webhook")

with st.expander("⚙️ Cấu hình đang dùng"):
    st.write("N8N_CHAT_WEBHOOK_URL:", N8N_URL or "⛔ Chưa đặt")
    st.write("N8N_CHAT_WEBHOOK_TEST_URL:", N8N_TEST or "—")
    st.write("AUTH_HEADER:", ("(đang bật)" if AUTH_HEADER else "—"))

def build_headers():
    h = {"Content-Type": "application/json"}
    if AUTH_HEADER:
        if AUTH_HEADER.lower().startswith("bearer "):
            h["Authorization"] = AUTH_HEADER
        elif ":" in AUTH_HEADER:
            k, v = AUTH_HEADER.split(":", 1)
            h[k.strip()] = v.strip()
        else:
            h["Authorization"] = f"Bearer {AUTH_HEADER}"
    return h

def post(url, payload):
    r = requests.post(url, headers=build_headers(), json=payload, timeout=60)
    ct = r.headers.get("Content-Type", "")
    try:
        data = r.json() if "application/json" in ct else {"text": r.text}
    except Exception:
        data = {"text": r.text}
    return r.status_code, data, r.text

# ==== FORM ====
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

prompt = st.text_input("Câu hỏi:", value="ping", placeholder="Nhập câu hỏi để gửi vào n8n…")
use_test = st.checkbox("Dùng TEST URL (webhook-test)", value=False)
if st.button("Gửi"):
    url = (N8N_TEST if use_test and N8N_TEST else N8N_URL).strip()
    if not url:
        st.error("Chưa cấu hình URL. Đặt N8N_CHAT_WEBHOOK_URL (và/hoặc N8N_CHAT_WEBHOOK_TEST_URL).")
    else:
        payload = {"chatInput": prompt, "sessionId": st.session_state.session_id}
        st.write("➡️ URL:", url)
        st.write("➡️ Payload:", payload)
        try:
            code, data, raw = post(url, payload)
            st.write("⬅️ Status:", code)
            st.write("⬅️ Parsed:", data)
            st.code(raw[:2000], language="json")
        except Exception as e:
            st.error(f"Exception: {e}")

st.caption(f"Session: {st.session_state.session_id} · {datetime.utcnow().isoformat()}Z")
