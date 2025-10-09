import os
import uuid
import json
import requests
import streamlit as st
from datetime import datetime

# ========== CẤU HÌNH CƠ BẢN ==========
st.set_page_config(
    page_title="Chuyên gia tư vấn pháp luật về tiền gửi",
    page_icon="💬",
    layout="centered",
)

# ========== ĐỌC THÔNG TIN WEBHOOK ==========
N8N_URL = st.secrets.get("N8N_CHAT_WEBHOOK_URL", os.getenv("N8N_CHAT_WEBHOOK_URL", ""))
N8N_TEST = st.secrets.get("N8N_CHAT_WEBHOOK_TEST_URL", os.getenv("N8N_CHAT_WEBHOOK_TEST_URL", ""))
AUTH_HEADER = st.secrets.get("N8N_AUTH_HEADER", os.getenv("N8N_AUTH_HEADER", ""))

# ========== CSS GIAO DIỆN ==========
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600&display=swap');
        html, body, [class*="css"] {
            font-family: 'Montserrat', sans-serif;
            background-color: #fafafa;
            color: #333;
        }
        .header-container {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
            margin-bottom: 20px;
        }
        .header-title {
            font-weight: 600;
            font-size: 24px;
            color: #8A1538;
            text-align: center;
        }
        .copyright {
            font-size: 13px;
            color: #777;
            text-align: center;
            margin-top: 30px;
        }
        .chat-box {
            background: white;
            padding: 16px;
            border-radius: 12px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.1);
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# ========== HEADER (CÓ LOGO) ==========
st.markdown("""
<div class='header-container'>
    <img src='logo.png' width='45'>
    <div class='header-title'>CHUYÊN GIA TƯ VẤN PHÁP LUẬT VỀ TIỀN GỬI</div>
</div>
""", unsafe_allow_html=True)

# ========== HÀM GỬI YÊU CẦU ==========
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

def post_to_n8n(prompt, session_id):
    if not N8N_URL:
        st.error("⚠️ Chưa cấu hình N8N_CHAT_WEBHOOK_URL trong secrets.toml hoặc biến môi trường.")
        return None

    payload = {"chatInput": prompt, "sessionId": session_id}
    try:
        r = requests.post(N8N_URL, headers=build_headers(), json=payload, timeout=60)
        if r.status_code != 200:
            st.error(f"⚠️ Lỗi {r.status_code}: {r.text}")
            return None
        try:
            data = r.json()
            return data.get("answer") or data.get("output") or data
        except Exception:
            return r.text
    except Exception as e:
        st.error(f"🚫 Không thể gửi yêu cầu: {e}")
        return None

# ========== LƯU SESSION ==========
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# ========== FORM CHAT ==========
with st.container():
    st.markdown("<div class='chat-box'>", unsafe_allow_html=True)
    for msg in st.session_state.messages:
        st.markdown(f"**{msg['role']}:** {msg['content']}")
    st.markdown("</div>", unsafe_allow_html=True)

prompt = st.text_input("Nhập câu hỏi của bạn:", placeholder="Ví dụ: Lãi suất tiền gửi kỳ hạn 6 tháng hiện nay là bao nhiêu?")
if st.button("Gửi"):
    if prompt.strip():
        st.session_state.messages.append({"role": "👤 Bạn", "content": prompt})
        answer = post_to_n8n(prompt, st.session_state.session_id)
        if answer:
            st.session_state.messages.append({"role": "🤖 Trợ lý", "content": str(answer)})
        st.rerun()

# ========== COPYRIGHT ==========
st.markdown("<div class='copyright'>© Đội 4: Tam Nông 2025 - Agribank Thọ Xuân</div>", unsafe_allow_html=True)
