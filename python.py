# app.py
# Agribank - CHUYÊN GIA TƯ VẤN PHÁP LUẬT VỀ TIỀN GỬI (Giao diện sáng + Montserrat)
# -------------------------------------------

import os
import uuid
import json
import requests
import streamlit as st
from datetime import datetime
from typing import Any, Dict

# ========== CẤU HÌNH CƠ BẢN ==========
st.set_page_config(
    page_title="CHUYÊN GIA TƯ VẤN PHÁP LUẬT VỀ TIỀN GỬI",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Ép giao diện sáng (Light Theme)
st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: #faf7f9 !important;
        color: #1f2937 !important;
    }
    [data-testid="stHeader"] {background: none;}
    </style>
""", unsafe_allow_html=True)

# ========== THÔNG TIN ỨNG DỤNG ==========
N8N_WEBHOOK_URL = st.secrets.get("N8N_CHAT_WEBHOOK_URL", os.getenv("N8N_CHAT_WEBHOOK_URL", ""))
N8N_WEBHOOK_TEST_URL = st.secrets.get("N8N_CHAT_WEBHOOK_TEST_URL", os.getenv("N8N_CHAT_WEBHOOK_TEST_URL", ""))
AUTH_HEADER = st.secrets.get("N8N_AUTH_HEADER", os.getenv("N8N_AUTH_HEADER", ""))

APP_TITLE = "CHUYÊN GIA TƯ VẤN PHÁP LUẬT VỀ TIỀN GỬI"
APP_BRAND = "Agribank"
FOOTER_TEXT = "Đội 4: Tam Nông 2025 — copyright © 2025"

AGRI_RED = "#8A1538"
AGRI_GREEN = "#009045"
INK = "#111827"
CARD = "#ffffff"

def now_vn():
    return datetime.utcnow().strftime("%H:%M")

# ========== FONT + CSS ==========
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

st.markdown(f"""
<style>
:root {{
  --agri-red: {AGRI_RED};
  --agri-green: {AGRI_GREEN};
  --ink: {INK};
  --card: {CARD};
  --radius: 16px;
}}
* {{
  font-family: 'Montserrat', system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, 'Helvetica Neue', Arial, sans-serif;
}}
.block-container {{ padding-top: 1rem !important; }}
.header {{
  display: grid;
  grid-template-columns: 56px auto;
  gap: 12px;
  align-items: center;
  background: var(--card);
  border: 1px solid #e5e7eb;
  border-radius: var(--radius);
  padding: 12px 14px;
  margin-bottom: 10px;
  box-shadow: 0 6px 22px rgba(0,0,0,.05);
}}
.logo {{
  width: 46px;
  height: 46px;
  border-radius: 12px;
  display: grid;
  place-items: center;
  background: radial-gradient(60% 60% at 50% 40%, #9b2044 0%, var(--agri-red) 60%, #5f0e28 100%);
  color: #fff;
  font-weight: 700;
}}
.title-wrap {{ display: flex; flex-direction: column; gap: 2px; }}
.h1 {{
  font-weight: 700;
  font-size: clamp(17px, 1.2vw + 12px, 22px); /* nhỏ hơn bản cũ */
  color: var(--agri-red);
  line-height: 1.2;
  letter-spacing: .2px;
}}
.subtitle {{ font-size: 12px; color: #6b7280; }}

.chat-area {{ background: transparent; }}
.msg {{ display: flex; gap: 10px; margin: 10px 0; }}
.msg .avatar {{
  flex: 0 0 36px; width: 36px; height: 36px;
  border-radius: 50%; display: grid; place-items: center;
  font-weight: 700; color: #fff;
}}
.msg.user .avatar {{ background: #374151; }}
.msg.bot .avatar {{ background: var(--agri-red); }}
.bubble {{
  max-width: 90%; padding: 10px 12px; border-radius: 14px;
  border: 1px solid #e5e7eb; font-size: 15px; line-height: 1.55;
}}
.msg.user .bubble {{
  background: #f3f4f6; color: var(--ink);
  border-top-left-radius: 4px;
}}
.msg.bot .bubble {{
  background: #fff; color: var(--ink);
  border-top-right-radius: 4px;
}}

.input-wrap {{ position: sticky; bottom: 0; background: transparent; padding-top: 8px; }}

.footer {{
  margin-top: 18px; font-size: 12px; color: #9ca3af; text-align: center;
}}
</style>
""", unsafe_allow_html=True)

# ========== KHỞI TẠO SESSION ==========
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "bot", "text": "Xin chào! Tôi có thể giúp bạn giải đáp các vấn đề pháp lý về tiền gửi. Hãy đặt câu hỏi nhé!"}
    ]

# ========== HÀM HỖ TRỢ ==========
def build_headers() -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if AUTH_HEADER:
        if AUTH_HEADER.lower().startswith("bearer "):
            headers["Authorization"] = AUTH_HEADER
        elif ":" in AUTH_HEADER:
            k, v = AUTH_HEADER.split(":", 1)
            headers[k.strip()] = v.strip()
        else:
            headers["Authorization"] = f"Bearer {AUTH_HEADER}"
    return headers

def parse_n8n_response(resp_json: Dict[str, Any]) -> str:
    for k in ["answer", "response", "text", "message", "output", "data", "result"]:
        v = resp_json.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    if isinstance(resp_json.get("items"), list):
        first = resp_json["items"][0]
        if isinstance(first, dict) and isinstance(first.get("json"), dict):
            for k, v in first["json"].items():
                if isinstance(v, str) and v.strip():
                    return v.strip()
    return json.dumps(resp_json, ensure_ascii=False)[:800]

def post_to_n8n(user_text: str, session_id: str) -> str:
    payload = {"chatInput": user_text, "sessionId": session_id}
    headers = build_headers()

    urls = []
    if N8N_WEBHOOK_URL:
        urls.append(("Production", N8N_WEBHOOK_URL))
    if N8N_WEBHOOK_TEST_URL:
        urls.append(("Test", N8N_WEBHOOK_TEST_URL))
    if not urls:
        return "⚠️ Chưa cấu hình webhook."

    for label, url in urls:
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=60)
            if r.status_code == 404 and "not registered" in r.text.lower():
                continue
            if r.status_code >= 400:
                return f"⚠️ Webhook ({label}) lỗi {r.status_code}: {r.text[:150]}"
            data = r.json() if "application/json" in r.headers.get("Content-Type", "") else {"text": r.text}
            return parse_n8n_response(data)
        except Exception as e:
            return f"⚠️ Không thể kết nối ({label}): {e}"
    return "⚠️ Workflow chưa kích hoạt."

# ========== HEADER ==========
with st.container():
    st.markdown(f"""
    <div class="header">
      <div class="logo">A</div>
      <div class="title-wrap">
        <div class="h1">{APP_TITLE}</div>
        <div class="subtitle">{APP_BRAND}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ========== KHUNG CHAT ==========
chat_area = st.container()
with chat_area:
    st.markdown('<div class="chat-area">', unsafe_allow_html=True)
    for m in st.session_state.messages:
        role, text, when = m.get("role", "bot"), m.get("text", ""), m.get("when", now_vn())
        html = f"""
        <div class="msg {role}">
          <div class="avatar">{'U' if role=='user' else 'A'}</div>
          <div>
            <div class="bubble">{text}</div>
            <div style="font-size:11px;color:#9ca3af;margin-top:2px;">{when}</div>
          </div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ========== Ô NHẬP ==========
with st.container():
    with st.form("chat_form", clear_on_submit=True):
        user_text = st.text_input(
            "Nhập câu hỏi...",
            placeholder="Ví dụ: Quy định lãi suất không kỳ hạn hiện nay?",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Gửi", use_container_width=True)

if submitted and user_text.strip():
    st.session_state.messages.append({"role": "user", "text": user_text.strip(), "when": now_vn()})
    with st.spinner("Đang xử lý..."):
        bot_reply = post_to_n8n(user_text.strip(), st.session_state.session_id)
        st.session_state.messages.append({"role": "bot", "text": bot_reply, "when": now_vn()})
        st.rerun()

# ========== FOOTER ==========
st.markdown(f"""
<div class="footer">
  <hr style="opacity:.2;margin:8px 0 12px 0" />
  {FOOTER_TEXT}
</div>
""", unsafe_allow_html=True)
