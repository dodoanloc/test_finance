# app.py
# Streamlit chat UI for Agribank legal assistant (Deposits Law)
# Author: ChatGPT (Đội 4: Tam Nông)
# Requires: streamlit>=1.36, requests

import os
import time
import uuid
import json
import requests
import streamlit as st
from datetime import datetime
from typing import Any, Dict, Optional

# ========================
# ---- CONFIG SECTION ----
# ========================

# 1) Đặt URL webhook n8n (Chat Trigger) qua secrets hoặc env:
#    - Ưu tiên: .streamlit/secrets.toml -> N8N_CHAT_WEBHOOK_URL = "https://<your-n8n>/webhook/<id>"
#    - Hoặc: export N8N_CHAT_WEBHOOK_URL="https://<your-n8n>/webhook/<id>"
N8N_WEBHOOK_URL = st.secrets.get("N8N_CHAT_WEBHOOK_URL", os.getenv("N8N_CHAT_WEBHOOK_URL", ""))

# 2) Tùy chọn Header bảo mật (Bearer token, API-Key...), để trống nếu không dùng.
#    - Ví dụ: st.secrets["N8N_AUTH_HEADER"] = "Bearer YOUR_TOKEN"
AUTH_HEADER = st.secrets.get("N8N_AUTH_HEADER", os.getenv("N8N_AUTH_HEADER", ""))

# 3) Tên chatbot
APP_TITLE = "CHUYÊN GIA TƯ VẤN PHÁP LUẬT VỀ TIỀN GỬI"
APP_BRAND = "Agribank"
FOOTER_TEXT = "Đội 4: Tam Nông 2025 — copyright © 2025"

# 4) Màu sắc thương hiệu Agribank
AGRI_RED = "#8A1538"
AGRI_GREEN = "#009045"
INK = "#1f2937"
BG = "#0b0b0c" if st.get_option("theme.base") == "dark" else "#faf7f9"
CARD = "#111113" if st.get_option("theme.base") == "dark" else "#ffffff"

# 5) Timezone hiển thị
def now_vn():
    # streamlit server thường UTC, chỉ format HH:MM cho gọn
    return datetime.utcnow().strftime("%H:%M")

# =========================
# ---- STYLES & HEADER ----
# =========================
st.set_page_config(page_title=APP_TITLE, page_icon="💬", layout="centered")

CUSTOM_CSS = f"""
<style>
:root {{
  --agri-red: {AGRI_RED};
  --agri-green: {AGRI_GREEN};
  --ink: {INK};
  --bg: {BG};
  --card: {CARD};
  --radius: 16px;
}}
/* App wrapper */
.block-container {{
  padding-top: 1rem !important;
}}

/* Top bar */
.header {{
  display: grid; grid-template-columns: 56px auto; gap: 12px;
  align-items: center; background: var(--card); border: 1px solid #00000020;
  border-radius: var(--radius); padding: 12px 14px; margin-bottom: 10px;
  box-shadow: 0 6px 22px rgba(0,0,0,.08);
}}
.logo {{
  width: 46px; height: 46px; border-radius: 12px; display: grid; place-items: center;
  background: radial-gradient(60% 60% at 50% 40%, #9b2044 0%, var(--agri-red) 60%, #5f0e28 100%);
  color: #fff; font-weight: 700;
}}
.title-wrap {{
  display: flex; flex-direction: column; gap: 2px;
}}
.h1 {{
  font-weight: 800; font-size: clamp(18px, 1.4vw + 14px, 26px);
  color: var(--agri-red); line-height: 1.1; letter-spacing: .2px;
}}
.subtitle {{
  font-size: 13px; color: #6b7280;
}}

/* Chat bubbles */
.chat-area {{
  background: transparent;
}}
.msg {{
  display: flex; gap: 10px; margin: 10px 0;
}}
.msg .avatar {{
  flex: 0 0 36px; width: 36px; height: 36px; border-radius: 50%;
  display: grid; place-items: center; font-weight: 700; color: #fff;
}}
.msg.user .avatar {{ background: #374151; }}
.msg.bot .avatar {{ background: var(--agri-red); }}
.bubble {{
  max-width: 90%; padding: 10px 12px; border-radius: 14px;
  border: 1px solid #00000015; font-size: 15px; line-height: 1.45;
}}
.msg.user .bubble {{
  background: #11182715; color: var(--ink); border-top-left-radius: 4px;
}}
.msg.bot .bubble {{
  background: var(--card); color: var(--ink); border-top-right-radius: 4px;
}}

/* Input row */
.input-wrap {{
  position: sticky; bottom: 0; background: transparent;
  padding-top: 8px; backdrop-filter: blur(4px);
}}
.disclaimer {{
  font-size: 12px; color: #6b7280; margin-top: 6px;
}}

/* Footer */
.footer {{
  margin-top: 18px; font-size: 12px; color: #9ca3af; text-align: center;
}}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ==========================
# ---- SESSION & STATE  ----
# ==========================
if "session_id" not in st.session_state:
    # Khớp payload với n8n: workflow mong muốn có 'sessionId'
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "bot", "text": "Xin chào! Tôi là trợ lý tư vấn pháp luật về tiền gửi. Bạn muốn hỏi điều gì?"}
    ]

# ==========================
# ---- UTILITIES ----------- 
# ==========================
def build_headers() -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if AUTH_HEADER:
        # Chấp nhận cả 'Bearer ...' hoặc 'X-API-Key: ...'
        if AUTH_HEADER.lower().startswith("bearer "):
            headers["Authorization"] = AUTH_HEADER
        elif ":" in AUTH_HEADER:
            # e.g. "X-API-Key: XXXXX"
            k, v = AUTH_HEADER.split(":", 1)
            headers[k.strip()] = v.strip()
        else:
            headers["Authorization"] = f"Bearer {AUTH_HEADER}"
    return headers

def parse_n8n_response(resp_json: Dict[str, Any]) -> str:
    """
    Chuẩn hóa để lấy lời đáp từ nhiều kiểu output khác nhau của n8n.
    Ưu tiên một số key phổ biến.
    """
    candidates = [
        "answer", "response", "text", "message", "output", "data", "result"
    ]
    # Nếu output dạng { data: { output: "..." } } cũng cố gắng lần mò.
    # Trả về chuỗi rỗng nếu không tìm thấy.
    def deep_get(d, keys):
        for k in keys:
            if isinstance(d, dict) and k in d:
                d = d[k]
            else:
                return None
        return d

    # Thử tầng 1
    for k in candidates:
        v = resp_json.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    # Thử sâu hơn
    nested_routes = [
        ["data", "answer"], ["data", "response"], ["data", "text"],
        ["output", "text"], ["output", "message"], ["output", "answer"],
        ["message", "text"], ["result", "text"],
    ]
    for route in nested_routes:
        v = deep_get(resp_json, route)
        if isinstance(v, str) and v.strip():
            return v.strip()

    # Nếu workflow trả về mảng items
    items = resp_json.get("items")
    if isinstance(items, list) and items:
        # Lấy trường 'json' trong items[0] nếu có
        first = items[0]
        if isinstance(first, dict):
            j = first.get("json")
            if isinstance(j, dict):
                for k in candidates:
                    v = j.get(k)
                    if isinstance(v, str) and v.strip():
                        return v.strip()

    # fallback: stringify gọn
    return json.dumps(resp_json, ensure_ascii=False)[:1200]

def post_to_n8n(user_text: str, session_id: str) -> str:
    """Gửi chatInput + sessionId tới n8n và nhận câu trả lời."""
    if not N8N_WEBHOOK_URL:
        return "⚠️ Chưa cấu hình N8N_CHAT_WEBHOOK_URL."

    payload = {
        "chatInput": user_text,
        "sessionId": session_id,
    }
    headers = build_headers()

    try:
        res = requests.post(N8N_WEBHOOK_URL, headers=headers, json=payload, timeout=60)
        if res.status_code >= 400:
            return f"⚠️ Lỗi webhook {res.status_code}: {res.text[:200]}"
        data = res.json() if "application/json" in res.headers.get("Content-Type", "") else {"text": res.text}
        answer = parse_n8n_response(data)
        return answer or "⚠️ Không nhận được câu trả lời hợp lệ từ n8n."
    except requests.exceptions.RequestException as e:
        return f"⚠️ Kết nối n8n thất bại: {e}"

# ==========================
# ---- HEADER (brand)  -----
# ==========================
with st.container():
    st.markdown(
        """
        <div class="header">
          <div class="logo">A</div>
          <div class="title-wrap">
            <div class="h1">CHUYÊN GIA TƯ VẤN PHÁP LUẬT VỀ TIỀN GỬI</div>
            <div class="subtitle">""" + APP_BRAND + """ · Tư vấn theo dữ liệu đã cung cấp</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ==========================
# ---- CHAT HISTORY --------
# ==========================
chat_area = st.container()
with chat_area:
    st.markdown('<div class="chat-area">', unsafe_allow_html=True)
    for m in st.session_state.messages:
        role = m.get("role", "bot")
        text = m.get("text", "")
        when = m.get("when", now_vn())
        if role == "user":
            st.markdown(
                f"""
                <div class="msg user">
                  <div class="avatar">U</div>
                  <div>
                    <div class="bubble">{text}</div>
                    <div style="font-size:11px;color:#9ca3af;margin-top:2px;">{when}</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:  # bot
            st.markdown(
                f"""
                <div class="msg bot">
                  <div class="avatar">A</div>
                  <div>
                    <div class="bubble">{text}</div>
                    <div style="font-size:11px;color:#9ca3af;margin-top:2px;">{when}</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    st.markdown("</div>", unsafe_allow_html=True)

# ==========================
# ---- INPUT AREA ----------
# ==========================
with st.container():
    st.markdown('<div class="input-wrap">', unsafe_allow_html=True)
    with st.form("chat_form", clear_on_submit=True):
        user_text = st.text_input(
            "Nhập câu hỏi pháp luật về tiền gửi…",
            placeholder="Ví dụ: Quy định lãi suất không kỳ hạn hiện hành? Quyền và nghĩa vụ của người gửi tiền?",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Gửi", use_container_width=True)
    st.markdown(
        '<div class="disclaimer">* Trợ lý chỉ tư vấn dựa trên dữ liệu đã nạp vào hệ thống. '
        'Vui lòng kiểm tra văn bản pháp luật hiện hành khi áp dụng thực tế.</div>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

if submitted and user_text.strip():
    # append user message
    st.session_state.messages.append({"role": "user", "text": user_text.strip(), "when": now_vn()})
    with st.spinner("Đang xử lý…"):
        bot_reply = post_to_n8n(user_text.strip(), st.session_state.session_id)
        # append bot message
        st.session_state.messages.append({"role": "bot", "text": bot_reply, "when": now_vn()})
        st.rerun()

# ==========================
# ---- FOOTER --------------
# ==========================
st.markdown(
    f"""
    <div class="footer">
      <hr style="opacity:.2;margin:8px 0 12px 0" />
      {FOOTER_TEXT}
    </div>
    """,
    unsafe_allow_html=True
)

# ============ DIAGNOSTICS ============
with st.expander("⚙️ Cấu hình (ẩn/hiện)"):
    st.write("Session ID:", st.session_state.session_id)
    st.write("Webhook URL đã cấu hình:", "✅" if bool(N8N_WEBHOOK_URL) else "❌ (chưa đặt N8N_CHAT_WEBHOOK_URL)")
    st.caption("Bạn có thể đặt biến trong .streamlit/secrets.toml hoặc biến môi trường OS.")
