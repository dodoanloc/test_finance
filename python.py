# app.py
# Agribank - CHUYÊN GIA TƯ VẤN PHÁP LUẬT VỀ TIỀN GỬI
# UI Streamlit, font Montserrat, kết nối n8n webhook (prod/test fallback)

import os
import uuid
import json
import requests
import streamlit as st
from datetime import datetime
from typing import Any, Dict

# ========================
# ---- CONFIG SECTION ----
# ========================
st.set_page_config(page_title="CHUYÊN GIA TƯ VẤN PHÁP LUẬT VỀ TIỀN GỬI", page_icon="💬", layout="centered")

# Lấy URL webhook từ secrets hoặc env
N8N_WEBHOOK_URL = st.secrets.get("N8N_CHAT_WEBHOOK_URL", os.getenv("N8N_CHAT_WEBHOOK_URL", ""))
N8N_WEBHOOK_TEST_URL = st.secrets.get("N8N_CHAT_WEBHOOK_TEST_URL", os.getenv("N8N_CHAT_WEBHOOK_TEST_URL", ""))

# Tuỳ chọn header bảo mật (Bearer ... hoặc "X-API-Key: XXX")
AUTH_HEADER = st.secrets.get("N8N_AUTH_HEADER", os.getenv("N8N_AUTH_HEADER", ""))

APP_TITLE = "CHUYÊN GIA TƯ VẤN PHÁP LUẬT VỀ TIỀN GỬI"
APP_BRAND = "Agribank"
FOOTER_TEXT = "Đội 4: Tam Nông 2025 — copyright © 2025"

AGRI_RED = "#8A1538"
AGRI_GREEN = "#009045"
INK = "#111827"
BG = "#0b0b0c" if st.get_option("theme.base") == "dark" else "#faf7f9"
CARD = "#111113" if st.get_option("theme.base") == "dark" else "#ffffff"

def now_vn():
    return datetime.utcnow().strftime("%H:%M")

# =========================
# ---- STYLES & HEADER ----
# =========================
st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700;800&display=swap" rel="stylesheet" />
    """,
    unsafe_allow_html=True
)

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
* {{
  font-family: 'Montserrat', system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, 'Helvetica Neue', Arial, sans-serif;
}}
.block-container {{
  padding-top: 1rem !important;
}}
.header {{
  display: grid; grid-template-columns: 56px auto; gap: 12px;
  align-items: center; background: var(--card); border: 1px solid #00000020;
  border-radius: var(--radius); padding: 12px 14px; margin-bottom: 10px;
  box-shadow: 0 6px 22px rgba(0,0,0,.08);
}}
.logo {{
  width: 46px; height: 46px; border-radius: 12px; display: grid; place-items: center;
  background: radial-gradient(60% 60% at 50% 40%, #9b2044 0%, var(--agri-red) 60%, #5f0e28 100%);
  color: #fff; font-weight: 800; letter-spacing: .2px;
}}
.title-wrap {{ display: flex; flex-direction: column; gap: 2px; }}
.h1 {{
  font-weight: 800; font-size: clamp(18px, 1.4vw + 14px, 26px);
  color: var(--agri-red); line-height: 1.1; letter-spacing: .2px;
}}
.subtitle {{ font-size: 12.5px; color: #6b7280; }}

.chat-area {{ background: transparent; }}
.msg {{ display: flex; gap: 10px; margin: 10px 0; }}
.msg .avatar {{
  flex: 0 0 36px; width: 36px; height: 36px; border-radius: 50%;
  display: grid; place-items: center; font-weight: 700; color: #fff;
}}
.msg.user .avatar {{ background: #374151; }}
.msg.bot .avatar {{ background: var(--agri-red); }}
.bubble {{
  max-width: 90%; padding: 10px 12px; border-radius: 14px;
  border: 1px solid #00000015; font-size: 15px; line-height: 1.55;
}}
.msg.user .bubble {{ background: #11182710; color: var(--ink); border-top-left-radius: 4px; }}
.msg.bot .bubble {{ background: var(--card); color: var(--ink); border-top-right-radius: 4px; }}

.input-wrap {{ position: sticky; bottom: 0; background: transparent; padding-top: 8px; }}

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
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "bot", "text": "Xin chào! Tôi có thể hỗ trợ bạn về pháp luật trong lĩnh vực tiền gửi. Bạn muốn hỏi điều gì?"}
    ]

# ==========================
# ---- UTILITIES -----------
# ==========================
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
    keys = ["answer", "response", "text", "message", "output", "data", "result"]
    for k in keys:
        v = resp_json.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    def deep_get(d, path):
        for k in path:
            if isinstance(d, dict) and k in d:
                d = d[k]
            else:
                return None
        return d

    nested = [
        ["data", "answer"], ["data", "response"], ["data", "text"],
        ["output", "text"], ["output", "message"], ["output", "answer"],
        ["message", "text"], ["result", "text"],
    ]
    for route in nested:
        v = deep_get(resp_json, route)
        if isinstance(v, str) and v.strip():
            return v.strip()

    items = resp_json.get("items")
    if isinstance(items, list) and items:
        first = items[0]
        if isinstance(first, dict) and isinstance(first.get("json"), dict):
            j = first["json"]
            for k in keys:
                v = j.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()

    return json.dumps(resp_json, ensure_ascii=False)[:1200]

def post_to_n8n(user_text: str, session_id: str) -> str:
    def _call(url):
        payload = {"chatInput": user_text, "sessionId": session_id}
        headers = build_headers()
        return requests.post(url, headers=headers, json=payload, timeout=60)

    if not N8N_WEBHOOK_URL and not N8N_WEBHOOK_TEST_URL:
        return "⚠️ Chưa cấu hình N8N_CHAT_WEBHOOK_URL hoặc N8N_CHAT_WEBHOOK_TEST_URL."

    # Thử PROD trước
    if N8N_WEBHOOK_URL:
        try:
            r = _call(N8N_WEBHOOK_URL)
            if r.status_code == 404 and "not registered" in r.text.lower():
                # Fallback TEST nếu có
                if N8N_WEBHOOK_TEST_URL:
                    rt = _call(N8N_WEBHOOK_TEST_URL)
                    if rt.status_code >= 400:
                        return f"⚠️ Test webhook lỗi {rt.status_code}: {rt.text[:200]}"
                    data = rt.json() if "application/json" in rt.headers.get("Content-Type","") else {"text": rt.text}
                    return parse_n8n_response(data) or "⚠️ Không nhận được câu trả lời hợp lệ từ n8n (test)."
                return "⚠️ Workflow chưa kích hoạt. Bật workflow hoặc dùng N8N_CHAT_WEBHOOK_TEST_URL."
            if r.status_code >= 400:
                return f"⚠️ Webhook lỗi {r.status_code}: {r.text[:200]}"
            data = r.json() if "application/json" in r.headers.get("Content-Type","") else {"text": r.text}
            return parse_n8n_response(data) or "⚠️ Không nhận được câu trả lời hợp lệ từ n8n."
        except requests.exceptions.RequestException as e:
            return f"⚠️ Kết nối n8n thất bại: {e}"

    # Không có PROD → dùng TEST
    try:
        rt = _call(N8N_WEBHOOK_TEST_URL)
        if rt.status_code >= 400:
            return f"⚠️ Test webhook lỗi {rt.status_code}: {rt.text[:200]}"
        data = rt.json() if "application/json" in rt.headers.get("Content-Type","") else {"text": rt.text}
        return parse_n8n_response(data) or "⚠️ Không nhận được câu trả lời hợp lệ từ n8n (test)."
    except requests.exceptions.RequestException as e:
        return f"⚠️ Kết nối n8n (test) thất bại: {e}"

# ==========================
# ---- HEADER (brand)  -----
# ==========================
with st.container():
    st.markdown(
        f"""
        <div class="header">
          <div class="logo">A</div>
          <div class="title-wrap">
            <div class="h1">{APP_TITLE}</div>
            <div class="subtitle">{APP_BRAND}</div>
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
        else:
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
    with st.form("chat_form", clear_on_submit=True):
        user_text = st.text_input(
            "Nhập câu hỏi...",
            placeholder="Ví dụ: Quy định lãi suất không kỳ hạn? Quyền và nghĩa vụ của người gửi tiền?",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Gửi", use_container_width=True)

if submitted and user_text.strip():
    st.session_state.messages.append({"role": "user", "text": user_text.strip(), "when": now_vn()})
    with st.spinner("Đang xử lý…"):
        bot_reply = post_to_n8n(user_text.strip(), st.session_state.session_id)
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
