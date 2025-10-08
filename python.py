import io
import unicodedata
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from google import genai
from google.genai.errors import APIError

# ------------------------
# Page setup
# ------------------------
st.set_page_config(page_title="App Phân Tích Báo Cáo Tài Chính", layout="wide")
st.title("Ứng dụng Phân Tích Báo Cáo Tài Chính 📊")

# ------------------------
# Helpers
# ------------------------

def normalize_vn(text: str) -> str:
    """Remove Vietnamese diacritics, collapse spaces, and upper-case for robust matching."""
    if text is None:
        return ""
    txt = unicodedata.normalize("NFD", str(text))
    txt = "".join(ch for ch in txt if unicodedata.category(ch) != "Mn")
    return " ".join(txt.split()).upper()


def safe_div(numer, denom, default=np.nan):
    denom = np.where(denom == 0, np.nan, denom)
    out = numer / denom
    return np.where(np.isfinite(out), out, default)


RE_TOTAL_ASSETS = normalize_vn("TỔNG CỘNG TÀI SẢN")
RE_CA = normalize_vn("TÀI SẢN NGẮN HẠN")
RE_CL = normalize_vn("NỢ NGẮN HẠN")

# ------------------------
# Core processing
# ------------------------
@st.cache_data(show_spinner=False)
def process_financial_data(df: pd.DataFrame, col_prev: str, col_curr: str) -> pd.DataFrame:
    """Compute growth and structure weights between two chosen period columns.
    Requires columns: Chỉ tiêu | <period columns>.
    """
    if "Chỉ tiêu" not in df.columns:
        raise ValueError("Thiếu cột 'Chỉ tiêu'.")
    for c in [col_prev, col_curr]:
        if c not in df.columns:
            raise ValueError(f"Thiếu cột kỳ dữ liệu: {c}")

    df = df.copy()

    # Ensure numeric for selected columns
    for col in [col_prev, col_curr]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Helper key for matching
    df["_KEY"] = df["Chỉ tiêu"].map(normalize_vn)

    # Growth %
    df["Tốc độ tăng trưởng (%)"] = safe_div(df[col_curr] - df[col_prev], df[col_prev]) * 100

    # Structure weights relative to Total Assets (for each selected period)
    tot_row = df.loc[df["_KEY"] == RE_TOTAL_ASSETS]
    if tot_row.empty:
        raise ValueError("Không tìm thấy chỉ tiêu 'TỔNG CỘNG TÀI SẢN'. Hãy đảm bảo tiêu đề đúng chính tả.")

    tot_prev = float(tot_row[col_prev].iloc[0])
    tot_curr = float(tot_row[col_curr].iloc[0])

    df[f"Tỷ trọng {col_prev} (%)"] = safe_div(df[col_prev], tot_prev, default=np.nan) * 100
    df[f"Tỷ trọng {col_curr} (%)"] = safe_div(df[col_curr], tot_curr, default=np.nan) * 100

    return df.drop(columns=["_KEY"])  # hide helper column


# ------------------------
# Gemini helpers
# ------------------------
def get_ai_analysis(data_for_ai: str, api_key: str) -> str:
    try:
        client = genai.Client(api_key=api_key)
        model_name = "gemini-2.5-flash"
        prompt = f"""
        Bạn là chuyên gia phân tích tài chính. Dựa trên các chỉ số sau, hãy đưa ra nhận xét khách quan, ngắn gọn (3–4 đoạn) về: tốc độ tăng trưởng, thay đổi cơ cấu tài sản, và khả năng thanh toán hiện hành.

        Dữ liệu đầu vào:
        {data_for_ai}
        """
        resp = client.models.generate_content(model=model_name, contents=prompt)
        return getattr(resp, "text", "Không nhận được văn bản phản hồi từ mô hình.")
    except APIError as e:
        return f"Lỗi gọi Gemini API: kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết: {e}"
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định khi gọi Gemini: {e}"


def gemini_chat(user_message: str, history: list, api_key: str) -> str:
    """Simple chat wrapper using history as context."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = "gemini-2.5-flash"
        sys_prompt = (
            "Bạn là trợ lý phân tích tài chính. Trả lời bằng tiếng Việt, rõ ràng, súc tích, có cấu trúc đầu mục khi phù hợp."
        )
        # Build a single contents string with brief history
        ctx = [f"Người dùng: {m['user']}\nTrợ lý: {m['ai']}" for m in history if m.get('ai')]
        context_blob = "\n\n".join(ctx)
        content = f"{sys_prompt}\n\nNgữ cảnh hội thoại trước:\n{context_blob}\n\nTin nhắn mới của người dùng:\n{user_message}"
        resp = client.models.generate_content(model=model_name, contents=content)
        return getattr(resp, "text", "")
    except Exception as e:
        return f"(Lỗi khi gọi Gemini: {e})"


# ------------------------
# Sidebar controls
# ------------------------
st.sidebar.header("⚙️ Tuỳ chọn")
api_key = st.sidebar.text_input("Gemini API Key", type="password", value=st.secrets.get("GEMINI_API_KEY", ""))

# ------------------------
# UI — Sample template download
# ------------------------
with st.expander("📥 Tải file mẫu (Excel)"):
    sample = pd.DataFrame(
        {
            "Chỉ tiêu": [
                "TÀI SẢN NGẮN HẠN",
                "TÀI SẢN DÀI HẠN",
                "TỔNG CỘNG TÀI SẢN",
                "NỢ NGẮN HẠN",
                "NỢ DÀI HẠN",
                "VỐN CHỦ SỞ HỮU",
            ],
            "2023": [5000, 7000, 12000, 3000, 2000, 7000],
            "2024": [6500, 7500, 14000, 3200, 2100, 8700],
        }
    )
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        sample.to_excel(writer, index=False)
    st.download_button("Tải xuống mẫu.xlsx", data=buf.getvalue(), file_name="mau_bctc.xlsx")

# ------------------------
# UI — File upload
# ------------------------
uploaded_file = st.file_uploader(
    "1) Tải file Báo cáo Tài chính (Excel hoặc CSV) — cột: Chỉ tiêu | các cột kỳ (VD: 2023, 2024)",
    type=["xlsx", "xls", "csv"],
)

if uploaded_file is None:
    st.info("Vui lòng tải lên file để bắt đầu phân tích.")
    st.stop()

# Read file safely
try:
    if uploaded_file.name.lower().endswith(".csv"):
        df_raw = pd.read_csv(uploaded_file)
    else:
        df_raw = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"Không thể đọc file: {e}")
    st.stop()

# Try to standardize columns
if "Chỉ tiêu" not in df_raw.columns:
    if len(df_raw.columns) >= 1:
        df_raw = df_raw.rename(columns={df_raw.columns[0]: "Chỉ tiêu"})
    else:
        st.error("Thiếu cột 'Chỉ tiêu'.")
        st.stop()

# Choose two periods to compare (support multi-period files)
period_cols = [c for c in df_raw.columns if c != "Chỉ tiêu"]
if len(period_cols) < 2:
    st.error("Cần ít nhất 2 cột kỳ để so sánh (VD: 2023, 2024 hoặc Năm trước, Năm sau).")
    st.stop()

col_left, col_right = st.columns(2)
with col_left:
    col_prev = st.selectbox("Chọn kỳ trước", period_cols, index=0)
with col_right:
    # Default to last column as current period
    default_idx = len(period_cols) - 1
    col_curr = st.selectbox("Chọn kỳ sau", period_cols, index=default_idx)

# Process
try:
    df_processed = process_financial_data(df_raw, col_prev, col_curr)
except ValueError as ve:
    st.error(f"Lỗi cấu trúc dữ liệu: {ve}")
    st.stop()
except Exception as e:
    st.error(f"Có lỗi khi xử lý dữ liệu: {e}")
    st.stop()

# ------------------------
# Results table
# ------------------------
st.subheader("2) Tốc độ tăng trưởng & 3) Tỷ trọng cơ cấu tài sản")
st.dataframe(
    df_processed.style.format(
        {
            col_prev: "{:,.0f}",
            col_curr: "{:,.0f}",
            "Tốc độ tăng trưởng (%)": "{:.2f}%",
            f"Tỷ trọng {col_prev} (%)": "{:.2f}%",
            f"Tỷ trọng {col_curr} (%)": "{:.2f}%",
        }
    ),
    use_container_width=True,
)

# ------------------------
# Charts
# ------------------------
with st.expander("📈 Biểu đồ trực quan"):
    # Growth bar chart
    growth_chart = alt.Chart(
        df_processed[["Chỉ tiêu", "Tốc độ tăng trưởng (%)"]]
    ).mark_bar().encode(
        x=alt.X("Tốc độ tăng trưởng (%)", title="%"),
        y=alt.Y("Chỉ tiêu", sort='-x'),
        tooltip=["Chỉ tiêu", alt.Tooltip("Tốc độ tăng trưởng (%)", format=".2f")],
    ).properties(height=400)

    st.altair_chart(growth_chart, use_container_width=True)

    # Structure bar (prev vs curr)
    melted = pd.melt(
        df_processed[
            [
                "Chỉ tiêu",
                f"Tỷ trọng {col_prev} (%)",
                f"Tỷ trọng {col_curr} (%)",
            ]
        ],
        id_vars=["Chỉ tiêu"],
        var_name="Kỳ",
        value_name="Tỷ trọng (%)",
    )

    structure_chart = alt.Chart(melted).mark_bar().encode(
        y=alt.Y("Chỉ tiêu", sort='-x'),
        x=alt.X("Tỷ trọng (%)", title="%"),
        color=alt.Color("Kỳ"),
        tooltip=["Chỉ tiêu", "Kỳ", alt.Tooltip("Tỷ trọng (%)", format=".2f")],
    ).properties(height=500)

    st.altair_chart(structure_chart, use_container_width=True)

# ------------------------
# 4) Basic ratios (Current Ratio)
# ------------------------
st.subheader("4) Các chỉ số tài chính cơ bản")

df_norm = df_raw.copy()
df_norm["_KEY"] = df_norm["Chỉ tiêu"].map(normalize_vn)

try:
    ca_curr = float(df_norm.loc[df_norm["_KEY"] == RE_CA, col_curr].iloc[0])
    ca_prev = float(df_norm.loc[df_norm["_KEY"] == RE_CA, col_prev].iloc[0])
    cl_curr = float(df_norm.loc[df_norm["_KEY"] == RE_CL, col_curr].iloc[0])
    cl_prev = float(df_norm.loc[df_norm["_KEY"] == RE_CL, col_prev].iloc[0])

    curr_ratio_prev = np.nan if cl_prev == 0 else ca_prev / cl_prev
    curr_ratio_curr = np.nan if cl_curr == 0 else ca_curr / cl_curr

    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            f"Chỉ số Thanh toán hiện hành ({col_prev})",
            f"{curr_ratio_prev:.2f} lần" if np.isfinite(curr_ratio_prev) else "N/A",
        )
    with col2:
        delta = (
            curr_ratio_curr - curr_ratio_prev
            if (np.isfinite(curr_ratio_curr) and np.isfinite(curr_ratio_prev))
            else np.nan
        )
        st.metric(
            f"Chỉ số Thanh toán hiện hành ({col_curr})",
            f"{curr_ratio_curr:.2f} lần" if np.isfinite(curr_ratio_curr) else "N/A",
            None if not np.isfinite(delta) else f"{delta:.2f}",
        )
except Exception:
    st.warning("Thiếu chỉ tiêu 'TÀI SẢN NGẮN HẠN' hoặc 'NỢ NGẮN HẠN' để tính chỉ số.")
    curr_ratio_prev = np.nan
    curr_ratio_curr = np.nan

# ------------------------
# 5) AI commentary (one-shot)
# ------------------------
st.subheader("5) Nhận xét tình hình tài chính (AI)")

md_table = df_processed.to_markdown(index=False)
summary_df = pd.DataFrame(
    {
        "Chỉ tiêu": [
            "Bảng phân tích (dữ liệu)",
            f"Tăng trưởng Tài sản ngắn hạn (%) {col_prev}->{col_curr}",
            f"Thanh toán hiện hành ({col_prev})",
            f"Thanh toán hiện hành ({col_curr})",
        ],
        "Giá trị": [
            md_table,
            (
                f"{df_processed.loc[df_norm['_KEY'] == RE_CA, 'Tốc độ tăng trưởng (%)'].iloc[0]:.2f}%"
                if any(df_norm["_KEY"] == RE_CA)
                else "N/A"
            ),
            "N/A" if not np.isfinite(curr_ratio_prev) else f"{curr_ratio_prev:.2f}",
            "N/A" if not np.isfinite(curr_ratio_curr) else f"{curr_ratio_curr:.2f}",
        ],
    }
)

st.dataframe(summary_df, use_container_width=True)

col_a, col_b = st.columns(2)
with col_a:
    if st.button("Yêu cầu AI phân tích"):
        if not api_key:
            st.error("Vui lòng nhập hoặc cấu hình 'GEMINI_API_KEY' trong Sidebar.")
        else:
            with st.spinner("Đang gửi dữ liệu đến Gemini…"):
                ai_text = get_ai_analysis(summary_df.to_markdown(index=False), api_key)
            st.markdown("**Kết quả Phân tích từ Gemini AI:**")
            st.info(ai_text)

with col_b:
    # Export processed table & summary
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine="xlsxwriter") as writer:
        df_processed.to_excel(writer, index=False, sheet_name="Phan_tich")
        summary_df.to_excel(writer, index=False, sheet_name="Tom_tat")
    st.download_button(
        "⬇️ Tải Excel kết quả",
        data=excel_buf.getvalue(),
        file_name=f"phan_tich_bctc_{now}.xlsx",
    )

# ------------------------
# 6) Khung chat với Gemini
# ------------------------
st.subheader("6) Khung chat với Gemini 🤖")
if "chat" not in st.session_state:
    st.session_state.chat = []  # list of {user: str, ai: str}

prompt_help = (
    "Hãy hỏi: 'Phân tích rủi ro thanh khoản', 'Đề xuất KPI theo dõi', 'Viết summary cho ban giám đốc',..."
)
user_msg = st.text_input("Tin nhắn", placeholder=prompt_help)

# Context card
with st.expander("📎 Ngữ cảnh gửi kèm cho AI"):
    st.markdown("Bảng dữ liệu đã xử lý sẽ được AI tham chiếu.")
    st.dataframe(df_processed, use_container_width=True)

send = st.button("Gửi & nhận trả lời")
if send and user_msg:
    if not api_key:
        st.error("Vui lòng nhập Gemini API Key ở Sidebar.")
    else:
        # We keep a compact history (last 5 turns)
        history = st.session_state.chat[-5:]
        # Attach a lightweight context prefix of the table
        context_text = (
            "Bối cảnh dữ liệu (rút gọn):\n" + df_processed.head(20).to_markdown(index=False)
        )
        full_user = f"{context_text}\n\nYêu cầu: {user_msg}"
        ai_reply = gemini_chat(full_user, history, api_key)
        st.session_state.chat.append({"user": user_msg, "ai": ai_reply})

# Render chat history
for turn in st.session_state.chat[-20:]:
    with st.chat_message("user"):
        st.write(turn["user"])
    with st.chat_message("assistant"):
        st.write(turn["ai"])

# ------------------------
# Tips & notes
# ------------------------
with st.expander("💡 Gợi ý sử dụng & lưu ý"):
    st.markdown(
        """
        - Đảm bảo tồn tại chỉ tiêu **TỔNG CỘNG TÀI SẢN** để tính tỷ trọng.
        - Có thể đặt tên kỳ tuỳ ý (VD: *Năm trước/Năm sau* hoặc *2023/2024*), sau đó chọn 2 kỳ ở phần *Tuỳ chọn kỳ*.
        - Hệ thống chuẩn hoá dấu tiếng Việt để khớp các chỉ tiêu như **TÀI SẢN NGẮN HẠN**, **NỢ NGẮN HẠN**.
        - Chia cho 0 trả về **N/A** thay vì gây lỗi.
        - Nút **Tải Excel kết quả** giúp bạn lưu lại bảng phân tích & tóm tắt.
        - Khung **chat với Gemini** giữ 5 lượt hội thoại gần nhất làm ngữ cảnh; bạn có thể yêu cầu báo cáo, bullet list khuyến nghị, hoặc tạo memo gửi lãnh đạo.
        """
    )
