# project_evaluation_app.py

import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from docx import Document
import io
import json

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Đánh Giá Phương Án Kinh Doanh",
    layout="wide"
)

st.title("Ứng dụng Đánh Giá Phương Án Kinh Doanh 📈")

# --- Khởi tạo Session State ---
if "extracted_data" not in st.session_state:
    st.session_state.extracted_data = None
if "cash_flow_df" not in st.session_state:
    st.session_state.cash_flow_df = None
if "metrics" not in st.session_state:
    st.session_state.metrics = {}
if "ai_analysis" not in st.session_state:
    st.session_state.ai_analysis = None

# --- Hàm đọc file Word ---
@st.cache_data
def read_word_file(uploaded_file):
    """Đọc nội dung từ file Word."""
    doc = Document(uploaded_file)
    full_text = []
    for paragraph in doc.paragraphs:
        full_text.append(paragraph.text)
    return '\n'.join(full_text)

# --- Hàm gọi API Gemini để Extract Dữ liệu ---
def extract_project_data(full_text, api_key):
    """Sử dụng Gemini để extract thông tin dự án từ text."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name='gemini-2.5-flash',
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                top_p=0.8,
                max_output_tokens=2048,  # Tăng để tránh MAX_TOKENS
            ),
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }
        )

        prompt = f"""
        Bạn là chuyên gia phân tích dự án kinh doanh. Từ văn bản mô tả phương án kinh doanh sau, hãy extract các thông tin sau dưới dạng JSON hợp lệ, ngắn gọn nhất có thể:
        - "von_dau_tu": Vốn đầu tư ban đầu (số, đơn vị VND hoặc USD, nếu không có giả sử 0).
        - "dong_doi_du_an": Dòng đời dự án (số năm, nếu không có giả sử 5).
        - "doanh_thu": Danh sách doanh thu theo năm [năm1, năm2, ..., nămN] (danh sách số, nếu constant thì lặp lại).
        - "chi_phi": Danh sách chi phí theo năm [năm1, năm2, ..., nămN] (danh sách số, tương ứng doanh thu).
        - "wacc": WACC (tỷ lệ %, dạng số thập phân như 0.08 cho 8%).
        - "thue": Tỷ lệ thuế (%, dạng số thập phân như 0.20 cho 20%).

        Nếu thông tin không đầy đủ, sử dụng giá trị mặc định hợp lý (ví dụ: doanh thu/chi phí tăng 5%/năm nếu chỉ có năm đầu).
        Đảm bảo danh sách doanh thu và chi phí có độ dài bằng dong_doi_du_an. Giữ JSON ngắn gọn.

        Văn bản:
        {full_text[:3000]}  # Giảm để tránh token input cao
        """

        response = model.generate_content(prompt)
        
        # Xử lý response an toàn
        if not response.candidates or len(response.candidates) == 0:
            return {"error": "Không có candidates trong response."}
        
        candidate = response.candidates[0]
        
        # Kiểm tra safety
        if candidate.safety_ratings:
            for rating in candidate.safety_ratings:
                if rating.probability.name == "BLOCKED":
                    return {"error": f"Response bị block bởi safety: {rating.category.name}"}
        
        # Kiểm tra finish_reason
        if candidate.finish_reason.name != "STOP":
            return {"error": f"Generation không hoàn tất. Finish reason: {candidate.finish_reason.name}"}
        
        # Lấy text từ parts
        if candidate.content and candidate.content.parts:
            text = candidate.content.parts[0].text
        else:
            return {"error": "Không có content parts trong response."}
        
        # Parse JSON
        try:
            data = json.loads(text.strip())
            return data
        except json.JSONDecodeError as e:
            return {"error": f"Không thể parse JSON từ AI response: {e}. Raw text: {text[:200]}"}

    except GoogleAPIError as e:
        return {"error": f"Lỗi gọi Gemini API: {e}"}
    except Exception as e:
        return {"error": f"Lỗi không xác định: {e}"}

# --- Hàm xây dựng bảng dòng tiền ---
@st.cache_data
def build_cash_flow(extracted_data):
    """Xây dựng bảng dòng tiền từ dữ liệu extract."""
    if "error" in extracted_data:
        return None, None

    investment = extracted_data.get("von_dau_tu", 0)
    years = extracted_data.get("dong_doi_du_an", 5)
    revenues = extracted_data.get("doanh_thu", [0] * years)
    costs = extracted_data.get("chi_phi", [0] * years)
    tax_rate = extracted_data.get("thue", 0.2)
    wacc = extracted_data.get("wacc", 0.1)

    # Đảm bảo lists có đúng length và là số
    if isinstance(revenues, list) and len(revenues) != years:
        if revenues:
            rev0 = float(revenues[0]) if revenues[0] else 1000000000
            revenues = [rev0 * (1.05 ** i) for i in range(years)]  # Giả sử tăng 5%
        else:
            revenues = [1000000000] * years
    elif not isinstance(revenues, list):
        revenues = [float(revenues)] * years if revenues else [1000000000] * years

    if isinstance(costs, list) and len(costs) != years:
        if costs:
            cost0 = float(costs[0]) if costs[0] else 800000000
            costs = [cost0 * (1.03 ** i) for i in range(years)]  # Giả sử tăng 3%
        else:
            costs = [800000000] * years
    elif not isinstance(costs, list):
        costs = [float(costs)] * years if costs else [800000000] * years

    cash_flows = [-float(investment)]
    for i in range(years):
        ebit = revenues[i] - costs[i]
        tax = ebit * tax_rate if ebit > 0 else 0
        net_cf = ebit - tax
        cash_flows.append(net_cf)

    df = pd.DataFrame({
        'Năm': list(range(0, years + 1)),
        'Dòng tiền (VND)': cash_flows
    })
    return df, float(wacc)

# --- Hàm tính IRR không dùng scipy ---
def calculate_irr(cash_flows, tol=1e-6, max_iter=100):
    """Tính IRR bằng phương pháp bisection."""
    if len(cash_flows) < 2:
        return np.nan
    
    low = -0.99
    high = 10.0  # Tăng high để handle IRR cao
    
    for _ in range(max_iter):
        mid = (low + high) / 2
        npv_mid = np.npv(mid, cash_flows)
        if abs(npv_mid) < tol:
            return mid
        if npv_mid > 0:
            low = mid
        else:
            high = mid
    
    return mid  # approximate

# --- Hàm tính toán chỉ số hiệu quả ---
@st.cache_data
def calculate_metrics(cash_flows, wacc):
    """Tính NPV, IRR, PP, DPP."""
    if len(cash_flows) < 2:
        return {}

    # NPV
    npv = np.npv(wacc, cash_flows)

    # IRR
    irr = calculate_irr(cash_flows)

    # PP (Payback Period) - interpolate for precision
    cumulative_cf = np.cumsum(cash_flows)
    if cumulative_cf[-1] < 0:
        pp = np.inf
    else:
        pp_idx = np.argmax(cumulative_cf >= 0)
        if pp_idx == 0:
            pp = 0
        elif pp_idx == len(cash_flows):
            pp = len(cash_flows) - 1
        else:
            prev_cum = cumulative_cf[pp_idx - 1]
            period_cf = cash_flows[pp_idx]
            pp = (pp_idx - 1) + (-prev_cum / period_cf)

    # DPP (Discounted Payback)
    discounted_cf = [cf / (1 + wacc)**t for t, cf in enumerate(cash_flows)]
    cumulative_disc = np.cumsum(discounted_cf)
    if cumulative_disc[-1] < 0:
        dpp = np.inf
    else:
        dpp_idx = np.argmax(cumulative_disc >= 0)
        if dpp_idx == 0:
            dpp = 0
        elif dpp_idx == len(discounted_cf):
            dpp = len(discounted_cf) - 1
        else:
            prev_cum_disc = cumulative_disc[dpp_idx - 1]
            period_disc_cf = discounted_cf[dpp_idx]
            dpp = (dpp_idx - 1) + (-prev_cum_disc / period_disc_cf)

    return {
        'NPV': npv,
        'IRR': irr * 100 if not np.isnan(irr) else np.nan,
        'PP': round(pp, 2) if not np.isinf(pp) else "Chưa hoàn vốn",
        'DPP': round(dpp, 2) if not np.isinf(dpp) else "Chưa hoàn vốn",
        'WACC': wacc * 100
    }

# --- Hàm gọi AI phân tích chỉ số ---
def get_ai_metrics_analysis(metrics, api_key):
    """Gửi metrics đến Gemini để phân tích."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name='gemini-2.5-flash',
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                top_p=0.8,
                max_output_tokens=1024,  # Tăng một chút
            ),
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }
        )

        prompt = f"""
        Bạn là chuyên gia đánh giá dự án kinh doanh. Dựa trên các chỉ số sau, hãy phân tích hiệu quả dự án một cách khách quan (3-4 đoạn ngắn gọn):
        - NPV > 0: Hấp dẫn.
        - IRR > WACC: Tốt.
        - PP < 3 năm: Nhanh.
        - DPP < 4 năm: Hợp lý.

        Chỉ số:
        NPV: {metrics.get('NPV', 0):,.0f} VND
        IRR: {metrics.get('IRR', 0):.2f}%
        PP: {metrics.get('PP', 0)} năm
        DPP: {metrics.get('DPP', 0)} năm
        WACC: {metrics.get('WACC', 0):.2f}%
        """

        response = model.generate_content(prompt)
        
        # Xử lý response an toàn tương tự
        if not response.candidates or len(response.candidates) == 0:
            return "Không có candidates trong response."
        
        candidate = response.candidates[0]
        
        if candidate.safety_ratings:
            for rating in candidate.safety_ratings:
                if rating.probability.name == "BLOCKED":
                    return f"Response bị block bởi safety: {rating.category.name}"
        
        if candidate.finish_reason.name != "STOP":
            return f"Generation không hoàn tất. Finish reason: {candidate.finish_reason.name}"
        
        if candidate.content and candidate.content.parts:
            text = candidate.content.parts[0].text
        else:
            return "Không có content parts trong response."
        
        return text

    except GoogleAPIError as e:
        return f"Lỗi gọi Gemini API: {e}"
    except Exception as e:
        return f"Lỗi: {e}"

# --- UI: Chức năng 1 - Tải và Lọc Dữ liệu ---
st.header("1. Tải File Word và Lọc Dữ liệu Dự án")

uploaded_file = st.file_uploader("Chọn file Word (.docx)", type=['docx'])

if uploaded_file is not None:
    full_text = read_word_file(uploaded_file)
    st.text_area("Nội dung file (preview):", full_text[:1000], height=200)

    if st.button("🔍 Lọc Dữ liệu bằng AI"):
        api_key = st.secrets.get("GEMINI_API_KEY")
        if not api_key:
            st.error("Lỗi: Cấu hình GEMINI_API_KEY trong Streamlit Secrets.")
        else:
            with st.spinner("Đang extract dữ liệu..."):
                extracted = extract_project_data(full_text, api_key)
                if "error" in extracted:
                    st.error(extracted["error"])
                else:
                    st.session_state.extracted_data = extracted
                    st.success("Extract thành công!")
                    st.json(extracted)

# --- Chức năng 2: Xây dựng Bảng Dòng Tiền ---
if st.session_state.extracted_data and "error" not in st.session_state.extracted_data:
    st.header("2. Bảng Dòng Tiền Dự Án")
    if st.button("📊 Xây dựng Bảng Dòng Tiền"):
        cash_flow_df, wacc = build_cash_flow(st.session_state.extracted_data)
        if cash_flow_df is not None:
            st.session_state.cash_flow_df = cash_flow_df
            st.session_state.wacc = wacc
            st.dataframe(cash_flow_df.style.format({'Dòng tiền (VND)': '{:,.0f}'}), use_container_width=True)

# --- Chức năng 3: Tính Toán Chỉ Số ---
if st.session_state.cash_flow_df is not None:
    st.header("3. Các Chỉ Số Đánh Giá Hiệu Quả")
    cash_flows = st.session_state.cash_flow_df['Dòng tiền (VND)'].tolist()
    wacc_rate = st.session_state.wacc

    metrics = calculate_metrics(cash_flows, wacc_rate)
    st.session_state.metrics = metrics

    col1, col2 = st.columns(2)
    with col1:
        st.metric("NPV", f"{metrics['NPV']:,.0f} VND")
        irr_val = f"{metrics['IRR']:.2f}%" if not np.isnan(metrics['IRR']) else "N/A"
        st.metric("IRR", irr_val)
    with col2:
        st.metric("PP", f"{metrics['PP']}" if isinstance(metrics['PP'], str) else f"{metrics['PP']} năm")
        st.metric("DPP", f"{metrics['DPP']}" if isinstance(metrics['DPP'], str) else f"{metrics['DPP']} năm")

# --- Chức năng 4: Phân Tích AI ---
if st.session_state.metrics:
    st.header("4. Phân Tích Chỉ Số bởi AI")
    if st.button("🤖 Yêu Cầu AI Phân Tích"):
        api_key = st.secrets.get("GEMINI_API_KEY")
        if api_key:
            with st.spinner("Đang phân tích..."):
                analysis = get_ai_metrics_analysis(st.session_state.metrics, api_key)
                st.session_state.ai_analysis = analysis
                st.markdown("**Phân tích từ AI:**")
                if "error" in analysis.lower() or "lỗi" in analysis.lower():
                    st.error(analysis)
                else:
                    st.info(analysis)
        else:
            st.error("Lỗi: Cấu hình GEMINI_API_KEY.")

# --- Export ---
if st.session_state.cash_flow_df is not None:
    col_export1, col_export2 = st.columns(2)
    with col_export1:
        if st.button("📥 Tải Bảng Dòng Tiền Excel"):
            output = io.BytesIO()
            st.session_state.cash_flow_df.to_excel(output, index=False)
            output.seek(0)
            st.download_button(
                label="Tải file",
                data=output.getvalue(),
                file_name="bang_dong_tien.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    with col_export2:
        if st.button("📥 Tải Metrics Excel"):
            # Format metrics for export
            export_metrics = {k: v if not isinstance(v, str) else v for k, v in st.session_state.metrics.items()}
            metrics_df = pd.DataFrame(list(export_metrics.items()), columns=['Chỉ số', 'Giá trị'])
            output = io.BytesIO()
            metrics_df.to_excel(output, index=False)
            output.seek(0)
            st.download_button(
                label="Tải file",
                data=output.getvalue(),
                file_name="chi_so_metrics.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
else:
    st.info("Vui lòng tải file Word và thực hiện các bước để bắt đầu.")
