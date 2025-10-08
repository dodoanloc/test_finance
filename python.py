# python.py

import streamlit as st
import pandas as pd
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import io  # Để export file

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Phân Tích Báo Cáo Tài Chính",
    layout="wide"
)

st.title("Ứng dụng Phân Tích Báo Cáo Tài Chính 📊")

# --- Khởi tạo Session State cho Chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "df_processed" not in st.session_state:
    st.session_state.df_processed = None
if "data_for_ai" not in st.session_state:
    st.session_state.data_for_ai = None
if "thanh_toan_hien_hanh_N_1" not in st.session_state:
    st.session_state.thanh_toan_hien_hanh_N_1 = "N/A"
if "thanh_toan_hien_hanh_N" not in st.session_state:
    st.session_state.thanh_toan_hien_hanh_N = "N/A"

# --- Hàm tính toán chính (Sử dụng Caching để Tối ưu hiệu suất) ---
@st.cache_data
def process_financial_data(df):
    """Thực hiện các phép tính Tăng trưởng và Tỷ trọng."""
    
    # Đảm bảo các giá trị là số để tính toán
    numeric_cols = ['Năm trước', 'Năm sau']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 1. Tính Tốc độ Tăng trưởng
    df['Tốc độ tăng trưởng (%)'] = (
        (df['Năm sau'] - df['Năm trước']) / df['Năm trước'].replace(0, 1e-9)
    ) * 100

    # 2. Tính Tỷ trọng theo Tổng Tài sản
    tong_tai_san_row = df[df['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN', case=False, na=False)]
    
    if tong_tai_san_row.empty:
        raise ValueError("Không tìm thấy chỉ tiêu 'TỔNG CỘNG TÀI SẢN'.")

    tong_tai_san_N_1 = tong_tai_san_row['Năm trước'].iloc[0]
    tong_tai_san_N = tong_tai_san_row['Năm sau'].iloc[0]

    # Xử lý lỗi chia cho 0 thủ công
    divisor_N_1 = tong_tai_san_N_1 if tong_tai_san_N_1 != 0 else 1e-9
    divisor_N = tong_tai_san_N if tong_tai_san_N != 0 else 1e-9

    df['Tỷ trọng Năm trước (%)'] = (df['Năm trước'] / divisor_N_1) * 100
    df['Tỷ trọng Năm sau (%)'] = (df['Năm sau'] / divisor_N) * 100
    
    return df

# --- Hàm gọi API Gemini cho Phân tích Ban đầu ---
def get_ai_analysis(data_for_ai, api_key):
    """Gửi dữ liệu phân tích đến Gemini API và nhận nhận xét."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name='gemini-2.5-flash',
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                top_p=0.8,
                max_output_tokens=1000,
            ),
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }
        ) 

        # Prompt cải thiện: Thêm so sánh benchmark ngành (giả sử ngành sản xuất)
        prompt = f"""
        Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. Dựa trên các chỉ số tài chính sau của doanh nghiệp (ngành sản xuất), hãy đưa ra một nhận xét khách quan, ngắn gọn (khoảng 3-4 đoạn) về tình hình tài chính. 
        Đánh giá tập trung vào tốc độ tăng trưởng, thay đổi cơ cấu tài sản và khả năng thanh toán hiện hành. 
        So sánh với benchmark ngành: Tăng trưởng tài sản > 10% là tốt; Thanh toán hiện hành > 1.5 lần là an toàn; Tỷ trọng tài sản ngắn hạn > 30% là cân bằng.
        
        Dữ liệu thô và chỉ số:
        {data_for_ai}
        """

        response = model.generate_content(prompt)
        return response.text

    except GoogleAPIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"

# --- Hàm gọi API Gemini cho Chat ---
def get_ai_chat_response(messages, api_key, df_context=None):
    """Gửi lịch sử chat đến Gemini và nhận phản hồi, với ngữ cảnh dữ liệu nếu có."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name='gemini-2.5-flash',
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                top_p=0.8,
                max_output_tokens=500,
            ),
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }
        )

        # Xây dựng prompt từ lịch sử chat
        chat_history = ""
        for msg in messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            chat_history += f"{role}: {msg['content']}\n"

        # Thêm ngữ cảnh dữ liệu nếu có
        if df_context is not None:
            context = f"""
            Ngữ cảnh dữ liệu tài chính (Bảng phân tích):
            {df_context}
            """
        else:
            context = ""

        prompt = f"""
        Bạn là một trợ lý AI chuyên về phân tích tài chính. Hãy trả lời câu hỏi của người dùng một cách chuyên nghiệp, ngắn gọn và dựa trên dữ liệu nếu có. 
        Nếu câu hỏi liên quan đến dữ liệu tài chính, hãy tham chiếu đến ngữ cảnh sau:
        {context}
        
        Lịch sử chat:
        {chat_history}
        
        Câu hỏi hiện tại: {messages[-1]['content']}
        """

        response = model.generate_content(prompt)
        return response.text

    except GoogleAPIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"

# --- Chức năng 1: Tải File ---
uploaded_file = st.file_uploader(
    "1. Tải file Excel Báo cáo Tài chính (Chỉ tiêu | Năm trước | Năm sau)",
    type=['xlsx', 'xls']
)

if uploaded_file is not None:
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("Đang đọc file...")
        progress_bar.progress(20)
        df_raw = pd.read_excel(uploaded_file)
        
        # Tiền xử lý
        df_raw.columns = ['Chỉ tiêu', 'Năm trước', 'Năm sau']
        progress_bar.progress(50)
        
        status_text.text("Đang xử lý dữ liệu...")
        df_processed = process_financial_data(df_raw.copy())
        st.session_state.df_processed = df_processed
        progress_bar.progress(80)
        
        # --- Chức năng 4: Tính Chỉ số Tài chính (tính trước để dùng cho data_for_ai) ---
        thanh_toan_hien_hanh_N = "N/A"
        thanh_toan_hien_hanh_N_1 = "N/A"
        try:
            # Lấy Tài sản ngắn hạn
            tsnh_row = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]
            if not tsnh_row.empty:
                tsnh_n = tsnh_row['Năm sau'].iloc[0]
                tsnh_n_1 = tsnh_row['Năm trước'].iloc[0]

                # Lấy Nợ ngắn hạn (fallback: nếu không có, dùng tổng nợ phải trả nếu có, hoặc cảnh báo)
                no_ngan_han_row = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]
                if no_ngan_han_row.empty:
                    # Fallback: Tìm 'TỔNG NỢ PHẢI TRẢ' nếu có
                    no_phai_tra_row = df_processed[df_processed['Chỉ tiêu'].str.contains('TỔNG NỢ PHẢI TRẢ', case=False, na=False)]
                    if not no_phai_tra_row.empty:
                        no_ngan_han_N = no_phai_tra_row['Năm sau'].iloc[0]
                        no_ngan_han_N_1 = no_phai_tra_row['Năm trước'].iloc[0]
                        st.warning("Sử dụng 'TỔNG NỢ PHẢI TRẢ' làm fallback cho 'NỢ NGẮN HẠN'.")
                    else:
                        raise IndexError("Không tìm thấy 'NỢ NGẮN HẠN' hoặc 'TỔNG NỢ PHẢI TRẢ'.")
                else:
                    no_ngan_han_N = no_ngan_han_row['Năm sau'].iloc[0]
                    no_ngan_han_N_1 = no_ngan_han_row['Năm trước'].iloc[0]

                # Xử lý chia 0
                divisor_N_1 = no_ngan_han_N_1 if no_ngan_han_N_1 != 0 else 1e-9
                divisor_N = no_ngan_han_N if no_ngan_han_N != 0 else 1e-9

                thanh_toan_hien_hanh_N = tsnh_n / divisor_N
                thanh_toan_hien_hanh_N_1 = tsnh_n_1 / divisor_N_1
                
                st.session_state.thanh_toan_hien_hanh_N_1 = f"{thanh_toan_hien_hanh_N_1:.2f}"
                st.session_state.thanh_toan_hien_hanh_N = f"{thanh_toan_hien_hanh_N:.2f}"
            else:
                raise IndexError("Không tìm thấy 'TÀI SẢN NGẮN HẠN'.")
        except IndexError:
            st.warning("Thiếu chỉ tiêu cần thiết để tính chỉ số thanh toán. Vui lòng kiểm tra file Excel.")
        
        # Chuẩn bị data_for_ai cho phân tích ban đầu và chat
        tsnh_row = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]
        tang_truong_tsnh = tsnh_row['Tốc độ tăng trưởng (%)'].iloc[0] if not tsnh_row.empty else "N/A"
        
        st.session_state.data_for_ai = pd.DataFrame({
            'Chỉ tiêu': [
                'Toàn bộ Bảng phân tích (dữ liệu thô)', 
                'Tăng trưởng Tài sản ngắn hạn (%)', 
                'Thanh toán hiện hành (N-1)', 
                'Thanh toán hiện hành (N)'
            ],
            'Giá trị': [
                df_processed.to_markdown(index=False),
                f"{tang_truong_tsnh:.2f}%", 
                st.session_state.thanh_toan_hien_hanh_N_1, 
                st.session_state.thanh_toan_hien_hanh_N
            ]
        }).to_markdown(index=False)
        
        progress_bar.progress(100)
        status_text.text("Xử lý hoàn tất!")
        st.success("File đã được tải và xử lý thành công.")

        # --- Chức năng 2 & 3: Hiển thị Kết quả ---
        st.subheader("2. Tốc độ Tăng trưởng & 3. Tỷ trọng Cơ cấu Tài sản")
        st.dataframe(df_processed.style.format({
            'Năm trước': '{:,.0f}',
            'Năm sau': '{:,.0f}',
            'Tốc độ tăng trưởng (%)': '{:.2f}%',
            'Tỷ trọng Năm trước (%)': '{:.2f}%',
            'Tỷ trọng Năm sau (%)': '{:.2f}%'
        }), use_container_width=True)
        
        # Nút Export
        col_export1, col_export2 = st.columns(2)
        with col_export1:
            if st.button("📥 Tải kết quả Excel"):
                output = io.BytesIO()
                df_processed.to_excel(output, index=False)
                output.seek(0)
                st.download_button(
                    label="Tải file",
                    data=output.getvalue(),
                    file_name="ket_qua_phan_tich.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        # --- Chức năng 4: Tính Chỉ số Tài chính (hiển thị) ---
        st.subheader("4. Các Chỉ số Tài chính Cơ bản")
        
        if st.session_state.thanh_toan_hien_hanh_N != "N/A":
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    label="Chỉ số Thanh toán Hiện hành (Năm trước)",
                    value=f"{float(st.session_state.thanh_toan_hien_hanh_N_1):.2f} lần"
                )
            with col2:
                st.metric(
                    label="Chỉ số Thanh toán Hiện hành (Năm sau)",
                    value=f"{float(st.session_state.thanh_toan_hien_hanh_N):.2f} lần",
                    delta=f"{float(st.session_state.thanh_toan_hien_hanh_N) - float(st.session_state.thanh_toan_hien_hanh_N_1):.2f}"
                )
        else:
            st.warning("Không thể tính chỉ số thanh toán do thiếu dữ liệu.")
        
        # --- Chức năng 5: Nhận xét AI Ban đầu ---
        st.subheader("5. Nhận xét Tình hình Tài chính (AI)")
        
        if st.button("Yêu cầu AI Phân tích Ban đầu"):
            api_key = st.secrets.get("GEMINI_API_KEY") 
            
            if api_key:
                with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
                    ai_result = get_ai_analysis(st.session_state.data_for_ai, api_key)
                    st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                    st.info(ai_result)
            else:
                st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets.")

        # --- Chức năng 6: Khung Chat với Gemini ---
        st.subheader("6. Chat với AI: Hỏi thêm về Dữ liệu Tài chính")
        st.info("💡 Bạn có thể hỏi thêm: 'Phân tích rủi ro nợ?', 'So sánh với năm trước?', hoặc bất kỳ câu hỏi nào liên quan đến dữ liệu.")
        
        # Hiển thị lịch sử chat
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
            # Thêm user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Gọi AI
            api_key = st.secrets.get("GEMINI_API_KEY")
            if api_key:
                with st.chat_message("assistant"):
                    with st.spinner("Đang suy nghĩ..."):
                        response = get_ai_chat_response(st.session_state.messages, api_key, st.session_state.data_for_ai)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                st.error("Lỗi: Không tìm thấy Khóa API cho chat.")

    except ValueError as ve:
        st.error(f"Lỗi cấu trúc dữ liệu: {ve}")
    except Exception as e:
        st.error(f"Có lỗi xảy ra khi đọc hoặc xử lý file: {e}. Vui lòng kiểm tra định dạng file.")

else:
    st.info("Vui lòng tải lên file Excel để bắt đầu phân tích.")
