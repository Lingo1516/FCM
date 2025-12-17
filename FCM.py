import streamlit as st
import pandas as pd
import re
import string
from io import BytesIO

# --- 嘗試匯入 AI 套件 ---
try:
    from groq import Groq
    import google.generativeai as genai
except ImportError:
    st.error("請先安裝必要套件: pip install groq google-generativeai pandas streamlit")

# --- 頁面設定 ---
st.set_page_config(page_title="AI 智慧文獻分析器", layout="wide", page_icon="🧠")

st.title("🧠 AI 智慧文獻分析器 (自動萃取關鍵字版)")
st.markdown("### 真正的全自動：丟入文字 -> AI 自動判斷領域與關鍵字 -> 生成表格")

# --- 1. 側邊欄：AI 設定 (因為要用 AI 判斷關鍵字，需要 Key) ---
with st.sidebar:
    st.header("1. 設定 AI 金鑰")
    st.info("為了讓程式能「讀懂」你的不同領域資料 (數學/商業/科技)，需要使用 AI 模型。")
    
    engine_choice = st.radio("選擇 AI 模型", ["Groq (Llama 3)", "Google (Gemini)"])
    
    api_key = ""
    if engine_choice == "Groq (Llama 3)":
        api_key = st.text_input("Groq API Key", type="password")
    else:
        api_key = st.text_input("Google API Key", type="password")

# --- 2. 主畫面：輸入資料 ---
st.header("2. 輸入原始文獻資料")
raw_text = st.text_area("請在此貼上所有雜亂的文獻文字：", height=250, placeholder="直接把 Word 或網頁內容全部貼進來，包含作者、年份、摘要...")

# --- 3. 核心功能：AI 自動萃取關鍵字 ---
st.header("3. AI 自動分析關鍵字")
st.markdown("在此步驟，AI 會閱讀你的文字，自動決定該分析哪些重點。")

# 用 session_state 記住分析出來的關鍵字，避免重整後消失
if 'ai_keywords' not in st.session_state:
    st.session_state.ai_keywords = ""

def extract_keywords_with_ai(text, engine, key):
    prompt = f"""
    任務：你是學術分析專家。請閱讀以下文獻資料，分析這些文獻共同探討的「核心準則」或「評估構面」。
    
    資料內容：
    {text[:5000]}  # 避免超過 token 上限，取前5000字
    
    要求：
    1. 不管領域是數學、商業或科技，請自動歸納出 15 到 20 個最重要的分析關鍵字。
    2. 只輸出關鍵字，用繁體中文，並用換行分隔。
    3. 不要輸出其他廢話或解釋。
    """
    
    try:
        if engine == "Groq (Llama 3)":
            client = Groq(api_key=key)
            completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
            )
            return completion.choices[0].message.content
        elif engine == "Google (Gemini)":
            genai.configure(api_key=key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            return response.text
    except Exception as e:
        return f"錯誤: {str(e)}"

col_btn, col_result = st.columns([1, 3])

with col_btn:
    if st.button("🔍 開始 AI 智能分析", type="primary"):
        if not api_key:
            st.error("請先在左側輸入 API Key")
        elif not raw_text:
            st.error("請先輸入文獻資料")
        else:
            with st.spinner("AI 正在研讀您的資料並歸納重點..."):
                keywords_result = extract_keywords_with_ai(raw_text, engine_choice, api_key)
                st.session_state.ai_keywords = keywords_result
                st.success("分析完成！")

with col_result:
    # 讓使用者可以編輯 AI 抓出來的結果
    final_criteria_str = st.text_area(
        "AI 抓到的關鍵字 (您可手動修改)：", 
        value=st.session_state.ai_keywords, 
        height=200,
        help="AI 自動產生的列表，您可以刪除不準的，或自己補上新的。"
    )

# --- 4. 生成矩陣 (使用動態抓取的關鍵字) ---
st.divider()
st.header("4. 生成分析矩陣")

# 解析文字的函數 (沿用之前的邏輯，因為這對於抓作者很有效)
def smart_parse_text(text):
    # 抓取 (20xx) 作為分隔點
    pattern = r'([^\n\r。]+?[\(\[\{](?:19|20)\d{2}[\)\]\}])'
    segments = re.split(pattern, text)
    parsed_data = []
    current_author = None
    
    for segment in segments:
        segment = segment.strip()
        if not segment: continue
        if re.search(r'[\(\[\{](?:19|20)\d{2}[\)\]\}]', segment):
            current_author = segment
            if len(current_author) > 50: current_author = current_author[-50:]
        else:
            if current_author:
                parsed_data.append({"author": current_author, "abstract": segment})
    return parsed_data

if st.button("📊 根據上方關鍵字生成圖表"):
    if not final_criteria_str or not raw_text:
        st.warning("請確保已經有文獻資料，並且已經產生(或輸入)了關鍵字。")
    else:
        # 1. 整理關鍵字
        criteria_list = [c.strip() for c in final_criteria_str.split('\n') if c.strip()]
        
        # 2. 解析文獻
        parsed_items = smart_parse_text(raw_text)
        
        if not parsed_items:
            st.error("無法從文字中辨識出作者與年份。請確認文字包含如 (2023) 的格式。")
        else:
            # 3. 建立矩陣
            labels = []
            authors_list = []
            matrix_data = {}
            
            # 產生代號生成器
            def get_label(index):
                if index < 26: return string.ascii_uppercase[index]
                else: return f"{string.ascii_uppercase[index // 26 - 1]}{string.ascii_uppercase[index % 26]}"
            
            for i, item in enumerate(parsed_items):
                label = get_label(i)
                labels.append(label)
                authors_list.append(item['author'])
                
                abstract_content = item['abstract']
                col_results = []
                
                for criterion in criteria_list:
                    # 這裡可以再進化：如果不只是關鍵字比對，而是要 AI 判斷「語意」是否符合，
                    # 那會需要花更多 Token，這裡先用關鍵字比對以確保速度
                    if criterion in abstract_content:
                        col_results.append("○")
                    else:
                        col_results.append("")
                
                matrix_data[label] = col_results
            
            # 4. 顯示結果
            df_matrix = pd.DataFrame(matrix_data, index=criteria_list)
            df_matrix.index.name = "構面/準則"
            
            df_legend = pd.DataFrame({"代號": labels, "作者": authors_list})
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("分析結果")
                st.dataframe(df_matrix, use_container_width=True, height=600)
            with col2:
                st.subheader("文獻對照")
                st.dataframe(df_legend, hide_index=True, use_container_width=True)
                
            # 下載功能
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_matrix.to_excel(writer, sheet_name='矩陣')
                df_legend.to_excel(writer, sheet_name='作者')
            st.download_button("📥 下載 Excel", output.getvalue(), "analysis.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
