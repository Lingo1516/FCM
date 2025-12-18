import streamlit as st
import pandas as pd
import requests
import string
from io import BytesIO

# --- 1. 基礎設定 ---
st.set_page_config(page_title="學術文獻分析器 (正式版)", layout="wide", page_icon="🎓")

# --- 2. 側邊欄：設定鑰匙 ---
with st.sidebar:
    st.header("🔑 設定")
    st.info("請貼上剛剛診斷通過的那把鑰匙 (結尾是 WY0iw)")
    
    # 讓使用者貼上 Key
    api_key = st.text_input("Google API Key", type="password")
    
    # 模型選擇 (預設用最穩的 Flash)
    model_name = st.selectbox("選擇模型", ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"])

# --- 3. 主畫面 ---
st.title("📄 學術文獻分析工作區")

# 輸入區
raw_text = st.text_area("請在此貼上文獻資料 (每篇請換行)：", height=300, placeholder="貼上你的論文摘要...")

# --- 4. 核心功能函數 ---
def run_analysis(text, key, model):
    # 這是剛剛診斷通過的連線方式
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    headers = {'Content-Type': 'application/json'}
    
    prompt = f"""
    任務：歸納 10 到 15 個最重要的「研究構面」或「評估準則」。
    規則：
    1. 只輸出名詞 (例如：滿意度、獲利能力)。
    2. 用頓號「、」隔開。
    3. 嚴格排除：日期、下午、作者名、報告、研究方法。
    
    內容：
    {text[:8000]}
    """
    
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            return "OK", response.json()['candidates'][0]['content']['parts'][0]['text']
        else:
            return "ERROR", f"連線錯誤 (代碼 {response.status_code}): {response.text}"
    except Exception as e:
        return "ERROR", str(e)

def parse_literature(text):
    lines = text.strip().split('\n')
    return [{"title": line[:15] + "..." if len(line)>15 else line, "content": line} for line in lines if len(line) > 5]

# --- 5. 執行按鈕 (修復了跳回原畫面的問題) ---
if st.button("🚀 開始分析", type="primary"):
    if not api_key:
        st.error("❌ 請先在左側貼上 API Key！")
    elif not raw_text:
        st.warning("⚠️ 請先輸入文獻資料！")
    else:
        with st.spinner("🤖 AI 正在閱讀文獻..."):
            status, result = run_analysis(raw_text, api_key, model_name)
            
            if status == "OK":
                st.success("✅ 分析完成！")
                
                # A. 處理關鍵字
                keywords = [k.strip() for k in result.replace("\n", "、").split("、") if k.strip()]
                
                # B. 讓使用者篩選
                st.subheader("1️⃣ AI 建議的構面")
                final_keywords = st.multiselect("請勾選要保留的項目：", options=keywords, default=keywords)
                
                if final_keywords:
                    # C. 建立矩陣
                    lit_data = parse_literature(raw_text)
                    matrix = {}
                    labels = []
                    titles = []
                    
                    for i, item in enumerate(lit_data):
                        lbl = string.ascii_uppercase[i % 26]
                        labels.append(lbl)
                        titles.append(item['title'])
                        # 判斷該文獻是否包含該關鍵字
                        matrix[lbl] = ["●" if k in item['content'] else "" for k in final_keywords]
                    
                    # D. 顯示結果
                    df_matrix = pd.DataFrame(matrix, index=final_keywords)
                    df_legend = pd.DataFrame({"代號": labels, "文獻標題": titles})
                    
                    st.divider()
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.subheader("📊 分析矩陣")
                        st.dataframe(df_matrix, use_container_width=True)
                        
                    with col2:
                        st.subheader("📝 文獻對照表")
                        st.dataframe(df_legend, hide_index=True, use_container_width=True)
                    
                    # E. 下載按鈕
                    output = BytesIO()
                    try:
                        import xlsxwriter
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            df_matrix.to_excel(writer, sheet_name='分析矩陣')
                            df_legend.to_excel(writer, sheet_name='對照表')
                        file_name = "ai_analysis_result.xlsx"
                        mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    except ImportError:
                        # 備用 CSV
                        output.write(df_matrix.to_csv().encode('utf-8-sig'))
                        file_name = "ai_analysis_result.csv"
                        mime_type = "text/csv"
                        
                    st.download_button(
                        label=f"📥 下載報表 ({file_name})",
                        data=output.getvalue(),
                        file_name=file_name,
                        mime=mime_type,
                        type="primary"
                    )
            else:
                st.error(result)
