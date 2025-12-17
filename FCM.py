import streamlit as st
import pandas as pd
import string
from io import BytesIO

# --- 嘗試匯入必要的套件 ---
try:
    import google.generativeai as genai
    import xlsxwriter
except ImportError:
    # 這裡只是為了防止本地端執行報錯，雲端只要 requirements.txt 對了就不會進來這裡
    st.error("環境安裝中...請稍候並重新整理")
    st.stop()

# --- 1. 設定您的 API Key ---
USER_API_KEY = "AIzaSyBlj24gBVr3RJhkukS9p6yo5s2-WVBH2H0" # 你的 Key

if USER_API_KEY:
    genai.configure(api_key=USER_API_KEY)

st.set_page_config(page_title="AI 文獻分析器", layout="wide", page_icon="🤖")
st.title("🤖 AI 文獻分析器 (Gemini 1.5 Flash)")

# --- 2. 測試連線按鈕 ---
if st.button("📡 測試連線 (Ping)"):
    try:
        # 使用 1.5 Flash 模型，這需要新版套件
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content("Hi")
        st.success(f"✅ 連線成功！版本正確！")
    except Exception as e:
        st.error(f"❌ 連線失敗: {e}")
        st.warning("請執行「Reboot App」或「刪除 App 重新部署」來強制更新環境。")

st.info("👇 請貼上文獻資料")
raw_text = st.text_area("文獻輸入區", height=200)

# --- 3. 分析函數 ---
def get_ai_analysis(text):
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"歸納 10 個學術構面名詞，用頓號隔開：{text[:5000]}"
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {e}"

def parse_text(text):
    lines = text.strip().split('\n')
    return [{"title": line[:15], "content": line} for line in lines if len(line) > 5]

if st.button("🚀 開始分析", type="primary"):
    if not raw_text:
        st.warning("請貼上資料")
    else:
        with st.spinner("AI 分析中..."):
            ai_result = get_ai_analysis(raw_text)
            lit_data = parse_text(raw_text)
            
            if "Error" in ai_result:
                st.error(ai_result)
            else:
                keywords = [k.strip() for k in ai_result.replace("\n", "、").split("、") if k.strip()]
                final_keywords = st.multiselect("AI 抓到的準則", options=keywords, default=keywords)
                
                if final_keywords:
                    matrix = {}
                    labels = []
                    titles = []
                    for i, item in enumerate(lit_data):
                        lbl = string.ascii_uppercase[i % 26]
                        labels.append(lbl)
                        titles.append(item['title'])
                        matrix[lbl] = ["○" if k in item['content'] else "" for k in final_keywords]
                    
                    df = pd.DataFrame(matrix, index=final_keywords)
                    df_legend = pd.DataFrame({"代號": labels, "文獻": titles})
                    
                    c1, c2 = st.columns([2, 1])
                    with c1: st.dataframe(df, use_container_width=True)
                    with c2: st.dataframe(df_legend, hide_index=True)
                    
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df.to_excel(writer, sheet_name='矩陣')
                        df_legend.to_excel(writer, sheet_name='對照表')
                    st.download_button("📥 下載 Excel", output.getvalue(), "analysis.xlsx")
