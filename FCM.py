import streamlit as st
import pandas as pd
import requests
import string
from io import BytesIO

# --- 嘗試匯入備用套件 ---
try:
    import xlsxwriter
except ImportError:
    pass

st.set_page_config(page_title="AI 文獻分析 (省油版)", layout="wide", page_icon="🍃")

# ==========================================
# 🛑 左側邊欄：設定區
# ==========================================
with st.sidebar:
    st.header("🍃 設定")
    st.info("此版本移除了所有自動掃描功能，以節省您的 API 額度。")
    
    # 1. 讓使用者貼上 Key (方便更換)
    user_key_input = st.text_input("請貼上 Google API Key：", type="password")
    
    # 如果沒填，就用程式碼預設的 (但建議你填新的)
    DEFAULT_KEY = "AIzaSyBlj24gBVr3RJhkukS9p6yo5s2-WVBH2H0" 
    
    final_key = user_key_input if user_key_input else DEFAULT_KEY
    
    # 2. 硬性選單 (不浪費額度去問 Google)
    # 這些是 Google 官方公告絕對存在的模型名單
    model_options = [
        "gemini-1.5-flash",  # 首選 (額度最高)
        "gemini-1.5-pro",    # 次選
        "gemini-1.0-pro"     # 備用
    ]
    selected_model = st.selectbox("請選擇模型：", model_options)
    
    st.divider()
    st.markdown("### 💡 狀態提示")
    if not user_key_input:
        st.caption("目前使用：預設金鑰 (若出現 429 請更換)")
    else:
        st.success("目前使用：您手動輸入的新金鑰")

# ==========================================
# 👉 右側主畫面
# ==========================================
st.title("📄 文獻分析工作區")
st.markdown(f"當前鎖定：`{selected_model}`")

raw_text = st.text_area("請在此貼上文獻資料 (每篇請換行)：", height=300)

# --- 分析函數 (直連 + 錯誤處理) ---
def run_analysis_saving_mode(text, model, key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    headers = {'Content-Type': 'application/json'}
    
    prompt = f"""
    任務：歸納 10 個學術研究構面關鍵字。
    規則：只列出名詞，用頓號隔開。排除無關詞彙(如日期、下午)。
    內容：{text[:8000]}
    """
    
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    
    try:
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            return "OK", response.json()['candidates'][0]['content']['parts'][0]['text']
        elif response.status_code == 429:
            return "429", "錯誤：額度已滿 (Resource Exhausted)。請更換 API Key 或等待 10 分鐘。"
        elif response.status_code == 404:
            return "404", f"錯誤：找不到模型 {model} (可能金鑰權限不足)。"
        elif response.status_code == 400:
            return "400", "錯誤：API Key 無效 (Bad Request)。"
        else:
            return "ERR", f"未知錯誤 ({response.status_code}): {response.text}"
            
    except Exception as e:
        return "ERR", f"連線錯誤: {str(e)}"

# --- 輔助函數 ---
def parse_text(text):
    lines = text.strip().split('\n')
    return [{"title": line[:15], "content": line} for line in lines if len(line) > 5]

if st.button("🚀 開始分析 (只耗費 1 次額度)", type="primary"):
    if not raw_text:
        st.warning("請先輸入資料！")
    else:
        with st.spinner("正在呼叫 Google AI..."):
            status, result = run_analysis_saving_mode(raw_text, selected_model, final_key)
            
            if status == "OK":
                st.success("✅ 分析成功！")
                keywords = [k.strip() for k in result.replace("\n", "、").split("、") if k.strip()]
                
                # --- 下面是製表邏輯 ---
                final_keywords = st.multiselect("分析準則", options=keywords, default=keywords)
                if final_keywords:
                    lit_data = parse_text(raw_text)
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
                    try:
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            df.to_excel(writer, sheet_name='矩陣')
                            df_legend.to_excel(writer, sheet_name='對照表')
                        st.download_button("📥 下載 Excel", output.getvalue(), "analysis.xlsx")
                    except:
                        st.download_button("📥 下載 CSV", df.to_csv().encode('utf-8-sig'), "analysis.csv")
            
            else:
                # 顯示錯誤訊息
                st.error(result)
                if status == "429":
                    st.info("💡 建議：去 Google AI Studio 申請一把新的 Key，貼在左側欄位即可立刻復活。")
