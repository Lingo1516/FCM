import streamlit as st
import pandas as pd
import requests
import string
import re
from collections import Counter
from io import BytesIO

# --- 嘗試匯入備用套件 (Jieba) ---
# 這是本機運算的關鍵，萬一 Google 失敗就靠它
try:
    import xlsxwriter
    import jieba
    import jieba.analyse
except ImportError:
    pass

st.set_page_config(page_title="全自動補位分析器", layout="wide", page_icon="🛡️")

# ==========================================
# 🛑 左側設定區
# ==========================================
with st.sidebar:
    st.header("🛡️ 設定")
    st.info("此版本為「不死鳥」模式：若 Google 連線失敗，系統會自動切換至本機運算，確保您一定能拿到結果。")
    
    # 讓使用者輸入 Key
    user_key = st.text_input("Google API Key (選填)", type="password")
    
    # 預設備用 Key (舊的，雖可能已滿但備著)
    DEFAULT_KEY = "AIzaSyBlj24gBVr3RJhkukS9p6yo5s2-WVBH2H0"
    target_key = user_key if user_key else DEFAULT_KEY

# ==========================================
# 👉 主畫面
# ==========================================
st.title("📄 文獻分析工作區 (保證產出版)")

raw_text = st.text_area("請在此貼上文獻資料 (每篇請換行)：", height=300)

# --- 1. 本機演算法 (Jieba) - 這是最強的備胎 ---
def run_local_jieba(text):
    # 如果有裝 jieba 就用 jieba，沒裝就用簡單統計
    try:
        tags = jieba.analyse.extract_tags(text, topK=15, allowPOS=('n', 'vn', 'v'))
        return tags
    except:
        # 最簡陋的斷詞 (每兩個字切一刀)，保證不報錯
        clean = re.sub(r'[^\u4e00-\u9fa5]', '', text)
        words = [clean[i:i+2] for i in range(len(clean)-1)]
        return [w for w, c in Counter(words).most_common(15)]

# --- 2. Google AI 演算法 ---
def run_google_ai(text, key):
    # 嘗試列表，哪個能通就用哪個
    models_to_try = ["gemini-1.5-flash", "gemini-1.0-pro", "gemini-pro"]
    
    for model in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
        headers = {'Content-Type': 'application/json'}
        prompt = f"歸納10個學術構面名詞，用頓號隔開。排除無關詞(日期、下午)：{text[:5000]}"
        data = {"contents": [{"parts": [{"text": prompt}]}]}
        
        try:
            # 設定 3 秒超時，不行就換下一個
            response = requests.post(url, headers=headers, json=data, timeout=3)
            if response.status_code == 200:
                res_text = response.json()['candidates'][0]['content']['parts'][0]['text']
                return "SUCCESS", res_text
            elif response.status_code == 429:
                continue # 這個滿了，試下一個
            elif response.status_code == 404:
                continue # 這個找不到，試下一個
        except:
            continue
            
    return "FAIL", None

# --- 3. 輔助函數 ---
def parse_text(text):
    lines = text.strip().split('\n')
    return [{"title": line[:15], "content": line} for line in lines if len(line) > 5]

# --- 執行按鈕 ---
if st.button("🚀 開始分析", type="primary"):
    if not raw_text:
        st.warning("請先輸入資料！")
    else:
        status_box = st.empty()
        status_box.info("🤖 正在嘗試連線 Google AI...")
        
        # 1. 先試 Google
        status, ai_result = run_google_ai(raw_text, target_key)
        
        final_keywords = []
        used_source = ""
        
        if status == "SUCCESS":
            status_box.success("✅ Google AI 分析成功！")
            final_keywords = [k.strip() for k in ai_result.replace("\n", "、").split("、") if k.strip()]
            used_source = "Google AI"
        else:
            # 2. Google 失敗，自動切換本機
            status_box.warning("⚠️ Google 連線異常 (404/429)，已自動切換至「本機演算法」完成分析。")
            final_keywords = run_local_jieba(raw_text)
            used_source = "本機演算法 (Jieba)"
            
        # --- 3. 產出結果 (絕對會執行到這裡) ---
        st.divider()
        st.caption(f"本次分析使用核心：{used_source}")
        
        if final_keywords:
            selected_keywords = st.multiselect("分析準則", options=final_keywords, default=final_keywords)
            
            if selected_keywords:
                lit_data = parse_text(raw_text)
                matrix = {}
                labels = []
                titles = []
                
                for i, item in enumerate(lit_data):
                    lbl = string.ascii_uppercase[i % 26]
                    labels.append(lbl)
                    titles.append(item['title'])
                    matrix[lbl] = ["○" if k in item['content'] else "" for k in selected_keywords]
                
                df = pd.DataFrame(matrix, index=selected_keywords)
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
