import streamlit as st
import pandas as pd
import requests
import string
import re
from collections import Counter
from io import BytesIO

# --- 嘗試匯入備用套件 ---
try:
    import xlsxwriter
    import jieba
    import jieba.analyse
except ImportError:
    pass

# --- 1. 設定 API Key (已內建) ---
USER_API_KEY = "AIzaSyBlj24gBVr3RJhkukS9p6yo5s2-WVBH2H0" 

# --- 2. 頁面設定 ---
st.set_page_config(page_title="AI 文獻分析 (分工版)", layout="wide", page_icon="🎛️")

# ==========================================
# 🛑 左側邊欄：設定與連線檢查 (這裡先做！)
# ==========================================
with st.sidebar:
    st.header("🎛️ 1. 模型設定")
    st.info("請先在此選擇模型，確認連線成功後，再到右邊貼資料。")
    
    # 讓使用者選擇要用哪一個版本
    model_option = st.radio(
        "請選擇 AI 版本：",
        ("Gemini 1.5 Flash (快速/新版)", "Gemini Pro (穩定/舊版)", "本機演算法 (備用/無額度限制)")
    )
    
    st.divider()
    st.subheader("📡 連線狀態")
    
    # 根據選擇的模型定義 API 網址
    target_model_name = ""
    if "Flash" in model_option:
        target_model_name = "gemini-1.5-flash"
    elif "Pro" in model_option:
        target_model_name = "gemini-pro"
    
    # 自動測試連線邏輯
    connection_status = st.empty() # 佔位符
    
    if "本機" in model_option:
        connection_status.success("✅ 本機模式：隨時可用 (無需連網)")
        active_mode = "LOCAL"
    else:
        # 測試按鈕
        if st.button("按此測試連線"):
            with st.spinner("連線檢查中..."):
                try:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model_name}:generateContent?key={USER_API_KEY}"
                    headers = {'Content-Type': 'application/json'}
                    data = {"contents": [{"parts": [{"text": "Hi"}]}]}
                    resp = requests.post(url, headers=headers, json=data)
                    
                    if resp.status_code == 200:
                        connection_status.success(f"✅ 連線成功！\n({target_model_name})")
                        active_mode = "GOOGLE"
                    elif resp.status_code == 429:
                        connection_status.error("❌ 額度滿了 (429)")
                        active_mode = "QUOTA_FULL"
                    else:
                        connection_status.error(f"❌ 失敗: {resp.status_code}")
                        active_mode = "ERROR"
                except Exception as e:
                    connection_status.error("❌ 網絡錯誤")
                    active_mode = "ERROR"
        else:
            connection_status.warning("⚠️ 請點擊上方按鈕測試")
            active_mode = "GOOGLE" # 預設先給過，等下執行再擋

# ==========================================
# 👉 右側主畫面：輸入與分析
# ==========================================
st.title("📄 文獻分析工作區")
st.markdown(f"**目前選擇模式：** `{model_option}`")

# 輸入區
raw_text = st.text_area("請在此貼上資料 (每篇請換行)：", height=250, placeholder="資料輸入區...")

# --- 函數區 ---

def run_google_analysis(text, model_name):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={USER_API_KEY}"
    headers = {'Content-Type': 'application/json'}
    prompt = f"任務：歸納 10 個學術研究構面關鍵字。規則：只列出名詞，用頓號隔開。排除無關詞彙(如日期、下午)。內容：{text[:5000]}"
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        else:
            return None
    except:
        return None

def run_local_analysis(text):
    try:
        return jieba.analyse.extract_tags(text, topK=15, allowPOS=('n', 'vn', 'v'))
    except:
        clean = re.sub(r'[^\u4e00-\u9fa5]', '', text)
        words = [clean[i:i+2] for i in range(len(clean)-1)]
        return [w for w, c in Counter(words).most_common(15)]

def parse_text(text):
    lines = text.strip().split('\n')
    return [{"title": line[:15], "content": line} for line in lines if len(line) > 5]

# --- 執行按鈕 ---
if st.button("🚀 開始分析", type="primary"):
    if not raw_text:
        st.warning("請先輸入資料！")
    else:
        st.divider()
        result_text = None
        keywords = []
        
        # 根據左邊的設定來跑
        if "本機" in model_option:
            with st.spinner("正在使用本機演算法計算..."):
                keywords = run_local_analysis(raw_text)
                st.success("✅ 本機分析完成")
        else:
            # Google 模式
            with st.spinner(f"正在呼叫 {target_model_name} ..."):
                ai_res = run_google_analysis(raw_text, target_model_name)
                if ai_res:
                    st.success("✅ AI 分析完成")
                    keywords = [k.strip() for k in ai_res.replace("\n", "、").split("、") if k.strip()]
                else:
                    st.error("❌ AI 連線失敗或額度已滿，自動切換至本機演算法救援...")
                    keywords = run_local_analysis(raw_text)

        # --- 顯示結果 (共用) ---
        if keywords:
            # 篩選器
            final_keywords = st.multiselect("分析準則 (可調整)", options=keywords, default=keywords)
            
            if final_keywords:
                lit_data = parse_text(raw_text)
                matrix = {}
                labels = []
                titles = []
                
                for i, item in enumerate(lit_data):
                    lbl = string.ascii_uppercase[i % 26]
                    labels.append(lbl)
                    titles.append(item['title'])
                    col_res = ["○" if k in item['content'] else "" for k in final_keywords]
                    matrix[lbl] = col_res
                
                df = pd.DataFrame(matrix, index=final_keywords)
                df_legend = pd.DataFrame({"代號": labels, "文獻": titles})
                
                c1, c2 = st.columns([2, 1])
                with c1: st.dataframe(df, use_container_width=True)
                with c2: st.dataframe(df_legend, hide_index=True)
                
                # 下載
                output = BytesIO()
                try:
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df.to_excel(writer, sheet_name='矩陣')
                        df_legend.to_excel(writer, sheet_name='對照表')
                    st.download_button("📥 下載 Excel", output.getvalue(), "analysis.xlsx")
                except:
                    st.download_button("📥 下載 CSV", df.to_csv().encode('utf-8-sig'), "analysis.csv")
