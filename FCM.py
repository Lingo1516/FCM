import streamlit as st
import pandas as pd
import requests
import string
import re
from collections import Counter
from io import BytesIO

# --- 嘗試匯入備用套件 (防呆) ---
try:
    import xlsxwriter
    import jieba
    import jieba.analyse
except ImportError:
    pass

# --- 1. 設定 API Key ---
USER_API_KEY = "AIzaSyBlj24gBVr3RJhkukS9p6yo5s2-WVBH2H0" 

# --- 2. 頁面設定 ---
st.set_page_config(page_title="AI 模型掃描與分析", layout="wide", page_icon="📡")

# 初始化 Session State (用來記住掃描到的模型，才不會一直重跑)
if 'available_models' not in st.session_state:
    st.session_state.available_models = []
if 'scan_done' not in st.session_state:
    st.session_state.scan_done = False

# ==========================================
# 🛑 左側邊欄：模型掃描站
# ==========================================
with st.sidebar:
    st.header("📡 第一步：模型掃描")
    st.info("請先點擊下方按鈕，搜尋目前可用的 Google AI 模型。")
    
    # 掃描函數
    def scan_google_models(key):
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                # 篩選出支援 generateContent 的 gemini 模型
                valid_list = []
                for m in data.get('models', []):
                    if 'generateContent' in m.get('supportedGenerationMethods', []) and 'gemini' in m['name']:
                        # 只取名字，去掉 'models/' 前綴讓畫面好看點
                        friendly_name = m['name'].replace("models/", "")
                        valid_list.append(friendly_name)
                return valid_list
            else:
                return []
        except:
            return []

    # 掃描按鈕
    if st.button("🔍 立即掃描可用模型", type="primary"):
        with st.spinner("正在連線 Google 伺服器查詢名單..."):
            found_models = scan_google_models(USER_API_KEY)
            
            if found_models:
                st.session_state.available_models = found_models
                st.session_state.scan_done = True
                st.success(f"掃描完成！找到 {len(found_models)} 個模型。")
            else:
                st.error("❌ 掃描失敗：無法連線或金鑰無效。")
                st.session_state.available_models = []
    
    st.divider()
    
    # 顯示選擇選單 (只有掃描成功才會出現)
    selected_model = None
    if st.session_state.scan_done and st.session_state.available_models:
        st.subheader("✅ 請選擇一個模型：")
        selected_model = st.radio(
            "建議選擇 Flash (快) 或 Pro (穩)：",
            st.session_state.available_models,
            index=0 # 預設選第一個
        )
        st.caption(f"目前已鎖定：`{selected_model}`")
    elif st.session_state.scan_done and not st.session_state.available_models:
        st.warning("⚠️ 無法使用 Google 模型，將自動切換至「本機演算法」。")
        selected_model = "Local (本機備用)"
    else:
        st.markdown("等待掃描中...")

# ==========================================
# 👉 右側主畫面：只有選好模型才會顯示
# ==========================================
st.title("📄 文獻分析工作區")

if not st.session_state.scan_done:
    # 尚未掃描時的畫面
    st.info("⬅️ 請先在左側點擊 **「🔍 立即掃描可用模型」** 開始。")
    st.markdown("這樣可以確保我們找到一個「有空」的模型，避免輸入資料後才發現連線失敗。")

else:
    # 掃描完成，顯示輸入框
    st.success(f"🚀 系統準備就緒！目前使用核心：**{selected_model if selected_model else '本機演算法'}**")
    
    raw_text = st.text_area("請在此貼上文獻資料 (每篇請換行)：", height=300, placeholder="將摘要貼在這裡...")

    # --- 分析函數 ---
    def run_analysis(text, model_name):
        # 如果是本機模式
        if model_name == "Local (本機備用)":
            try:
                return jieba.analyse.extract_tags(text, topK=15, allowPOS=('n', 'vn', 'v'))
            except:
                clean = re.sub(r'[^\u4e00-\u9fa5]', '', text)
                words = [clean[i:i+2] for i in range(len(clean)-1)]
                return [w for w, c in Counter(words).most_common(15)]
        
        # 如果是 Google 模式
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={USER_API_KEY}"
        headers = {'Content-Type': 'application/json'}
        prompt = f"任務：歸納 10 個學術研究構面關鍵字。規則：只列出名詞，用頓號隔開。排除無關詞彙(如日期、下午)。內容：{text[:5000]}"
        data = {"contents": [{"parts": [{"text": prompt}]}]}
        
        try:
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                return response.json()['candidates'][0]['content']['parts'][0]['text']
            elif response.status_code == 429:
                return "QUOTA_FULL"
            else:
                return None
        except:
            return None

    def parse_text(text):
        lines = text.strip().split('\n')
        return [{"title": line[:15], "content": line} for line in lines if len(line) > 5]

    # 執行按鈕
    if st.button("🚀 開始分析", type="primary"):
        if not raw_text:
            st.warning("請先輸入資料！")
        else:
            keywords = []
            
            with st.spinner(f"正在使用 {selected_model} 進行分析..."):
                result = run_analysis(raw_text, selected_model)
                
                if result == "QUOTA_FULL":
                    st.error("❌ 哎呀！這個模型的額度剛好滿了 (429)。")
                    st.info("💡 建議：請在左側換另一個模型試試看（例如從 Flash 換成 Pro）。")
                elif result and isinstance(result, str):
                    # Google 成功回傳字串
                    keywords = [k.strip() for k in result.replace("\n", "、").split("、") if k.strip()]
                    st.success("✅ AI 分析成功！")
                elif isinstance(result, list):
                    # 本機回傳列表
                    keywords = result
                    st.success("✅ 本機運算成功！")
                else:
                    st.error("❌ 連線發生未知錯誤，請嘗試切換其他模型。")

            # --- 顯示結果 ---
            if keywords:
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
                    
                    st.divider()
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
