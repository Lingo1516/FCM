import streamlit as st
import pandas as pd
import requests
import string
import re
import time
from collections import Counter
from io import BytesIO

# --- 嘗試匯入備用套件 ---
try:
    import xlsxwriter
    import jieba
    import jieba.analyse
except ImportError:
    pass

# --- 1. 設定 API Key ---
USER_API_KEY = "AIzaSyBlj24gBVr3RJhkukS9p6yo5s2-WVBH2H0" 

# --- 2. 頁面設定 ---
st.set_page_config(page_title="AI 溫柔分析版", layout="wide", page_icon="🕊️")

if 'model_list' not in st.session_state:
    st.session_state.model_list = []
if 'list_loaded' not in st.session_state:
    st.session_state.list_loaded = False

# ==========================================
# 🛑 左側邊欄：溫柔選單
# ==========================================
with st.sidebar:
    st.header("🕊️ 第一步：選擇模型")
    st.info("這次我們不暴力測試，而是先列出清單，您選中哪個，我們才測哪個。")
    
    # 1. 獲取清單函數 (不耗額度)
    def fetch_model_list(key):
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                valid_list = []
                for m in data.get('models', []):
                    # 只抓 gemini 系列
                    if 'generateContent' in m.get('supportedGenerationMethods', []) and 'gemini' in m['name']:
                        valid_list.append(m['name'].replace("models/", ""))
                return valid_list
            else:
                return []
        except:
            return []

    # 2. 載入清單按鈕
    if st.button("📋 載入模型清單 (不耗額度)", type="primary"):
        with st.spinner("正在讀取 Google 菜單..."):
            models = fetch_model_list(USER_API_KEY)
            if models:
                st.session_state.model_list = models
                st.session_state.list_loaded = True
                st.success(f"讀取成功！共有 {len(models)} 個選擇。")
            else:
                st.error("無法讀取清單，請檢查網路或金鑰。")
    
    st.divider()
    
    # 3. 讓使用者選擇
    selected_model = None
    if st.session_state.list_loaded:
        st.subheader("👇 請選擇一個模型：")
        
        # 預設選 flash (通常最穩)
        default_idx = 0
        for i, m in enumerate(st.session_state.model_list):
            if "flash" in m and "1.5" in m:
                default_idx = i
                break
                
        selected_model = st.radio(
            "點擊選擇後，系統會自動測試該模型：",
            st.session_state.model_list,
            index=default_idx
        )
        
        # 4. 單點測試 (只測這一個！)
        st.markdown("---")
        st.caption(f"正在測試連線：`{selected_model}` ...")
        
        # 實測連線
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{selected_model}:generateContent?key={USER_API_KEY}"
        headers = {'Content-Type': 'application/json'}
        data = {"contents": [{"parts": [{"text": "Hi"}]}]}
        
        try:
            # 設定 3 秒超時，避免卡太久
            resp = requests.post(url, headers=headers, json=data, timeout=3)
            
            if resp.status_code == 200:
                st.success("🟢 此模型連線正常！請至右側使用。")
                active_status = True
            elif resp.status_code == 429:
                st.error("🔴 此模型額度已滿 (429)，請換一個選。")
                active_status = False
            else:
                st.error(f"❌ 連線失敗 ({resp.status_code})")
                active_status = False
        except Exception as e:
            st.error("❌ 網路連線錯誤")
            active_status = False

    else:
        st.markdown("等待載入清單...")
        active_status = False

# ==========================================
# 👉 右側主畫面
# ==========================================
st.title("📄 文獻分析工作區")

if not active_status:
    if st.session_state.list_loaded:
        st.warning("⚠️ 左側選中的模型目前無法使用，請試試看清單中的其他選項。")
    else:
        st.info("⬅️ 請先在左側點擊 **「📋 載入模型清單」**。")
else:
    # 只有綠燈才會顯示這裡
    st.success(f"🚀 已鎖定核心：**{selected_model}**")
    
    raw_text = st.text_area("請在此貼上文獻資料 (每篇請換行)：", height=300)

    # 分析函數
    def run_analysis_final(text, model_name):
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

    def parse_text(text):
        lines = text.strip().split('\n')
        return [{"title": line[:15], "content": line} for line in lines if len(line) > 5]

    if st.button("🚀 開始分析", type="primary"):
        if not raw_text:
            st.warning("請先輸入資料！")
        else:
            keywords = []
            with st.spinner(f"正在使用 {selected_model} 分析..."):
                res = run_analysis_final(raw_text, selected_model)
                
                if res:
                    keywords = [k.strip() for k in res.replace("\n", "、").split("、") if k.strip()]
                    st.success("✅ 分析成功")
                else:
                    st.error("❌ 分析中途斷線，可能剛好額度滿了，請稍後再試。")
                    # 備用方案提示
                    st.info("💡 如果一直失敗，可能是今日額度用盡，請明天再來。")

            if keywords:
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
