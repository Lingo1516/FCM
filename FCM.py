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
st.set_page_config(page_title="AI 模型深度健檢", layout="wide", page_icon="🩺")

if 'working_models' not in st.session_state:
    st.session_state.working_models = []
if 'scan_performed' not in st.session_state:
    st.session_state.scan_performed = False

# ==========================================
# 🛑 左側邊欄：深度健檢站
# ==========================================
with st.sidebar:
    st.header("🩺 第一步：模型健檢")
    st.info("這個按鈕會實際測試每個模型，過濾掉「額度已滿」的壞模型。")
    
    # 測試函數
    def check_model_health(key, model_name):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={key}"
        headers = {'Content-Type': 'application/json'}
        # 傳送一個極短的字符來測試
        data = {"contents": [{"parts": [{"text": "Hi"}]}]}
        try:
            response = requests.post(url, headers=headers, json=data, timeout=5)
            if response.status_code == 200:
                return True # 活著
            else:
                return False # 死掉 (429 或其他)
        except:
            return False

    # 深度掃描按鈕
    if st.button("🚀 執行深度掃描 (只留活口)", type="primary"):
        st.session_state.working_models = [] # 清空舊紀錄
        
        # 我們只測試這幾個最常用且可能有額度的 (避免測試太多導致自己被鎖)
        target_candidates = [
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-2.0-flash",     # 新版
            "gemini-2.0-flash-lite-preview-02-05", # 輕量版(通常比較空)
            "gemini-1.0-pro"        # 舊版(備用)
        ]
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        found_any = False
        
        for i, model in enumerate(target_candidates):
            status_text.text(f"正在測試：{model} ...")
            
            # 實際打一次 API
            is_healthy = check_model_health(USER_API_KEY, model)
            
            if is_healthy:
                st.session_state.working_models.append(model)
                st.toast(f"✅ {model} 測試通過！")
                found_any = True
            else:
                # 失敗就不加入清單
                print(f"{model} 測試失敗")
            
            # 更新進度條
            progress_bar.progress((i + 1) / len(target_candidates))
            time.sleep(0.5) # 稍微停頓一下，避免被判定攻擊
            
        st.session_state.scan_performed = True
        status_text.text("掃描完成！")
        
        if not found_any:
            st.error("❌ 所有 Google 模型都忙線中 (429)。建議使用本機模式。")

    st.divider()
    
    # 顯示「經過篩選」的選單
    final_selection = None
    
    if st.session_state.scan_performed:
        if st.session_state.working_models:
            st.success(f"✅ 找到 {len(st.session_state.working_models)} 個可用模型！")
            final_selection = st.radio(
                "請選擇一個 (這些都是確定能用的)：",
                st.session_state.working_models
            )
        else:
            st.warning("⚠️ Google 全線崩潰，已自動切換至「本機備用模式」。")
            final_selection = "Local (本機備用)"
    else:
        st.markdown("等待掃描中...")

# ==========================================
# 👉 右側主畫面
# ==========================================
st.title("📄 文獻分析工作區 (健檢版)")

if not st.session_state.scan_performed:
    st.info("⬅️ 請先在左側點擊 **「🚀 執行深度掃描」**。")
    st.markdown("""
    **為什麼要這麼做？**
    先前的掃描只是列出名字，沒有檢查額度。
    這次我們會真的去「敲門」，確認對方有空才讓你選，避免你白忙一場。
    """)
else:
    # 顯示輸入框
    st.success(f"🚀 當前使用核心：**{final_selection}**")
    
    raw_text = st.text_area("請在此貼上文獻資料 (每篇請換行)：", height=300)

    # 分析函數
    def run_analysis_smart(text, model_name):
        if model_name == "Local (本機備用)":
            try:
                return jieba.analyse.extract_tags(text, topK=15, allowPOS=('n', 'vn', 'v'))
            except:
                clean = re.sub(r'[^\u4e00-\u9fa5]', '', text)
                words = [clean[i:i+2] for i in range(len(clean)-1)]
                return [w for w, c in Counter(words).most_common(15)]
        
        # Google 模式
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
            with st.spinner(f"正在分析..."):
                res = run_analysis_smart(raw_text, final_selection)
                
                if isinstance(res, str):
                    keywords = [k.strip() for k in res.replace("\n", "、").split("、") if k.strip()]
                    st.success("✅ 分析成功")
                elif isinstance(res, list):
                    keywords = res
                    st.success("✅ 本機運算成功")
                else:
                    st.error("❌ 哎呀，剛測過能用，結果現在又滿了。請重試一次或切換模型。")

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
