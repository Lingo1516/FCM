import streamlit as st
import pandas as pd
import requests
import string
import re
import time # 這是關鍵，用來控制速度
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
st.set_page_config(page_title="AI 智慧篩選分析", layout="wide", page_icon="🕵️")

if 'verified_models' not in st.session_state:
    st.session_state.verified_models = []
if 'filter_done' not in st.session_state:
    st.session_state.filter_done = False

# ==========================================
# 🛑 左側邊欄：智慧篩選站
# ==========================================
with st.sidebar:
    st.header("🕵️ 第一步：智慧篩選")
    st.info("點擊下方按鈕，系統會「慢速」逐一測試，只幫您留下真正能用的模型。")
    
    # 測試單一模型函數
    def test_single_model(key, model_name):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={key}"
        headers = {'Content-Type': 'application/json'}
        data = {"contents": [{"parts": [{"text": "Hi"}]}]}
        try:
            # 設定超時
            response = requests.post(url, headers=headers, json=data, timeout=5)
            if response.status_code == 200:
                return True
            else:
                return False
        except:
            return False

    # 執行篩選按鈕
    if st.button("🔍 開始自動篩選 (約需 10 秒)", type="primary"):
        st.session_state.verified_models = []
        
        # 我們只挑這幾個「精英模型」來測，不要測垃圾模型浪費時間
        candidates = [
            "gemini-1.5-flash",       # 最快、最穩
            "gemini-1.5-pro",         # 最聰明
            "gemini-2.0-flash",       # 最新版
            "gemini-2.0-flash-lite-preview-02-05", # 預覽輕量版(通常沒人用，額度多)
            "gemini-1.0-pro",         # 經典舊版
            "gemini-1.5-flash-8b"     # 極速版
        ]
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        found_count = 0
        
        for i, model in enumerate(candidates):
            status_text.markdown(f"正在測試：`{model}` ...")
            
            # 1. 測試
            is_alive = test_single_model(USER_API_KEY, model)
            
            # 2. 判定
            if is_alive:
                st.session_state.verified_models.append(model)
                found_count += 1
                st.toast(f"✅ {model} 可用！")
            
            # 3. 更新進度
            progress_bar.progress((i + 1) / len(candidates))
            
            # 4. 【關鍵】暫停 1.5 秒，避免被 Google 鎖 IP
            time.sleep(1.5)
            
        st.session_state.filter_done = True
        status_text.text("篩選完成！")
        
        if found_count == 0:
            st.error("❌ 全部忙線中，請稍後再試或用本機模式。")

    st.divider()
    
    # 顯示「乾淨」的選單
    final_selection = None
    
    if st.session_state.filter_done:
        if st.session_state.verified_models:
            st.success(f"🎉 成功找到 {len(st.session_state.verified_models)} 個可用模型！")
            st.caption("以下列表保證剛剛測試是綠燈的：")
            final_selection = st.radio(
                "請選擇一個開始分析：",
                st.session_state.verified_models
            )
        else:
            st.warning("⚠️ 為了不讓你空手而歸，已自動切換至「本機備用模式」。")
            final_selection = "Local (本機備用)"
    else:
        st.markdown("等待篩選中...")

# ==========================================
# 👉 右側主畫面
# ==========================================
st.title("📄 文獻分析工作區")

if not st.session_state.filter_done:
    st.info("⬅️ 請先在左側點擊 **「🔍 開始自動篩選」**。")
    st.markdown("""
    **這個版本會自動幫您：**
    1. 測試目前最熱門的 6 個模型。
    2. 自動過濾掉壞掉的、額度滿的。
    3. **只列出能用的給您選**。
    """)
else:
    # 只有篩選過才會顯示這裡
    st.success(f"🚀 已鎖定核心：**{final_selection}**")
    
    raw_text = st.text_area("請在此貼上文獻資料 (每篇請換行)：", height=300)

    # 分析函數
    def run_analysis_final(text, model_name):
        # 本機模式
        if "Local" in model_name:
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
            with st.spinner(f"正在使用 {final_selection} 分析..."):
                res = run_analysis_final(raw_text, final_selection)
                
                if isinstance(res, str): # Google 回傳字串
                    keywords = [k.strip() for k in res.replace("\n", "、").split("、") if k.strip()]
                    st.success("✅ 分析成功")
                elif isinstance(res, list): # 本機回傳 List
                    keywords = res
                    st.success("✅ 本機運算成功")
                else:
                    st.error("❌ 分析失敗，該模型可能剛好額度用盡，請左側換一個試試。")

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
