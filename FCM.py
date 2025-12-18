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

# --- 1. 設定 API Key ---
# ⚠️ 請確認這把鑰匙是你最新的、沒被刪除的
USER_API_KEY = "AIzaSyBlj24gBVr3RJhkukS9p6yo5s2-WVBH2H0" 

# --- 2. 頁面設定 ---
st.set_page_config(page_title="AI 文獻分析 (直連狙擊版)", layout="wide", page_icon="🎯")

# ==========================================
# 🛑 左側邊欄：手動狙擊站
# ==========================================
with st.sidebar:
    st.header("🎯 模型選擇")
    st.info("不再自動掃描，請直接選擇一個模型進行連線。")
    
    # 顯示目前金鑰後四碼，讓你確認有沒有用錯
    if len(USER_API_KEY) > 4:
        st.caption(f"目前使用的金鑰結尾：...{USER_API_KEY[-4:]}")
    
    # 1. 硬編碼的精英名單 (保證存在)
    target_models = [
        "gemini-1.5-flash",  # 首選：最快、免費額度最高
        "gemini-1.5-pro",    # 次選：聰明但慢
        "gemini-1.0-pro"     # 備選：舊版穩定
    ]
    
    selected_model = st.radio("請選擇核心：", target_models)
    
    # 2. 測試按鈕 (只測這一個！)
    if st.button("📡 測試連線", type="primary"):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{selected_model}:generateContent?key={USER_API_KEY}"
        headers = {'Content-Type': 'application/json'}
        data = {"contents": [{"parts": [{"text": "Hi"}]}]}
        
        with st.spinner(f"正在呼叫 {selected_model}..."):
            try:
                # 這裡不設太短的 timeout，給它一點時間
                response = requests.post(url, headers=headers, json=data)
                
                if response.status_code == 200:
                    st.success(f"✅ 成功！{selected_model} 運作正常！")
                    st.session_state.model_ready = True
                    st.session_state.active_model = selected_model
                elif response.status_code == 429:
                    st.error("❌ 額度滿了 (429)。請休息 2 分鐘再試，或換一個模型。")
                elif response.status_code == 400:
                    st.error("❌ 金鑰無效 (400)。請檢查 API Key 是否正確。")
                else:
                    st.error(f"❌ 連線失敗 (代碼 {response.status_code}): {response.text}")
            except Exception as e:
                st.error(f"❌ 網路錯誤：{e}")

# ==========================================
# 👉 右側主畫面
# ==========================================
st.title("📄 文獻分析工作區")

if 'model_ready' not in st.session_state:
    st.session_state.model_ready = False

if not st.session_state.model_ready:
    st.info("⬅️ 請先在左側選擇模型並點擊 **「📡 測試連線」**。")
else:
    st.success(f"🚀 當前鎖定核心：**{st.session_state.active_model}**")
    
    raw_text = st.text_area("請在此貼上文獻資料 (每篇請換行)：", height=300)

    # --- 分析函數 (使用 requests 直連) ---
    def run_analysis_direct(text, model_name):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={USER_API_KEY}"
        headers = {'Content-Type': 'application/json'}
        
        # Prompt 優化
        prompt = f"""
        你是一個學術分析助手。請閱讀以下文獻內容，歸納出 10 個最重要的「研究構面」或「評估準則」關鍵字。
        【規則】：
        1. 只輸出名詞。
        2. 用頓號「、」隔開。
        3. 不要包含：日期、時間、作者名、報告、下午、研究方法。
        
        【內容】：
        {text[:8000]}
        """
        
        data = {"contents": [{"parts": [{"text": prompt}]}]}
        
        try:
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                return response.json()['candidates'][0]['content']['parts'][0]['text']
            else:
                return f"Error: {response.text}"
        except Exception as e:
            return f"Error: {str(e)}"

    def parse_text(text):
        lines = text.strip().split('\n')
        return [{"title": line[:15], "content": line} for line in lines if len(line) > 5]

    if st.button("🚀 開始分析", type="primary"):
        if not raw_text:
            st.warning("請先輸入資料！")
        else:
            with st.spinner(f"正在使用 {st.session_state.active_model} 進行分析..."):
                ai_result = run_analysis_direct(raw_text, st.session_state.active_model)
                
                if "Error" in ai_result:
                    st.error(f"分析失敗：{ai_result}")
                    if "429" in ai_result:
                        st.warning("您的 API 額度暫時滿了，請稍後再試。")
                else:
                    st.success("✅ 分析完成！")
                    
                    # 處理關鍵字
                    keywords = [k.strip() for k in ai_result.replace("\n", "、").split("、") if k.strip()]
                    
                    if keywords:
                        final_keywords = st.multiselect("分析準則", options=keywords, default=keywords)
                        
                        if final_keywords:
                            # 製作表格
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
                            
                            # 下載
                            output = BytesIO()
                            try:
                                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                    df.to_excel(writer, sheet_name='矩陣')
                                    df_legend.to_excel(writer, sheet_name='對照表')
                                st.download_button("📥 下載 Excel", output.getvalue(), "analysis.xlsx")
                            except:
                                st.download_button("📥 下載 CSV", df.to_csv().encode('utf-8-sig'), "analysis.csv")
