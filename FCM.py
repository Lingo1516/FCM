import streamlit as st
import pandas as pd
import google.generativeai as genai
import string
from io import BytesIO

# --- 1. 設定您的 API Key ---
# ⚠️ 請把你的新鑰匙貼在下面這個引號裡面 (不要留空白！)
USER_API_KEY = "AIzaSyBlj24gBVr3RJhkukS9p6yo5s2-WVBH2H0" 

# 設定 Google Gemini
if USER_API_KEY and "AIza" in USER_API_KEY:
    genai.configure(api_key=USER_API_KEY)

# --- 2. 頁面設定 ---
st.set_page_config(page_title="AI 文獻分析器 (連線測試版)", layout="wide", page_icon="⚡")
st.title("⚡ AI 文獻分析器 (含連線檢測)")

# --- 3. 測試連線區 (新增功能) ---
st.info("👇 如果擔心卡住，請先點擊下方的「測試連線」按鈕")
if st.button("📡 測試 AI 連線 (Ping)"):
    if "AIza" not in USER_API_KEY:
        st.error("❌ 程式碼第 9 行還沒有貼上正確的金鑰喔！")
    else:
        with st.spinner("正在嘗試呼叫 Google..."):
            try:
                # 測試用最簡單的指令
                model = genai.GenerativeModel('gemini-pro')
                response = model.generate_content("Hello, reply 'OK' if you see this.")
                st.success(f"✅ 連線成功！Google 回應：{response.text}")
                st.balloons() # 放氣球慶祝
            except Exception as e:
                st.error(f"❌ 連線失敗！原因：{e}")
                st.warning("請檢查：1. 金鑰是否正確？ 2. 舊金鑰是否已刪除？")

st.divider()

# --- 4. 原本文獻輸入區 ---
st.markdown("### 文獻分析區")
raw_text = st.text_area("文獻輸入區", height=200, placeholder="貼上文獻內容...\n記得換行...")

# --- 5. 分析核心邏輯 ---
def get_ai_analysis(text):
    model = genai.GenerativeModel('gemini-pro')
    # 增加 timeout 設定 (避免無限轉圈)
    # 雖然 python library 不一定完全支援 timeout 參數，但我們透過 prompt 簡化來加速
    prompt = f"""
    任務：歸納 10 個學術研究構面關鍵字。
    規則：只列出名詞，用頓號隔開。排除無關詞彙。
    內容：{text[:5000]}
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# 輔助：切割
def parse_text(text):
    lines = text.strip().split('\n')
    return [{"title": line[:15], "content": line} for line in lines if len(line) > 5]

# --- 6. 執行按鈕 ---
if st.button("🚀 開始正式分析", type="primary"):
    if not raw_text:
        st.warning("請先貼上資料！")
    else:
        # 顯示進度條，讓你心安
        progress_text = "AI 正在閱讀中，請稍候... (約需 5-10 秒)"
        my_bar = st.progress(0, text=progress_text)
        
        try:
            # 模擬進度 (因為 API 是同步的，無法精準顯示 %，只能給個感覺)
            my_bar.progress(30, text="正在傳送資料給 Google...")
            
            # A. 切割
            lit_data = parse_text(raw_text)
            
            # B. 呼叫
            ai_result = get_ai_analysis(raw_text)
            my_bar.progress(80, text="正在整理數據...")
            
            if "Error" in ai_result:
                st.error(f"發生錯誤：{ai_result}")
            else:
                st.success("✅ 分析完成！")
                my_bar.progress(100, text="完成！")
                
                # C. 後續處理 (簡化版顯示)
                keywords = [k.strip() for k in ai_result.replace("\n", "、").split("、") if k.strip()]
                final_keywords = st.multiselect("AI 抓到的準則", options=keywords, default=keywords)
                
                if final_keywords:
                    # 建表
                    matrix = {}
                    labels = []
                    titles = []
                    for i, item in enumerate(lit_data):
                        lbl = string.ascii_uppercase[i % 26]
                        labels.append(lbl)
                        titles.append(item['title'])
                        matrix[lbl] = ["○" if k in item['content'] else "" for k in final_keywords]
                    
                    df = pd.DataFrame(matrix, index=final_keywords)
                    st.dataframe(df)
                    
        except Exception as e:
            st.error(f"系統錯誤：{e}")
