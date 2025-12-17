import streamlit as st
import pandas as pd
import requests # 改用這個最基礎的套件
import json
import string
from io import BytesIO

# --- 嘗試匯入 xlsxwriter (防呆) ---
try:
    import xlsxwriter
except ImportError:
    pass # 沒裝就算了，後面有防呆

# --- 1. 設定您的 API Key ---
# ⚠️ 請在下方引號內貼上你的 AIza 開頭金鑰
USER_API_KEY = "AIzaSyBlj24gBVr3RJhkukS9p6yo5s2-WVBH2H0" 

# --- 2. 頁面設定 ---
st.set_page_config(page_title="AI 文獻分析器 (API直連版)", layout="wide", page_icon="⚡")
st.title("⚡ AI 文獻分析器 (直連版)")
st.markdown("### 使用 API 直連模式，繞過套件版本問題")

# --- 3. 測試連線按鈕 ---
if st.button("📡 測試 API 連線"):
    if "AIza" not in USER_API_KEY:
        st.error("❌ 金鑰格式錯誤！")
    else:
        with st.spinner("正在直連 Google 主機..."):
            try:
                # 直接呼叫網址，不透過套件
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={USER_API_KEY}"
                headers = {'Content-Type': 'application/json'}
                data = {"contents": [{"parts": [{"text": "Hello"}]}]}
                
                response = requests.post(url, headers=headers, json=data)
                
                if response.status_code == 200:
                    st.success(f"✅ 連線成功！Google 回應：{response.json()['candidates'][0]['content']['parts'][0]['text']}")
                else:
                    st.error(f"❌ 連線失敗 (代碼 {response.status_code}): {response.text}")
            except Exception as e:
                st.error(f"❌ 網路錯誤：{str(e)}")

# --- 4. 文獻輸入與處理 ---
st.info("👇 請貼上文獻資料 (每篇請換行)")
raw_text = st.text_area("文獻輸入區", height=200)

def get_ai_analysis_via_api(text, key):
    # 使用 REST API 直接呼叫，不需要 google-generativeai 套件
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={key}"
    headers = {'Content-Type': 'application/json'}
    
    prompt = f"""
    任務：歸納 10 個學術研究構面關鍵字。
    規則：只列出名詞，用頓號隔開。排除無關詞彙(如日期、下午)。
    內容：{text[:8000]}
    """
    
    data = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            result = response.json()
            # 解析複雜的 JSON 結構
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            return f"Error: {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"

# 切割文字邏輯
def parse_text(text):
    lines = text.strip().split('\n')
    return [{"title": line[:15], "content": line} for line in lines if len(line) > 5]

# --- 5. 執行分析 ---
if st.button("🚀 開始分析", type="primary"):
    if not raw_text:
        st.warning("請先貼上資料！")
    else:
        with st.spinner("🤖 AI (直連模式) 分析中..."):
            lit_data = parse_text(raw_text)
            ai_result = get_ai_analysis_via_api(raw_text, USER_API_KEY)
            
            if "Error" in ai_result:
                st.error(f"分析失敗：{ai_result}")
            else:
                st.success("✅ 分析完成！")
                
                # 處理關鍵字
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
                        col_res = []
                        for kw in final_keywords:
                            if kw in item['content']: col_res.append("○")
                            else: col_res.append("")
                        matrix[lbl] = col_res
                    
                    # 顯示
                    df = pd.DataFrame(matrix, index=final_keywords)
                    df_legend = pd.DataFrame({"代號": labels, "對應文獻": titles})
                    
                    c1, c2 = st.columns([2, 1])
                    with c1: st.dataframe(df, use_container_width=True)
                    with c2: st.dataframe(df_legend, hide_index=True)
                    
                    # 下載 Excel
                    output = BytesIO()
                    try:
                        import xlsxwriter
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            df.to_excel(writer, sheet_name='矩陣')
                            df_legend.to_excel(writer, sheet_name='對照表')
                        st.download_button("📥 下載 Excel", output.getvalue(), "analysis.xlsx")
                    except ImportError:
                        # 萬一連 xlsxwriter 都沒裝成功，至少給 CSV
                        st.download_button("📥 下載 CSV (Excel無法用)", df.to_csv().encode('utf-8-sig'), "analysis.csv")
