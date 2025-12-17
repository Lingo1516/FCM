import streamlit as st
import pandas as pd
import requests
import json
import string
from io import BytesIO

# --- 嘗試匯入 xlsxwriter (防呆) ---
try:
    import xlsxwriter
except ImportError:
    pass 

# --- 1. 設定您的 API Key ---
USER_API_KEY = "AIzaSyBlj24gBVr3RJhkukS9p6yo5s2-WVBH2H0" 

# --- 2. 頁面設定 ---
st.set_page_config(page_title="AI 文獻分析器 (自動偵測版)", layout="wide", page_icon="🛡️")
st.title("🛡️ AI 文獻分析器 (自動偵測模型版)")
st.markdown("### 系統將自動尋找您的金鑰可用的模型，解決 404 問題")

# --- 3. 核心：自動偵測可用模型 ---
def find_working_model(api_key):
    # 問 Google: 我能用什麼模型？
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            # 找一個支援 'generateContent' 的模型
            for model in data.get('models', []):
                if 'generateContent' in model.get('supportedGenerationMethods', []):
                    # 優先找 gemini 系列
                    if 'gemini' in model['name']:
                        return model['name'] # 找到就回傳，例如 'models/gemini-1.5-flash'
            return None
        else:
            return None
    except:
        return None

# --- 4. 測試連線按鈕 ---
if st.button("📡 測試連線與自動偵測"):
    with st.spinner("正在詢問 Google 您可用的模型列表..."):
        valid_model = find_working_model(USER_API_KEY)
        
        if valid_model:
            st.success(f"✅ 連線成功！系統自動為您選用了模型：`{valid_model}`")
            # 測試一下
            url = f"https://generativelanguage.googleapis.com/v1beta/{valid_model}:generateContent?key={USER_API_KEY}"
            headers = {'Content-Type': 'application/json'}
            data = {"contents": [{"parts": [{"text": "Hello"}]}]}
            try:
                test_resp = requests.post(url, headers=headers, json=data)
                if test_resp.status_code == 200:
                    st.info(f"回應測試：{test_resp.json()['candidates'][0]['content']['parts'][0]['text']}")
                else:
                    st.error(f"雖然找到了模型，但測試失敗：{test_resp.text}")
            except Exception as e:
                st.error(f"測試請求錯誤：{e}")
        else:
            st.error("❌ 無法找到任何可用模型！可能是您的 API Key 沒有權限，或該專案未啟用 Generative AI API。")

# --- 5. 文獻輸入與處理 ---
st.info("👇 請貼上文獻資料 (每篇請換行)")
raw_text = st.text_area("文獻輸入區", height=200)

def get_ai_analysis_auto(text, key):
    # 1. 先找模型
    model_name = find_working_model(key)
    if not model_name:
        return "Error: 無法偵測到任何可用模型，請檢查 API Key 權限。"
    
    # 2. 用找到的模型去跑
    url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent?key={key}"
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
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            return f"Error (Code {response.status_code}): {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"

# 切割文字邏輯
def parse_text(text):
    lines = text.strip().split('\n')
    return [{"title": line[:15], "content": line} for line in lines if len(line) > 5]

# --- 6. 執行分析 ---
if st.button("🚀 開始分析", type="primary"):
    if not raw_text:
        st.warning("請先貼上資料！")
    else:
        with st.spinner("🤖 AI 正在自動偵測模型並分析中..."):
            lit_data = parse_text(raw_text)
            ai_result = get_ai_analysis_auto(raw_text, USER_API_KEY)
            
            if "Error" in ai_result:
                st.error(f"分析失敗：{ai_result}")
                st.warning("如果一直失敗，建議去 Google AI Studio 重新申請一把新的 Key，舊的可能權限壞了。")
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
                        st.download_button("📥 下載 CSV", df.to_csv().encode('utf-8-sig'), "analysis.csv")
