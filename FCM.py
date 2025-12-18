import streamlit as st
import pandas as pd
import requests
import string
from io import BytesIO

# --- 1. 基礎設定 ---
st.set_page_config(page_title="學術文獻分析器 (自動適配版)", layout="wide", page_icon="🎓")

# --- 2. 側邊欄 ---
with st.sidebar:
    st.header("🔑 設定")
    st.info("此版本會自動偵測您的金鑰可用的模型，無需手動選擇。")
    
    # 這裡請貼上你那把 OK 的鑰匙
    api_key = st.text_input("Google API Key", type="password")

# --- 3. 核心邏輯：自動尋找可用模型 ---
def get_best_model(key):
    # 問 Google 這把鑰匙能用誰
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            models = response.json().get('models', [])
            # 優先找 gemini-1.5-flash (最快)，沒有就找其他的
            for m in models:
                if 'gemini-1.5-flash' in m['name']: return m['name']
            for m in models:
                if 'gemini-1.5-pro' in m['name']: return m['name']
            for m in models:
                if 'gemini' in m['name'] and 'generateContent' in m.get('supportedGenerationMethods', []):
                    return m['name']
            return None
        else:
            return None
    except:
        return None

# --- 4. 主畫面 ---
st.title("📄 學術文獻分析工作區")

raw_text = st.text_area("請在此貼上文獻資料 (每篇請換行)：", height=300)

# --- 5. 分析函數 ---
def run_analysis(text, key, model_name):
    # 這裡的 model_name 已經是自動偵測到的正確名稱
    url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent?key={key}"
    headers = {'Content-Type': 'application/json'}
    
    prompt = f"""
    任務：歸納 10 到 15 個最重要的「研究構面」或「評估準則」。
    規則：只輸出名詞，用頓號「、」隔開。排除無關詞彙。
    內容：{text[:8000]}
    """
    
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            return "OK", response.json()['candidates'][0]['content']['parts'][0]['text']
        else:
            return "ERROR", f"錯誤 ({response.status_code}): {response.text}"
    except Exception as e:
        return "ERROR", str(e)

def parse_text(text):
    lines = text.strip().split('\n')
    return [{"title": line[:15], "content": line} for line in lines if len(line) > 5]

# --- 6. 執行按鈕 ---
if st.button("🚀 開始分析", type="primary"):
    if not api_key:
        st.error("❌ 請先貼上 API Key！")
    elif not raw_text:
        st.warning("⚠️ 請先輸入文獻資料！")
    else:
        with st.spinner("🔍 正在自動尋找可用模型..."):
            # A. 先自動找模型
            valid_model = get_best_model(api_key)
            
            if not valid_model:
                st.error("❌ 無法找到任何可用模型。請確認您的 Key 是否正確，或專案是否已啟用 API。")
            else:
                st.success(f"✅ 連線成功！自動選用模型：`{valid_model}`")
                
                # B. 開始分析
                with st.spinner("🤖 AI 正在分析中..."):
                    status, result = run_analysis(raw_text, api_key, valid_model)
                    
                    if status == "OK":
                        st.success("✅ 分析完成！")
                        keywords = [k.strip() for k in result.replace("\n", "、").split("、") if k.strip()]
                        
                        st.subheader("1️⃣ AI 建議構面")
                        final_keywords = st.multiselect("請勾選：", options=keywords, default=keywords)
                        
                        if final_keywords:
                            lit_data = parse_text(raw_text)
                            matrix = {}
                            labels = []
                            titles = []
                            
                            for i, item in enumerate(lit_data):
                                lbl = string.ascii_uppercase[i % 26]
                                labels.append(lbl)
                                titles.append(item['title'])
                                matrix[lbl] = ["●" if k in item['content'] else "" for k in final_keywords]
                            
                            df = pd.DataFrame(matrix, index=final_keywords)
                            df_legend = pd.DataFrame({"代號": labels, "標題": titles})
                            
                            st.divider()
                            c1, c2 = st.columns([2, 1])
                            with c1: st.dataframe(df, use_container_width=True)
                            with c2: st.dataframe(df_legend, hide_index=True)
                            
                            # 下載
                            output = BytesIO()
                            try:
                                import xlsxwriter
                                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                    df.to_excel(writer, sheet_name='矩陣')
                                    df_legend.to_excel(writer, sheet_name='對照表')
                                st.download_button("📥 下載 Excel", output.getvalue(), "analysis.xlsx")
                            except:
                                st.download_button("📥 下載 CSV", df.to_csv().encode('utf-8-sig'), "analysis.csv")
                    else:
                        st.error(result)
