import streamlit as st
import pandas as pd
import requests
import json
import string
from io import BytesIO

# --- 1. 基礎設定 ---
st.set_page_config(page_title="AI 文獻分析 (APA 完美版)", layout="wide", page_icon="🎓")

# --- 2. 側邊欄 ---
with st.sidebar:
    st.header("🔑 設定")
    st.info("此版本會自動將您的文獻轉為 APA 格式，不再會因為換行而切碎。")
    api_key = st.text_input("Google API Key", type="password")

# --- 3. 自動尋找可用模型 ---
def get_best_model(key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            models = response.json().get('models', [])
            for m in models:
                if 'gemini-1.5-flash' in m['name']: return m['name']
            for m in models:
                if 'gemini-1.5-pro' in m['name']: return m['name']
            for m in models:
                if 'gemini' in m['name'] and 'generateContent' in m.get('supportedGenerationMethods', []):
                    return m['name']
        return "models/gemini-1.5-flash" # 預設 fallback
    except:
        return "models/gemini-1.5-flash"

# --- 4. 核心分析邏輯 (改用 JSON 強制結構化) ---
def run_smart_analysis(text, key, model_name):
    url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent?key={key}"
    headers = {'Content-Type': 'application/json'}
    
    # 這是最強的 Prompt：要求 AI 直接回傳整理好的 JSON
    prompt = f"""
    你是一個學術分析專家。請閱讀以下雜亂的文獻原始資料（可能包含一篇或多篇）。
    請幫我做兩件事：
    1. 辨識出有幾篇不同的文獻，將每一篇整理成「APA 引用格式 (作者, 年份, 標題)」。
    2. 針對每一篇文獻，分析出 5-10 個「研究構面」關鍵字(名詞)。
    
    請務必回傳純 JSON 格式，不要有 markdown 標記，格式如下：
    [
      {{
        "apa": "陳小明 (2024). 餐飲業的服務創新研究...",
        "keywords": ["服務創新", "滿意度", "忠誠度"]
      }},
      {{
        "apa": "王大華 (2023). ...",
        "keywords": ["商業模式", "SWOT", ...]
      }}
    ]

    原始資料：
    {text[:10000]}
    """
    
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            res_text = response.json()['candidates'][0]['content']['parts'][0]['text']
            # 清理可能的回傳雜訊 (有些模型會加 ```json )
            clean_json = res_text.replace("```json", "").replace("```", "").strip()
            return "OK", json.loads(clean_json)
        else:
            return "ERROR", f"錯誤 ({response.status_code}): {response.text}"
    except Exception as e:
        return "ERROR", str(e)

# --- 5. 主畫面與執行 ---
st.title("📄 文獻分析工作區 (APA 自動整理)")

raw_text = st.text_area("請在此貼上文獻資料 (亂一點沒關係，AI 會自己整理)：", height=300)

if st.button("🚀 開始分析", type="primary"):
    if not api_key:
        st.error("❌ 請先貼上 API Key！")
    elif not raw_text:
        st.warning("⚠️ 請先輸入文獻資料！")
    else:
        with st.spinner("🔍 正在自動選擇模型並整理文獻..."):
            valid_model = get_best_model(api_key)
            
            # 開始 AI 分析
            status, result_data = run_smart_analysis(raw_text, api_key, valid_model)
            
            if status == "OK":
                st.success("✅ 分析完成！")
                
                # result_data 是一個 List [ {"apa":..., "keywords":...}, ... ]
                
                # 1. 收集所有出現過的關鍵字 (取聯集)
                all_keywords = set()
                for paper in result_data:
                    for k in paper.get("keywords", []):
                        all_keywords.add(k)
                
                sorted_keywords = sorted(list(all_keywords))
                
                # 2. 讓使用者篩選關鍵字
                st.subheader("1️⃣ 篩選分析構面")
                final_keywords = st.multiselect("請勾選要保留的構面：", options=sorted_keywords, default=sorted_keywords)
                
                if final_keywords:
                    # 3. 建立矩陣
                    matrix = {}
                    labels = []
                    apa_list = []
                    
                    for i, paper in enumerate(result_data):
                        lbl = string.ascii_uppercase[i % 26]
                        labels.append(lbl)
                        # 這裡就是你要的：右邊欄位直接顯示 APA 格式
                        apa_list.append(paper["apa"]) 
                        
                        # 檢查該篇論文的關鍵字清單
                        paper_keywords = paper.get("keywords", [])
                        matrix[lbl] = ["●" if k in paper_keywords else "" for k in final_keywords]
                    
                    # 4. 顯示結果
                    df_matrix = pd.DataFrame(matrix, index=final_keywords)
                    df_legend = pd.DataFrame({"代號": labels, "APA 文獻來源": apa_list})
                    
                    st.divider()
                    c1, c2 = st.columns([1.5, 2]) # 調整比例，讓右邊寬一點顯示 APA
                    with c1: 
                        st.subheader("📊 分析矩陣")
                        st.dataframe(df_matrix, use_container_width=True)
                    with c2: 
                        st.subheader("📝 文獻對照表 (APA)")
                        st.dataframe(df_legend, hide_index=True, use_container_width=True)
                    
                    # 5. 下載
                    output = BytesIO()
                    try:
                        import xlsxwriter
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            df_matrix.to_excel(writer, sheet_name='矩陣')
                            df_legend.to_excel(writer, sheet_name='APA來源表')
                        st.download_button("📥 下載 Excel", output.getvalue(), "analysis.xlsx")
                    except:
                        st.download_button("📥 下載 CSV", df_matrix.to_csv().encode('utf-8-sig'), "analysis.csv")
            else:
                st.error("分析失敗，請檢查 API Key 或重試。")
                st.code(result_data)
