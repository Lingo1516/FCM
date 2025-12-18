import streamlit as st
import pandas as pd
import requests
import json
import string
from io import BytesIO

# --- 1. 基礎設定 ---
st.set_page_config(page_title="MCDM 文獻準則提取器", layout="wide", page_icon="⚖️")

# --- 2. 側邊欄 ---
with st.sidebar:
    st.header("⚖️ 設定")
    st.info("此版本專為 MCDM 研究設計。AI 將自動歸納「評估準則」並整理 APA 格式。")
    api_key = st.text_input("Google API Key", type="password")

# --- 3. 自動尋找可用模型 ---
def get_best_model(key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            models = response.json().get('models', [])
            # 優先順序：Flash -> Pro -> 任何 Gemini
            for m in models:
                if 'gemini-1.5-flash' in m['name']: return m['name']
            for m in models:
                if 'gemini-1.5-pro' in m['name']: return m['name']
            for m in models:
                if 'gemini' in m['name'] and 'generateContent' in m.get('supportedGenerationMethods', []):
                    return m['name']
        return "models/gemini-1.5-flash"
    except:
        return "models/gemini-1.5-flash"

# --- 4. 核心分析邏輯 (MCDM 專用 Prompt) ---
def run_mcdm_analysis(text, key, model_name):
    url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent?key={key}"
    headers = {'Content-Type': 'application/json'}
    
    # 這是 MCDM 專用的指令
    prompt = f"""
    你是一個 MCDM（多準則決策分析）的研究專家。請閱讀以下文獻資料。
    
    【任務目標】：
    1. 辨識出文獻來源，並轉換為「APA 格式 citation」（例如：王小明 (2024). 論文題目...）。
    2. 針對每一篇文獻，歸納出作者在研究中使用的「評估準則 (Evaluation Criteria)」或「屬性 (Attributes)」。
    
    【準則提取規則】：
    - 準則必須是「名詞」或「名詞片語」（例如：建置成本、服務品質、系統穩定性）。
    - 排除「研究方法」（如 AHP、TOPSIS、SWOT、BCG矩陣），這些不是準則。
    - 排除「產業名稱」（如 不動產、餐飲業）。
    - 只列出該文獻真正探討的衡量指標。
    
    【輸出格式】：
    請直接回傳純 JSON 格式 (不要 Markdown)，結構如下：
    [
      {{
        "apa": "作者 (年份). 文獻標題...",
        "criteria": ["準則A", "準則B", "準則C"]
      }},
      ...
    ]

    【原始文獻資料】：
    {text[:12000]}
    """
    
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            res_text = response.json()['candidates'][0]['content']['parts'][0]['text']
            # 清理 JSON 格式 (有些模型會加 markdown 標記)
            clean_json = res_text.replace("```json", "").replace("```", "").strip()
            return "OK", json.loads(clean_json)
        else:
            return "ERROR", f"錯誤 ({response.status_code}): {response.text}"
    except Exception as e:
        return "ERROR", str(e)

# --- 5. 主畫面與執行 ---
st.title("⚖️ MCDM 評估準則提取工作區")

raw_text = st.text_area("請在此貼上文獻摘要 (AI 會自動分篇並提取準則)：", height=300)

if st.button("🚀 開始提取準則", type="primary"):
    if not api_key:
        st.error("❌ 請先貼上 API Key！")
    elif not raw_text:
        st.warning("⚠️ 請先輸入文獻資料！")
    else:
        with st.spinner("🔍 AI 正在進行 MCDM 準則歸納..."):
            valid_model = get_best_model(api_key)
            
            # 呼叫 AI
            status, result_data = run_mcdm_analysis(raw_text, api_key, valid_model)
            
            if status == "OK":
                st.success("✅ 提取完成！")
                
                # result_data 結構: [ {"apa":..., "criteria":[...]}, ... ]
                
                # 1. 收集所有準則 (取聯集並排序)
                all_criteria = set()
                for paper in result_data:
                    for c in paper.get("criteria", []):
                        all_criteria.add(c)
                
                sorted_criteria = sorted(list(all_criteria))
                
                # 2. 讓使用者篩選 (預設全選)
                st.subheader("1️⃣ AI 歸納出的 MCDM 準則")
                final_criteria = st.multiselect("請勾選您要納入矩陣的準則：", options=sorted_criteria, default=sorted_criteria)
                
                if final_criteria:
                    # 3. 建立矩陣
                    matrix = {}
                    labels = []
                    apa_list = []
                    
                    for i, paper in enumerate(result_data):
                        lbl = string.ascii_uppercase[i % 26] # A, B, C...
                        labels.append(lbl)
                        apa_list.append(paper["apa"]) 
                        
                        # 檢查該篇論文是否包含該準則
                        paper_criteria = paper.get("criteria", [])
                        # 使用實心圓點 ● 表示有提到
                        matrix[lbl] = ["●" if c in paper_criteria else "" for c in final_criteria]
                    
                    # 4. 轉為 DataFrame
                    df_matrix = pd.DataFrame(matrix, index=final_criteria)
                    df_legend = pd.DataFrame({"代號": labels, "文獻來源 (APA)": apa_list})
                    
                    st.divider()
                    
                    # 5. 顯示 (調整比例)
                    c1, c2 = st.columns([1.5, 2.5]) 
                    with c1: 
                        st.subheader("📊 準則檢核矩陣")
                        st.dataframe(df_matrix, use_container_width=True)
                    with c2: 
                        st.subheader("📝 文獻對照表")
                        st.dataframe(df_legend, hide_index=True, use_container_width=True)
                    
                    # 6. 下載功能
                    output = BytesIO()
                    try:
                        import xlsxwriter
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            df_matrix.to_excel(writer, sheet_name='MCDM矩陣')
                            df_legend.to_excel(writer, sheet_name='文獻來源')
                        st.download_button("📥 下載 Excel 報表", output.getvalue(), "mcdm_analysis.xlsx", type="primary")
                    except:
                        st.download_button("📥 下載 CSV", df_matrix.to_csv().encode('utf-8-sig'), "mcdm_analysis.csv")
            else:
                st.error("分析失敗，請檢查內容或 Key。")
                st.code(result_data)
