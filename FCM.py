import streamlit as st
import pandas as pd
import requests
import json
import string
import re
from io import BytesIO

# --- 1. 基礎設定 ---
st.set_page_config(page_title="MCDM 層級架構分析 (構面->準則)", layout="wide", page_icon="🏗️")

# --- 2. 側邊欄 ---
with st.sidebar:
    st.header("🏗️ 層級架構設定")
    st.info("AI 將執行：文獻 -> 原始細項 -> 收斂準則 -> 歸納構面 的完整流程。")
    
    api_key = st.text_input("Google API Key", type="password")
    st.divider()
    thesis_topic = st.text_input("論文題目：", value="餐飲業導入 AI 服務之評估準則")
    
    st.subheader("層級數量設定")
    c1, c2, c3 = st.columns(3)
    with c1:
        pool_size = st.number_input("1.原始池", value=50, help="Step 1 找出的數量")
    with c2:
        criteria_size = st.number_input("2.準則數", value=15, help="Step 2 收斂出的準則數量")
    with c3:
        dim_size = st.number_input("3.構面數", value=4, help="Step 3 歸納出的構面數量")

# --- 3. 自動適配模型 ---
def get_best_model(key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            models = response.json().get('models', [])
            for m in models:
                if 'gemini-1.5-pro' in m['name']: return m['name']
            for m in models:
                if 'gemini-1.5-flash' in m['name']: return m['name']
            for m in models:
                if 'gemini' in m['name'] and 'generateContent' in m.get('supportedGenerationMethods', []):
                    return m['name']
            return None
        return None
    except:
        return None

# --- 4. 核心分析邏輯 (包含構面歸納) ---
def run_hierarchy_analysis(text, key, model_name, topic, pool_n, crit_n, dim_n):
    url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent?key={key}"
    headers = {'Content-Type': 'application/json'}
    
    prompt = f"""
    你是一個 MCDM 研究專家。題目：{topic}。
    請建立一個完整的「構面 (Dimensions) -> 準則 (Criteria)」層級架構。

    【任務流程】：
    1. **文獻處理**：辨識文獻並編號 (ID 0, 1...)，轉為 APA。
    2. **發散 (Pooling)**：從文獻找出約 {pool_n} 個「原始細項」。
    3. **收斂 (Convergence)**：將其歸納為 {crit_n} 個「評估準則 (Criteria)」。
    4. **歸納構面 (Grouping)**：請將這 {crit_n} 個準則，依照性質歸納分類到 {dim_n} 個「評估構面 (Dimensions)」底下。
       (例如：成本、技術、效益、風險...等構面)。

    【輸出 JSON 格式 (嚴格遵守)】：
    {{
      "papers": [
        {{ "id": 0, "apa": "作者A..." }},
        ...
      ],
      "step1_raw_pool": [
        {{ "name": "原始細項1", "matched_ids": [0] }},
        ...
      ],
      "final_hierarchy": [
        {{
          "dimension_name": "構面名稱 (例如：財務構面)",
          "contained_criteria": [
             {{
               "criteria_name": "準則名稱 (例如：建置成本)",
               "source_raw_items": ["原始細項A", "原始細項B"],
               "reasoning": "合併理由...",
               "matched_paper_ids": [0, 2]
             }},
             ... (該構面底下的準則)
          ]
        }},
        ... (共 {dim_n} 個構面，所有準則加總需約 {crit_n} 個)
      ]
    }}
    
    文獻內容：
    {text[:13000]}
    """
    
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            try:
                res_text = response.json()['candidates'][0]['content']['parts'][0]['text']
                match = re.search(r'\{.*\}', res_text, re.DOTALL)
                if match:
                    return "OK", json.loads(match.group(0))
                else:
                    return "ERROR", "JSON 解析失敗"
            except:
                return "ERROR", "AI 回傳結構異常"
        else:
            return "ERROR", f"API Error: {response.status_code}"
    except Exception as e:
        return "ERROR", str(e)

# --- 5. 主畫面 ---
st.title("🏗️ MCDM 層級架構分析工作區")

raw_text = st.text_area("請貼上文獻摘要：", height=250)

if st.button("🚀 執行層級分析 (構面->準則)", type="primary"):
    if not api_key:
        st.error("❌ 請輸入 Key")
    elif not raw_text:
        st.warning("⚠️ 請輸入文獻")
    else:
        with st.spinner(f"🔍 AI 正在運算：將 {criteria_size} 個準則歸納為 {dim_size} 個構面..."):
            valid_model = get_best_model(api_key)
            
            if not valid_model:
                st.error("❌ 找不到可用模型")
            else:
                status, result = run_hierarchy_analysis(raw_text, api_key, valid_model, thesis_topic, pool_size, criteria_size, dim_size)
                
                if status == "OK":
                    st.success("✅ 架構建立完成！")
                    
                    # 資料解析
                    papers = result.get("papers", [])
                    raw_pool = result.get("step1_raw_pool", [])
                    hierarchy = result.get("final_hierarchy", [])
                    
                    # 建立代號對照 (ID -> A, B, C)
                    id_to_code = {}
                    legend_rows = []
                    for idx, p in enumerate(papers):
                        code = string.ascii_uppercase[idx % 26]
                        id_to_code[p['id']] = code
                        legend_rows.append({"代號": code, "文獻來源 (APA)": p['apa']})
                    
                    df_legend = pd.DataFrame(legend_rows)
                    
                    # --- 建立分頁 ---
                    t1, t2, t3, t4 = st.tabs([
                        "1️⃣ Step 1: 原始池", 
                        "2️⃣ Step 2 & 3: 層級架構表", 
                        "3️⃣ Step 4: 矩陣圖", 
                        "4️⃣ 文獻對照"
                    ])
                    
                    # Tab 1: 原始池
                    with t1:
                        if raw_pool:
                            raw_rows = []
                            for i, item in enumerate(raw_pool):
                                ids = item.get("matched_ids", [])
                                codes = sorted([id_to_code.get(pid, "?") for pid in ids])
                                raw_rows.append({
                                    "序號": i + 1,
                                    "原始細項": item.get("name"),
                                    "出處": ", ".join(codes)
                                })
                            st.dataframe(pd.DataFrame(raw_rows), hide_index=True, use_container_width=True)
                            
                    # Tab 2: 層級架構 (構面 -> 準則)
                    with t2:
                        hier_rows = []
                        criterion_counter = 1
                        
                        for dim in hierarchy:
                            dim_name = dim.get("dimension_name")
                            criteria_list = dim.get("contained_criteria", [])
                            
                            for crit in criteria_list:
                                ids = crit.get("matched_paper_ids", [])
                                codes = sorted([id_to_code.get(pid, "?") for pid in ids])
                                
                                hier_rows.append({
                                    "層級一：構面 (Dimension)": dim_name,
                                    "層級二：準則 (Criteria)": crit.get("criteria_name"),
                                    "原始細項來源": ", ".join(crit.get("source_raw_items", [])),
                                    "出處代號": ", ".join(codes),
                                    "收斂理由": crit.get("reasoning")
                                })
                                criterion_counter += 1
                        
                        df_hier = pd.DataFrame(hier_rows)
                        st.dataframe(df_hier, hide_index=True, use_container_width=True)

                    # Tab 3: 矩陣圖 (左邊是準則，但在表格中可以加入構面欄位)
                    with t3:
                        matrix_rows = []
                        all_codes = [d["代號"] for d in legend_rows]
                        
                        for row_data in hier_rows: # 重用上面的資料
                            m_row = {
                                "構面": row_data["層級一：構面 (Dimension)"],
                                "準則": row_data["層級二：準則 (Criteria)"]
                            }
                            # 填點
                            source_codes = row_data["出處代號"].split(", ")
                            for code in all_codes:
                                m_row[code] = "●" if code in source_codes else ""
                            
                            matrix_rows.append(m_row)
                            
                        df_matrix = pd.DataFrame(matrix_rows)
                        st.dataframe(df_matrix, hide_index=True, use_container_width=True)

                    # Tab 4
                    with t4:
                        st.dataframe(df_legend, hide_index=True, use_container_width=True)
                        
                    # --- 下載 ---
                    st.divider()
                    output = BytesIO()
                    try:
                        import xlsxwriter
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            if raw_pool: pd.DataFrame(raw_rows).to_excel(writer, sheet_name='原始池', index=False)
                            df_hier.to_excel(writer, sheet_name='層級架構表', index=False)
                            df_matrix.to_excel(writer, sheet_name='矩陣圖', index=False)
                            df_legend.to_excel(writer, sheet_name='文獻對照', index=False)
                        st.download_button("📥 下載完整層級報告 Excel", output.getvalue(), "mcdm_hierarchy.xlsx", type="primary")
                    except:
                        st.error("Excel 匯出失敗")
                else:
                    st.error("分析失敗")
                    st.code(result)
