import streamlit as st
import pandas as pd
import requests
import json
import string
import re
from io import BytesIO

# --- 1. 基礎設定 ---
st.set_page_config(page_title="MCDM 全功能層級分析系統", layout="wide", page_icon="💎")

# --- 2. 側邊欄 (安全性 + 參數設定) ---
with st.sidebar:
    st.header("💎 系統設定")
    
    # === 安全性檢查：優先讀取 Secrets ===
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
        st.success("✅ 已載入雲端金鑰 (安全模式)")
    else:
        st.warning("⚠️ 未偵測到雲端金鑰")
        api_key = st.text_input("請手動輸入 API Key", type="password")
    # ===================================

    st.divider()
    thesis_topic = st.text_input("論文題目：", value="餐飲業導入 AI 服務之評估準則")
    
    st.subheader("📊 研究參數設定")
    st.caption("請設定您希望 AI 歸納的數量級距：")
    c1, c2, c3 = st.columns(3)
    with c1:
        pool_size = st.number_input("1.原始池", value=50, help="Step 1: 預計從文獻找出多少個細項")
    with c2:
        criteria_size = st.number_input("2.準則數", value=15, help="Step 2: 收斂後希望剩下多少個準則")
    with c3:
        dim_size = st.number_input("3.構面數", value=4, help="Step 3: 將準則歸納為幾個構面")

# --- 3. 模型自動適配 (防呆機制) ---
def get_best_model(key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            models = response.json().get('models', [])
            # 優先順序: Pro (聰明) -> Flash (快) -> 其他
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

# --- 4. 核心分析邏輯 (包含所有層級與矩陣資訊) ---
def run_full_analysis(text, key, model_name, topic, pool_n, crit_n, dim_n):
    url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent?key={key}"
    headers = {'Content-Type': 'application/json'}
    
    prompt = f"""
    你是一個 MCDM 研究專家。題目：{topic}。
    請執行完整的「文獻回顧 -> 發散 -> 收斂 -> 層級化 -> 矩陣化」流程。

    【任務流程】：
    1. **文獻處理**：辨識文獻並編號 (ID 0, 1...)，轉為 APA 格式。
    2. **Step 1 發散 (Pooling)**：找出約 {pool_n} 個「原始細項」，並標記出處 ID。
    3. **Step 2 收斂 (Convergence)**：將其歸納為 {crit_n} 個「評估準則 (Criteria)」。
    4. **Step 3 層級 (Hierarchy)**：將這 {crit_n} 個準則，歸納分類到 {dim_n} 個「評估構面 (Dimensions)」。

    【輸出 JSON 格式 (嚴格遵守)】：
    {{
      "papers": [
        {{ "id": 0, "apa": "作者A (2024). 標題..." }},
        ...
      ],
      "step1_raw_pool": [
        {{ "name": "原始細項名稱", "matched_ids": [0, 2] }},
        ... (約 {pool_n} 個)
      ],
      "final_hierarchy": [
        {{
          "dimension_name": "構面名稱 (如：財務構面)",
          "contained_criteria": [
             {{
               "criteria_name": "準則名稱 (如：建置成本)",
               "source_raw_items": ["原始細項A", "原始細項B"],
               "reasoning": "合併理由說明...",
               "matched_paper_ids": [0, 2]
             }},
             ...
          ]
        }},
        ... (共 {dim_n} 個構面)
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
                    return "ERROR", "JSON 解析失敗，AI 回傳格式不符。"
            except:
                return "ERROR", "AI 回傳結構異常。"
        else:
            return "ERROR", f"API Error: {response.status_code}"
    except Exception as e:
        return "ERROR", str(e)

# --- 5. 主畫面 ---
st.title("💎 MCDM 全功能層級分析工作區")

raw_text = st.text_area("請在此貼上文獻摘要：", height=250)

if st.button("🚀 執行全功能分析", type="primary"):
    if not api_key:
        st.error("❌ 請檢查側邊欄，確認已輸入 API Key 或設定 Secrets。")
    elif not raw_text:
        st.warning("⚠️ 請輸入文獻資料！")
    else:
        with st.spinner(f"🔍 AI 正在執行運算：發散({pool_size}) -> 收斂({criteria_size}) -> 構面({dim_size})..."):
            valid_model = get_best_model(api_key)
            
            if not valid_model:
                st.error("❌ 找不到可用模型，請確認 API Key 是否有效。")
            else:
                status, result = run_full_analysis(raw_text, api_key, valid_model, thesis_topic, pool_size, criteria_size, dim_size)
                
                if status == "OK":
                    st.success("✅ 分析完成！所有表格已生成。")
                    
                    # --- 資料前處理 ---
                    papers = result.get("papers", [])
                    raw_pool = result.get("step1_raw_pool", [])
                    hierarchy = result.get("final_hierarchy", [])
                    
                    # 建立代號對照 (ID -> A, B, C...)
                    id_to_code = {}
                    legend_rows = []
                    for idx, p in enumerate(papers):
                        code = string.ascii_uppercase[idx % 26]
                        id_to_code[p['id']] = code
                        legend_rows.append({"代號": code, "文獻來源 (APA)": p['apa']})
                    
                    df_legend = pd.DataFrame(legend_rows)
                    
                    # --- 建立 4 個分頁 ---
                    t1, t2, t3, t4 = st.tabs([
                        "1️⃣ Step 1: 原始發散池", 
                        "2️⃣ Step 2&3: 層級架構與收斂", 
                        "3️⃣ Step 4: 矩陣分析圖", 
                        "4️⃣ 文獻對照表"
                    ])
                    
                    # Tab 1: 原始池 (Raw Pool)
                    with t1:
                        if raw_pool:
                            raw_rows = []
                            for i, item in enumerate(raw_pool):
                                ids = item.get("matched_ids", [])
                                codes = sorted([id_to_code.get(pid, "?") for pid in ids])
                                raw_rows.append({
                                    "序號": i + 1,
                                    "原始細項準則": item.get("name"),
                                    "出處代號": ", ".join(codes)
                                })
                            st.subheader(f"Step 1: 原始文獻篩選 (共 {len(raw_rows)} 項)")
                            st.dataframe(pd.DataFrame(raw_rows), hide_index=True, use_container_width=True)
                        else:
                            st.warning("無資料")

                    # Tab 2: 層級架構 (Hierarchy)
                    with t2:
                        hier_rows = []
                        for dim in hierarchy:
                            dim_name = dim.get("dimension_name")
                            criteria_list = dim.get("contained_criteria", [])
                            
                            for crit in criteria_list:
                                ids = crit.get("matched_paper_ids", [])
                                codes = sorted([id_to_code.get(pid, "?") for pid in ids])
                                
                                hier_rows.append({
                                    "層級一：構面": dim_name,
                                    "層級二：準則": crit.get("criteria_name"),
                                    "涵蓋之原始細項": ", ".join(crit.get("source_raw_items", [])),
                                    "出處代號": ", ".join(codes),
                                    "收斂與歸納理由": crit.get("reasoning")
                                })
                        
                        st.subheader("Step 2 & 3: 準則收斂與層級架構")
                        st.dataframe(pd.DataFrame(hier_rows), hide_index=True, use_container_width=True)

                    # Tab 3: 矩陣圖 (Matrix)
                    with t3:
                        matrix_rows = []
                        all_codes = [d["代號"] for d in legend_rows]
                        
                        # 使用層級表的資料來建立矩陣
                        for row_data in hier_rows:
                            m_row = {
                                "構面": row_data["層級一：構面"],
                                "準則": row_data["層級二：準則"]
                            }
                            # 填入黑點
                            source_codes = row_data["出處代號"].split(", ")
                            for code in all_codes:
                                m_row[code] = "●" if code in source_codes else ""
                            
                            matrix_rows.append(m_row)
                            
                        st.subheader("Step 4: 準則 vs 文獻 矩陣圖")
                        st.dataframe(pd.DataFrame(matrix_rows), hide_index=True, use_container_width=True)

                    # Tab 4: 對照表 (Legend)
                    with t4:
                        st.subheader("文獻代號對照表")
                        st.dataframe(df_legend, hide_index=True, use_container_width=True)
                        
                    # --- 全表格下載 ---
                    st.divider()
                    output = BytesIO()
                    try:
                        import xlsxwriter
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            if raw_pool: pd.DataFrame(raw_rows).to_excel(writer, sheet_name='1_原始發散池', index=False)
                            pd.DataFrame(hier_rows).to_excel(writer, sheet_name='2_層級與收斂', index=False)
                            pd.DataFrame(matrix_rows).to_excel(writer, sheet_name='3_分析矩陣', index=False)
                            df_legend.to_excel(writer, sheet_name='4_文獻對照', index=False)
                        
                        st.download_button(
                            label="📥 下載完整 Excel 報告 (含所有表格)",
                            data=output.getvalue(),
                            file_name="MCDM_Full_Analysis.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            type="primary"
                        )
                    except Exception as e:
                        st.error(f"Excel 匯出模組發生錯誤: {e}")

                else:
                    st.error("分析失敗，請查看下方錯誤訊息：")
                    st.code(result)
