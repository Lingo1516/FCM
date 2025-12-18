import streamlit as st
import pandas as pd
import requests
import json
import string
import re
from io import BytesIO

# --- 1. 基礎設定 ---
st.set_page_config(page_title="MCDM 全功能分析 (含出處註記)", layout="wide", page_icon="💎")

# --- 2. 側邊欄 ---
with st.sidebar:
    st.header("💎 全功能設定")
    st.info("此版本已在「原始表」與「收斂表」的最右側增加【作者代號】欄位。")
    
    api_key = st.text_input("Google API Key", type="password")
    st.divider()
    thesis_topic = st.text_input("論文題目：", value="餐飲業導入 AI 服務之評估準則")
    
    c1, c2 = st.columns(2)
    with c1:
        pool_size = st.number_input("原始發散數量", value=50)
    with c2:
        target_size = st.number_input("最終收斂數量", value=15)

# --- 3. 自動適配模型 ---
def get_best_model(key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            models = response.json().get('models', [])
            # 優先權：Pro (邏輯好) > Flash (速度快)
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

# --- 4. 核心分析邏輯 (更新 Prompt 以索取 Step 1 的出處) ---
def run_all_in_one_analysis(text, key, model_name, topic, pool_n, target_n):
    url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent?key={key}"
    headers = {'Content-Type': 'application/json'}
    
    prompt = f"""
    你是一個 MCDM 研究專家。題目：{topic}。
    請執行完整的「發散 -> 收斂 -> 矩陣」流程。

    【任務要求】：
    1. 辨識文獻並編號 (ID 0, 1, 2...)，轉為 APA 格式。
    2. Step 1 (Pooling): 從文中找出約 {pool_n} 個「原始細項準則」。
       **重要：** 針對每一個原始細項，請標註是哪幾篇文獻提到的 (Paper IDs)。
    3. Step 2 (Convergence): 將其歸納為 {target_n} 個「最終準則」。
       - 說明每個最終準則是由哪些原始項目合併的。
       - 說明合併/收斂的理由 (Reasoning)。
       - 標註每個最終準則出現在哪幾篇論文中 (Paper IDs)。

    【回傳 JSON 格式 (嚴格遵守)】：
    {{
      "papers": [
        {{ "id": 0, "apa": "作者A (2024). 標題..." }},
        {{ "id": 1, "apa": "作者B (2023). 標題..." }}
      ],
      "step1_raw_pool": [
        {{ "name": "原始細項1", "matched_ids": [0, 1] }},
        {{ "name": "原始細項2", "matched_ids": [2] }},
        ... (約 {pool_n} 個)
      ],
      "step2_convergence": [
        {{
          "id": 1,
          "final_name": "最終準則名稱",
          "source_raw_items": ["細項1", "細項5"],
          "reasoning": "因為皆涉及財務支出...",
          "matched_paper_ids": [0, 2] 
        }},
        ... (共 {target_n} 個)
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
st.title("💎 MCDM 完整研究報告 (含出處註記)")

raw_text = st.text_area("請貼上文獻摘要：", height=250)

if st.button("🚀 執行全功能分析", type="primary"):
    if not api_key:
        st.error("❌ 請輸入 Key")
    elif not raw_text:
        st.warning("⚠️ 請輸入文獻")
    else:
        with st.spinner("🔍 AI 正在處理：發散(含出處) -> 收斂(含出處) -> 矩陣建構..."):
            valid_model = get_best_model(api_key)
            
            if not valid_model:
                st.error("❌ 找不到可用模型")
            else:
                status, result = run_all_in_one_analysis(raw_text, api_key, valid_model, thesis_topic, pool_size, target_size)
                
                if status == "OK":
                    st.success("✅ 分析完成！")
                    
                    # 準備資料
                    papers = result.get("papers", [])
                    raw_pool = result.get("step1_raw_pool", [])
                    conv_data = result.get("step2_convergence", [])
                    
                    # 建立代號對照 Map (id -> A, B, C...)
                    id_to_code = {}
                    legend_rows = []
                    for idx, p in enumerate(papers):
                        code = string.ascii_uppercase[idx % 26]
                        id_to_code[p['id']] = code
                        legend_rows.append({"代號": code, "文獻來源 (APA)": p['apa']})
                    
                    df_legend = pd.DataFrame(legend_rows)
                    
                    # --- 建立 4 個分頁 ---
                    t1, t2, t3, t4 = st.tabs([
                        "1️⃣ Step 1: 原始列表 (50)", 
                        "2️⃣ Step 2: 收斂邏輯 (15)", 
                        "3️⃣ Step 3: 分析矩陣圖", 
                        "4️⃣ 文獻代號對照"
                    ])
                    
                    # Tab 1: 原始池 (增加出處欄位)
                    with t1:
                        if raw_pool:
                            raw_rows = []
                            for i, item in enumerate(raw_pool):
                                # 判斷 item 是字串還是字典 (為了相容性)
                                name = item["name"] if isinstance(item, dict) else str(item)
                                ids = item.get("matched_ids", []) if isinstance(item, dict) else []
                                codes = [id_to_code.get(pid, "?") for pid in ids]
                                codes.sort()
                                
                                raw_rows.append({
                                    "序號": i + 1,
                                    "原始細項準則": name,
                                    "出處代號": ", ".join(codes)  # 這裡就是你要的 A, B, C
                                })
                            
                            df_raw = pd.DataFrame(raw_rows)
                            st.dataframe(df_raw, hide_index=True, use_container_width=True)
                        else:
                            st.warning("無資料")
                            
                    # Tab 2: 收斂邏輯 (增加出處欄位)
                    with t2:
                        conv_rows = []
                        for item in conv_data:
                            # 找出對應的代號
                            ids = item.get("matched_paper_ids", [])
                            codes = [id_to_code.get(pid, "?") for pid in ids]
                            codes.sort()
                            
                            conv_rows.append({
                                "序號": item.get("id"),
                                "最終準則": item.get("final_name"),
                                "涵蓋之原始細項": ", ".join(item.get("source_raw_items", [])),
                                "收斂/合併理由": item.get("reasoning"),
                                "出處代號": ", ".join(codes) # 這裡就是你要的 A, C, D
                            })
                        df_conv = pd.DataFrame(conv_rows)
                        st.dataframe(df_conv, hide_index=True, use_container_width=True)
                        
                    # Tab 3: 矩陣圖 (保持不變，因為這是你要的黑點點)
                    with t3:
                        matrix_rows = []
                        all_codes = [d["代號"] for d in legend_rows]
                        
                        for item in conv_data:
                            row = {"最終準則名稱": item.get("final_name")}
                            matched = item.get("matched_paper_ids", [])
                            
                            for code in all_codes:
                                target_id = -1
                                for pid, pcode in id_to_code.items():
                                    if pcode == code: target_id = pid
                                
                                if target_id in matched:
                                    row[code] = "●"
                                else:
                                    row[code] = ""
                            matrix_rows.append(row)
                        
                        df_matrix = pd.DataFrame(matrix_rows)
                        st.dataframe(df_matrix, hide_index=True, use_container_width=True)
                        
                    # Tab 4: 文獻對照
                    with t4:
                        st.dataframe(df_legend, hide_index=True, use_container_width=True)
                        
                    # --- 下載 ---
                    st.divider()
                    output = BytesIO()
                    try:
                        import xlsxwriter
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            if raw_pool: df_raw.to_excel(writer, sheet_name='Step1_原始(50)', index=False)
                            if conv_data: df_conv.to_excel(writer, sheet_name='Step2_收斂(15)', index=False)
                            if conv_data: df_matrix.to_excel(writer, sheet_name='Step3_矩陣', index=False)
                            df_legend.to_excel(writer, sheet_name='文獻對照表', index=False)
                        st.download_button("📥 下載完整 Excel (含出處註記)", output.getvalue(), "mcdm_full_report.xlsx", type="primary")
                    except:
                        st.error("Excel 匯出模組錯誤")

                else:
                    st.error("分析失敗")
                    st.code(result)
