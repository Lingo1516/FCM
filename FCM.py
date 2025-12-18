import streamlit as st
import pandas as pd
import requests
import json
import string
import re
from io import BytesIO

# --- 1. 基礎設定 ---
st.set_page_config(page_title="MCDM 矩陣回歸版", layout="wide", page_icon="📊")

# --- 2. 側邊欄 ---
with st.sidebar:
    st.header("📊 設定")
    st.info("此版本已修復：矩陣圖 (黑點點) 與 作者對照表 (A, B, C) 將完整呈現。")
    
    api_key = st.text_input("Google API Key", type="password")
    st.divider()
    thesis_topic = st.text_input("論文題目：", value="餐飲業導入 AI 服務之評估準則")
    
    c1, c2 = st.columns(2)
    with c1:
        pool_size = st.number_input("Step 1 原始數量", value=50)
    with c2:
        target_size = st.number_input("Step 2 收斂數量", value=15)

# --- 3. 自動找模型 (防 404) ---
def get_best_model(key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            models = response.json().get('models', [])
            # 優先找 Pro (邏輯好)，沒有就找 Flash (速度快)，再沒有就隨便抓
            priority = ['gemini-1.5-pro', 'gemini-1.5-flash', 'gemini-1.0-pro']
            
            # 建立可用模型清單
            available = [m['name'] for m in models if 'generateContent' in m.get('supportedGenerationMethods', [])]
            
            # 依照優先順序媒合
            for p in priority:
                for a in available:
                    if p in a: return a
            
            return available[0] if available else None
        return None
    except:
        return None

# --- 4. 核心分析邏輯 (矩陣專用) ---
def run_matrix_analysis(text, key, model_name, topic, pool_n, target_n):
    url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent?key={key}"
    headers = {'Content-Type': 'application/json'}
    
    prompt = f"""
    你是一個 MCDM 研究專家。題目：{topic}。
    請閱讀文獻，並產生一個「準則 vs 文獻」的矩陣資料。

    【步驟 1】：先辨識文獻中有幾篇不同的論文，並給予編號 (0, 1, 2...) 與 APA 格式。
    【步驟 2】：找出約 {pool_n} 個原始準則。
    【步驟 3】：歸納出 {target_n} 個「最終準則」，並指明每一項準則出現在「哪幾篇論文 (編號)」中。

    【回傳格式 JSON Only】：
    {{
      "papers": [
        {{ "id": 0, "apa": "作者A (2024). 標題..." }},
        {{ "id": 1, "apa": "作者B (2023). 標題..." }}
      ],
      "step1_raw_pool": [ "原始準則1", "原始準則2", ... ],
      "step2_matrix": [
        {{
          "final_name": "最終準則名稱 (如：服務品質)",
          "matched_paper_ids": [0, 2] // 代表第0篇和第2篇有提到
        }},
        ...
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
                # 清洗 JSON
                match = re.search(r'\{.*\}', res_text, re.DOTALL)
                if match:
                    return "OK", json.loads(match.group(0))
                else:
                    return "ERROR", "無法解析 JSON 結構"
            except:
                return "ERROR", "AI 回傳格式異常"
        else:
            return "ERROR", f"API Error: {response.status_code}"
    except Exception as e:
        return "ERROR", str(e)

# --- 5. 主畫面 ---
st.title("📊 MCDM 準則矩陣生成器")

raw_text = st.text_area("請貼上文獻摘要：", height=250)

if st.button("🚀 生成矩陣與對照表", type="primary"):
    if not api_key:
        st.error("❌ 請輸入 Key")
    elif not raw_text:
        st.warning("⚠️ 請輸入文獻")
    else:
        with st.spinner("🔍 正在重建矩陣與作者名單..."):
            valid_model = get_best_model(api_key)
            
            if not valid_model:
                st.error("❌ 找不到可用模型 (Key 可能權限不足)")
            else:
                status, result = run_matrix_analysis(raw_text, api_key, valid_model, thesis_topic, pool_size, target_size)
                
                if status == "OK":
                    st.success("✅ 生成成功！")
                    
                    # 1. 解析論文清單 (建立 A, B, C...)
                    papers = result.get("papers", [])
                    paper_map = {} # id -> "A"
                    legend_data = []
                    
                    for idx, p in enumerate(papers):
                        code = string.ascii_uppercase[idx % 26]
                        p_id = p.get("id")
                        paper_map[p_id] = code
                        legend_data.append({"代號": code, "文獻來源 (APA)": p.get("apa")})
                    
                    df_legend = pd.DataFrame(legend_data)
                    
                    # 2. 解析矩陣 (建立黑點點)
                    matrix_data = result.get("step2_matrix", [])
                    rows = []
                    
                    # 準備所有的欄位 A, B, C...
                    all_codes = [d["代號"] for d in legend_data]
                    
                    for item in matrix_data:
                        row = {"最終準則名稱": item.get("final_name")}
                        matched_ids = item.get("matched_paper_ids", [])
                        
                        # 填入黑點
                        for code in all_codes:
                            # 找出這個 code 對應的 id
                            # (這裡簡單處理，假設順序一致)
                            # 嚴謹作法：反查
                            target_id = -1
                            for pid, pcode in paper_map.items():
                                if pcode == code:
                                    target_id = pid
                                    break
                            
                            if target_id in matched_ids:
                                row[code] = "●"
                            else:
                                row[code] = ""
                        rows.append(row)
                        
                    df_matrix = pd.DataFrame(rows)
                    
                    # --- 顯示結果 ---
                    
                    st.subheader("1️⃣ 分析矩陣 (準則 vs 文獻)")
                    st.dataframe(df_matrix, hide_index=True, use_container_width=True)
                    
                    st.subheader("2️⃣ 文獻代號對照表")
                    st.dataframe(df_legend, hide_index=True, use_container_width=True)
                    
                    # --- 下載 ---
                    output = BytesIO()
                    try:
                        import xlsxwriter
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            df_matrix.to_excel(writer, sheet_name='矩陣圖', index=False)
                            df_legend.to_excel(writer, sheet_name='文獻對照', index=False)
                            
                            # 原始 Step 1 池子也放進去
                            raw_pool = result.get("step1_raw_pool", [])
                            if raw_pool:
                                pd.DataFrame({"原始準則": raw_pool}).to_excel(writer, sheet_name='原始準則池', index=False)
                                
                        st.download_button("📥 下載 Excel (含矩陣與APA)", output.getvalue(), "mcdm_matrix.xlsx", type="primary")
                    except:
                        st.error("Excel 匯出模組錯誤")

                else:
                    st.error("分析失敗")
                    st.code(result)
