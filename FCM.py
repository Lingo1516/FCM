import streamlit as st
import pandas as pd
import requests
import json
import string
import re
from io import BytesIO

# --- 1. 基礎設定 ---
st.set_page_config(page_title="MCDM 雙階段分析 (自動適配穩定版)", layout="wide", page_icon="🧬")

# --- 2. 側邊欄 ---
with st.sidebar:
    st.header("🧬 設定")
    st.info("此版本會自動偵測您的金鑰權限，優先使用可用的模型，避免 404 錯誤。")
    
    api_key = st.text_input("Google API Key", type="password")
    
    st.divider()
    
    thesis_topic = st.text_input("論文題目：", value="餐飲業導入 AI 服務之評估準則")
    
    c1, c2 = st.columns(2)
    with c1:
        pool_size = st.number_input("Step 1 原始數量", value=50)
    with c2:
        target_size = st.number_input("Step 2 收斂數量", value=15)

# --- 3. 核心：自動尋找可用的模型 (修復 404 的關鍵) ---
def get_best_model(key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            models = response.json().get('models', [])
            
            # 1. 先找 Flash (通常最穩，權限最開放)
            for m in models:
                if 'gemini-1.5-flash' in m['name']: return m['name']
            
            # 2. 如果沒有，找 Pro
            for m in models:
                if 'gemini-1.5-pro' in m['name']: return m['name']
            
            # 3. 再沒有，找任何 Gemini
            for m in models:
                if 'gemini' in m['name'] and 'generateContent' in m.get('supportedGenerationMethods', []):
                    return m['name']
            
            return None # 真的找不到
        else:
            return None
    except:
        return None

# --- 4. 分析邏輯 (含 JSON 強力清洗) ---
def run_two_stage_analysis(text, key, model_name, topic, pool_n, target_n):
    url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent?key={key}"
    headers = {'Content-Type': 'application/json'}
    
    prompt = f"""
    你是一個 MCDM 研究專家。使用者題目：{topic}。
    請執行嚴格的「兩階段準則篩選」，並回傳 JSON 資料。

    【任務 1：建立準則池 (Pooling)】
    從文獻中找出約 {pool_n} 個「原始細項準則 (Raw Criteria)」。

    【任務 2：準則收斂 (Convergence)】
    將上述原始準則進行邏輯分類與合併，歸納出 {target_n} 個「最終構面/準則 (Final Criteria)」。
    
    【重要】：在 "step2_convergence" 中，必須包含 "reasoning" 欄位，詳細解釋該最終準則是由哪些原始準則合併而來，以及原因。

    【輸出格式 (JSON Only)】：
    請務必只回傳 JSON，不要有 markdown code block。
    {{
      "step1_raw_pool": [
        {{ "id": 1, "name": "原始準則A" }},
        ... (約 {pool_n} 個)
      ],
      "step2_convergence": [
        {{
          "id": 1,
          "final_name": "最終準則名稱",
          "source_raw_items": ["原始準則A", "原始準則B"],
          "reasoning": "合併理由..."
        }},
        ... (約 {target_n} 個)
      ]
    }}
    
    文獻：
    {text[:13000]}
    """
    
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            try:
                res_text = response.json()['candidates'][0]['content']['parts'][0]['text']
            except:
                return "ERROR", "AI 回傳結構異常。"

            # 強力清洗：只抓取 { ... }
            match = re.search(r'\{.*\}', res_text, re.DOTALL)
            if match:
                clean_json_str = match.group(0)
                try:
                    return "OK", json.loads(clean_json_str)
                except json.JSONDecodeError as e:
                    return "ERROR", f"JSON 解析失敗 (格式錯誤)。\n內容片段: {clean_json_str[:200]}"
            else:
                return "ERROR", f"找不到 JSON 結構。\nAI 回傳: {res_text[:200]}"
        else:
            return "ERROR", f"API 連線錯誤 ({response.status_code}): {response.text}"
    except Exception as e:
        return "ERROR", str(e)

# --- 5. 主畫面 ---
st.title("🧬 MCDM 準則：雙階段報告生成")

raw_text = st.text_area("請在此貼上文獻摘要：", height=250)

if st.button("🚀 執行雙階段分析", type="primary"):
    if not api_key:
        st.error("❌ 請輸入 Key")
    elif not raw_text:
        st.warning("⚠️ 請輸入文獻")
    else:
        with st.spinner("🔍 正在自動偵測可用模型並執行分析..."):
            
            # 1. 自動找模型 (這步最重要)
            valid_model = get_best_model(api_key)
            
            if not valid_model:
                st.error("❌ 無法找到任何可用模型。請檢查 Key 是否正確或權限是否開啟。")
            else:
                st.success(f"✅ 連線成功！自動選用模型：`{valid_model}`")
                
                # 2. 執行分析
                status, result = run_two_stage_analysis(raw_text, api_key, valid_model, thesis_topic, pool_size, target_size)
                
                if status == "OK":
                    st.success("✅ 分析完成！")
                    
                    tab1, tab2 = st.tabs(["📑 Step 1: 原始準則池", "🎯 Step 2: 最終收斂表"])
                    
                    # Tab 1
                    with tab1:
                        raw_data = result.get("step1_raw_pool", [])
                        if raw_data:
                            df_raw = pd.DataFrame(raw_data)
                            if "id" in df_raw.columns:
                                df_raw.rename(columns={"id": "序號", "name": "原始細項準則"}, inplace=True)
                            st.dataframe(df_raw, hide_index=True, use_container_width=True)

                    # Tab 2
                    with tab2:
                        conv_data = result.get("step2_convergence", [])
                        if conv_data:
                            rows = []
                            for item in conv_data:
                                row = {
                                    "序號": item.get("id"),
                                    "最終準則": item.get("final_name"),
                                    "涵蓋之原始細項": ", ".join(item.get("source_raw_items", [])),
                                    "收斂邏輯/原因": item.get("reasoning")
                                }
                                rows.append(row)
                            
                            df_conv = pd.DataFrame(rows)
                            st.markdown("👉 **最右邊有詳細的收斂原因**")
                            st.dataframe(df_conv, hide_index=True, use_container_width=True)
                            
                            # 下載
                            output = BytesIO()
                            try:
                                import xlsxwriter
                                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                    if raw_data: df_raw.to_excel(writer, sheet_name='Step1_原始', index=False)
                                    if conv_data: df_conv.to_excel(writer, sheet_name='Step2_收斂', index=False)
                                st.download_button("📥 下載完整 Excel", output.getvalue(), "mcdm_final.xlsx", type="primary")
                            except:
                                st.error("Excel 匯出失敗")
                else:
                    st.error("分析失敗，原因如下：")
                    st.code(result)
