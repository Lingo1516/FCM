import streamlit as st
import pandas as pd
import requests
import json
import string
from io import BytesIO

# --- 1. 基礎設定 ---
st.set_page_config(page_title="MCDM 雙階段篩選分析器", layout="wide", page_icon="🧬")

# --- 2. 側邊欄 ---
with st.sidebar:
    st.header("🧬 雙表輸出設定")
    st.info("此版本將嚴格區分為「第一階段：原始列表」與「第二階段：收斂歸納」。")
    
    api_key = st.text_input("Google API Key", type="password")
    
    st.divider()
    
    thesis_topic = st.text_input("論文題目：", value="餐飲業導入 AI 服務之評估準則")
    
    # 設定兩階段數量
    c1, c2 = st.columns(2)
    with c1:
        pool_size = st.number_input("Step 1 原始數量", value=50)
    with c2:
        target_size = st.number_input("Step 2 收斂數量", value=15)

# --- 3. 模型選擇 ---
def get_best_model(key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            models = response.json().get('models', [])
            for m in models: # 優先用 Pro 處理複雜邏輯
                if 'gemini-1.5-pro' in m['name']: return m['name']
            for m in models:
                if 'gemini' in m['name']: return m['name']
        return "models/gemini-1.5-flash"
    except:
        return "models/gemini-1.5-flash"

# --- 4. 核心分析邏輯 (雙表專用 Prompt) ---
def run_two_stage_analysis(text, key, model_name, topic, pool_n, target_n):
    url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent?key={key}"
    headers = {'Content-Type': 'application/json'}
    
    prompt = f"""
    你是一個 MCDM 研究專家。使用者題目：{topic}。
    請執行嚴格的「兩階段準則篩選」，並回傳 JSON 資料。

    【任務 1：建立準則池 (Pooling)】
    從文獻中找出約 {pool_n} 個「原始細項準則 (Raw Criteria)」。
    這些是未經修飾的、散落在各文獻中的具體指標。

    【任務 2：準則收斂 (Convergence)】
    將上述原始準則進行邏輯分類與合併，歸納出 {target_n} 個「最終構面/準則 (Final Criteria)」。
    必須清楚說明每個最終準則包含了哪些原始準則，以及合併理由。

    【輸出格式 (JSON)】：
    {{
      "step1_raw_pool": [
        {{ "id": 1, "name": "原始準則A" }},
        {{ "id": 2, "name": "原始準則B" }},
        ... (約 {pool_n} 個)
      ],
      "step2_convergence": [
        {{
          "id": 1,
          "final_name": "最終準則名稱 (例如：營運成本)",
          "source_raw_items": ["原始準則A", "原始準則B"],
          "reasoning": "A與B皆涉及資金支出，故合併為成本構面..."
        }},
        ... (約 {target_n} 個)
      ]
    }}
    
    【原始文獻】：
    {text[:14000]}
    """
    
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            res_text = response.json()['candidates'][0]['content']['parts'][0]['text']
            clean_json = res_text.replace("```json", "").replace("```", "").strip()
            return "OK", json.loads(clean_json)
        else:
            return "ERROR", f"錯誤 ({response.status_code}): {response.text}"
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
        with st.spinner(f"🔍 AI 正在進行兩階段運算：先列出 {pool_size} 個，再收斂為 {target_size} 個..."):
            valid_model = get_best_model(api_key)
            status, result = run_two_stage_analysis(raw_text, api_key, valid_model, thesis_topic, pool_size, target_size)
            
            if status == "OK":
                st.success("✅ 分析完成！請查看下方兩個分頁。")
                
                # 建立兩個分頁
                tab1, tab2 = st.tabs(["📑 表一：原始準則池 (50項)", "🎯 表二：收斂歸納表 (15項)"])
                
                # --- Tab 1: 原始列表 ---
                with tab1:
                    raw_data = result.get("step1_raw_pool", [])
                    if raw_data:
                        df_raw = pd.DataFrame(raw_data)
                        df_raw.rename(columns={"id": "序號", "name": "原始細項準則名稱"}, inplace=True)
                        st.subheader(f"Step 1: 初始篩選準則 (共 {len(raw_data)} 項)")
                        st.dataframe(df_raw, hide_index=True, use_container_width=True)
                    else:
                        st.warning("AI 未能產生原始列表")

                # --- Tab 2: 收斂結果 ---
                with tab2:
                    conv_data = result.get("step2_convergence", [])
                    if conv_data:
                        rows = []
                        for item in conv_data:
                            # 整理資料格式
                            row = {
                                "序號": item.get("id"),
                                "最終準則名稱": item.get("final_name"),
                                "涵蓋之原始細項 (來自表一)": ", ".join(item.get("source_raw_items", [])),
                                "收斂/合併理由說明": item.get("reasoning") # 這是最重要的欄位
                            }
                            rows.append(row)
                        
                        df_conv = pd.DataFrame(rows)
                        st.subheader(f"Step 2: 最終收斂準則 (共 {len(conv_data)} 項)")
                        st.markdown("👉 **最右側欄位** 為詳細的歸納邏輯說明")
                        st.dataframe(df_conv, hide_index=True, use_container_width=True)
                        
                        # --- 下載區 ---
                        st.divider()
                        output = BytesIO()
                        try:
                            import xlsxwriter
                            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                if raw_data: df_raw.to_excel(writer, sheet_name='Step1_原始準則(50)', index=False)
                                if conv_data: df_conv.to_excel(writer, sheet_name='Step2_收斂準則(15)', index=False)
                            st.download_button("📥 下載完整雙表 Excel", output.getvalue(), "mcdm_two_stage.xlsx", type="primary")
                        except:
                            st.error("Excel 模組錯誤")

            else:
                st.error("分析失敗")
                st.code(result)
