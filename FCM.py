import streamlit as st
import pandas as pd
import requests
import json
import string
import re
from io import BytesIO

# --- 1. 基礎設定 ---
st.set_page_config(page_title="MCDM 雙階段篩選分析器 (強效版)", layout="wide", page_icon="🧬")

# --- 2. 側邊欄 ---
with st.sidebar:
    st.header("🧬 雙表輸出設定")
    st.info("此版本增加了 JSON 強力清洗功能，能防止格式錯誤。")
    
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
    # 優先嘗試 Pro，若失敗則退回 Flash
    return "models/gemini-1.5-pro"

# --- 4. 核心分析邏輯 (增加 Regex 清洗) ---
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

    【輸出格式 (JSON Only)】：
    請務必只回傳 JSON，不要有 markdown code block，不要有解釋文字。
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
    {text[:12000]}
    """
    
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            try:
                res_text = response.json()['candidates'][0]['content']['parts'][0]['text']
            except:
                return "ERROR", "AI 回傳結構異常，可能被安全阻擋。"

            # --- 強力清洗：只抓取第一個 { 到最後一個 } 之間的內容 ---
            match = re.search(r'\{.*\}', res_text, re.DOTALL)
            if match:
                clean_json_str = match.group(0)
                try:
                    return "OK", json.loads(clean_json_str)
                except json.JSONDecodeError as e:
                    return "ERROR", f"JSON 解析失敗: {e}\n\nAI 回傳原始內容:\n{clean_json_str}"
            else:
                return "ERROR", f"找不到 JSON 結構。\nAI 回傳內容:\n{res_text}"
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
        with st.spinner(f"🔍 AI 正在進行運算 (這可能需要 30 秒)..."):
            # 這裡我們用 try-except 包起來，如果 Pro 失敗自動切換 Flash
            try:
                valid_model = "models/gemini-1.5-pro"
                status, result = run_two_stage_analysis(raw_text, api_key, valid_model, thesis_topic, pool_size, target_size)
            except:
                st.warning("Pro 模型失敗，切換至 Flash 模型重試...")
                valid_model = "models/gemini-1.5-flash"
                status, result = run_two_stage_analysis(raw_text, api_key, valid_model, thesis_topic, pool_size, target_size)
            
            if status == "OK":
                st.success("✅ 分析完成！")
                
                # 建立兩個分頁
                tab1, tab2 = st.tabs(["📑 表一：原始準則池 (50項)", "🎯 表二：收斂歸納表 (15項)"])
                
                # --- Tab 1 ---
                with tab1:
                    raw_data = result.get("step1_raw_pool", [])
                    if raw_data:
                        df_raw = pd.DataFrame(raw_data)
                        if "id" in df_raw.columns:
                            df_raw.rename(columns={"id": "序號", "name": "原始細項準則名稱"}, inplace=True)
                        st.dataframe(df_raw, hide_index=True, use_container_width=True)

                # --- Tab 2 ---
                with tab2:
                    conv_data = result.get("step2_convergence", [])
                    if conv_data:
                        rows = []
                        for item in conv_data:
                            row = {
                                "序號": item.get("id"),
                                "最終準則名稱": item.get("final_name"),
                                "涵蓋之原始細項": ", ".join(item.get("source_raw_items", [])),
                                "收斂/合併理由說明": item.get("reasoning")
                            }
                            rows.append(row)
                        
                        df_conv = pd.DataFrame(rows)
                        st.markdown("👉 **向右滑動查看詳細歸納邏輯**")
                        st.dataframe(df_conv, hide_index=True, use_container_width=True)
                        
                        # 下載
                        output = BytesIO()
                        try:
                            import xlsxwriter
                            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                if raw_data: df_raw.to_excel(writer, sheet_name='Step1_原始準則', index=False)
                                if conv_data: df_conv.to_excel(writer, sheet_name='Step2_收斂準則', index=False)
                            st.download_button("📥 下載完整雙表 Excel", output.getvalue(), "mcdm_two_stage.xlsx", type="primary")
                        except:
                            st.error("Excel 模組錯誤")
            else:
                st.error("分析失敗，請查看下方錯誤訊息：")
                st.code(result) # 把錯誤訊息直接印出來
