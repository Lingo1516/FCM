import streamlit as st
import pandas as pd
import requests
import json
import string
from io import BytesIO

# --- 1. 基礎設定 ---
st.set_page_config(page_title="MCDM 兩階段收斂分析器", layout="wide", page_icon="🧬")

# --- 2. 側邊欄：參數設定 ---
with st.sidebar:
    st.header("🧬 兩階段收斂設定")
    st.info("此系統將執行：發散 (找出大量細項) -> 收斂 (歸納核心準則) 的邏輯運算。")
    
    api_key = st.text_input("Google API Key", type="password")
    
    st.divider()
    
    # 研究題目
    thesis_topic = st.text_input("您的論文題目：", value="餐飲業導入 AI 服務之評估準則")
    
    # 兩階段參數
    c1, c2 = st.columns(2)
    with c1:
        pool_size = st.number_input("第一步：廣泛列出", value=50, help="希望 AI 先從文獻抓出多少個細項")
    with c2:
        target_size = st.number_input("第二步：收斂成", value=15, help="最後希望歸納成幾個主要準則")

# --- 3. 自動尋找可用模型 ---
def get_best_model(key):
    # 優先嘗試 Pro 模型，因為收斂邏輯需要較強的推理能力
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            models = response.json().get('models', [])
            # 優先順序: 1.5-Pro -> 1.5-Flash
            for m in models:
                if 'gemini-1.5-pro' in m['name']: return m['name']
            for m in models:
                if 'gemini-1.5-flash' in m['name']: return m['name']
            for m in models:
                if 'gemini' in m['name']: return m['name']
        return "models/gemini-1.5-pro"
    except:
        return "models/gemini-1.5-flash"

# --- 4. 核心分析邏輯 (雙階段收斂 Prompt) ---
def run_convergence_analysis(text, key, model_name, topic, pool_n, target_n):
    url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent?key={key}"
    headers = {'Content-Type': 'application/json'}
    
    # 這個 Prompt 是整個程式的靈魂
    prompt = f"""
    你是一個 MCDM 研究方法的專家。
    【研究題目】：{topic}
    【任務目標】：請執行「兩階段準則篩選法」。
    
    【階段一：發散 (Brainstorming)】
    請先閱讀文獻，從中盡可能找出約 {pool_n} 個與題目相關的「原始細項準則 (Raw Criteria)」。

    【階段二：收斂 (Convergence)】
    請運用你的邏輯，將上述細項準則進行合併、分類，歸納出最具代表性的 {target_n} 個「最終準則 (Final Criteria)」。
    
    【輸出要求】：
    1. 建立矩陣：標示每一篇文獻是否提到了該「最終準則」。
    2. 解釋邏輯：必須詳細說明每個「最終準則」是由哪些「原始細項」合併而來，以及合併的理由。
    
    請直接回傳純 JSON 格式 (不要 Markdown)，結構嚴格如下：
    {{
      "final_dimensions": [
        {{
          "id": 1,
          "name": "最終準則名稱 (例如：營運成本)",
          "composition_logic": "本準則合併了：原始細項A、原始細項B。原因：它們都屬於成本結構...",
          "matched_papers_indices": [0, 2] // 代表第1篇和第3篇文獻有提到此準則
        }},
        ... (共 {target_n} 個)
      ],
      "papers": [
        "文獻1的APA格式 citation...",
        "文獻2的APA格式 citation...",
        ...
      ]
    }}

    【原始文獻資料】：
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
st.title("🧬 MCDM 準則：發散與收斂工作區")

raw_text = st.text_area("請在此貼上文獻摘要 (AI 會先抓大池子，再收斂成精華)：", height=300)

if st.button("🚀 執行收斂運算", type="primary"):
    if not api_key:
        st.error("❌ 請先貼上 API Key！")
    elif not raw_text:
        st.warning("⚠️ 請輸入文獻資料！")
    else:
        status_box = st.empty()
        status_box.info(f"🔍 AI 正在思考：先找出 {pool_size} 個細項，再邏輯歸納為 {target_size} 個主準則...")
        
        valid_model = get_best_model(api_key)
        
        # 執行分析
        status, result_data = run_convergence_analysis(raw_text, api_key, valid_model, thesis_topic, pool_size, target_size)
        
        if status == "OK":
            status_box.success("✅ 收斂完成！邏輯矩陣已生成。")
            
            try:
                # 解析資料
                dimensions = result_data.get("final_dimensions", [])
                papers_list = result_data.get("papers", [])
                
                # 準備 DataFrame 的資料容器
                # 欄位順序：序號 | 最終準則 | [文獻A] | [文獻B]... | 收斂邏輯說明 (最右邊)
                
                rows = []
                paper_labels = [string.ascii_uppercase[i % 26] for i in range(len(papers_list))]
                
                for dim in dimensions:
                    row_data = {}
                    # 1. 序號
                    row_data["序號"] = dim.get("id")
                    # 2. 準則名稱
                    row_data["最終評估準則"] = dim.get("name")
                    
                    # 3. 文獻矩陣 (中間)
                    matched_indices = dim.get("matched_papers_indices", [])
                    for idx, label in enumerate(paper_labels):
                        row_data[label] = "●" if idx in matched_indices else ""
                    
                    # 4. 收斂邏輯 (最右邊 - 這是你特別要求的)
                    row_data["收斂邏輯與原始細項來源"] = dim.get("composition_logic")
                    
                    rows.append(row_data)
                
                # 建立主表
                df_main = pd.DataFrame(rows)
                
                # 建立文獻對照表
                df_papers = pd.DataFrame({
                    "代號": paper_labels,
                    "文獻來源 (APA)": papers_list
                })
                
                st.divider()
                
                # 顯示區域
                st.subheader("📊 收斂結果矩陣")
                st.markdown("請向右滑動表格查看最右側的**「收斂邏輯」**欄位 👉")
                st.dataframe(df_main, hide_index=True, use_container_width=True)
                
                st.subheader("📝 文獻來源對照")
                st.dataframe(df_papers, hide_index=True, use_container_width=True)
                
                # 下載功能
                output = BytesIO()
                try:
                    import xlsxwriter
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df_main.to_excel(writer, sheet_name='收斂矩陣', index=False)
                        df_papers.to_excel(writer, sheet_name='文獻來源', index=False)
                    st.download_button("📥 下載完整 Excel (含邏輯說明)", output.getvalue(), "mcdm_convergence.xlsx", type="primary")
                except:
                    st.download_button("📥 下載 CSV", df_main.to_csv().encode('utf-8-sig'), "mcdm_convergence.csv")

            except Exception as e:
                st.error(f"資料解析發生錯誤：{e}")
                st.json(result_data)
        else:
            status_box.error("分析失敗")
            st.code(result_data)
