import streamlit as st
import pandas as pd
import requests
import json
import string
from io import BytesIO

# --- 1. 基礎設定 ---
st.set_page_config(page_title="MCDM 論文準則定向分析器", layout="wide", page_icon="🎯")

# --- 2. 側邊欄：設定區 ---
with st.sidebar:
    st.header("🎯 研究設定")
    st.info("請輸入您的論文題目與目標數量，AI 將為您量身打造評估準則。")
    
    # API Key 輸入
    api_key = st.text_input("Google API Key", type="password")
    
    st.divider()
    
    # 新增：論文題目輸入
    thesis_topic = st.text_input("您的論文/研究題目：", placeholder="例如：餐飲業導入 AI 服務之評估準則研究")
    
    # 新增：指定準則數量
    criteria_count = st.number_input("希望萃取的準則數量：", min_value=3, max_value=20, value=12, step=1)

# --- 3. 自動尋找可用模型 ---
def get_best_model(key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            models = response.json().get('models', [])
            for m in models:
                if 'gemini-1.5-pro' in m['name']: return m['name'] # 優先用 Pro (比較聰明)
            for m in models:
                if 'gemini-1.5-flash' in m['name']: return m['name']
            for m in models:
                if 'gemini' in m['name']: return m['name']
        return "models/gemini-1.5-flash"
    except:
        return "models/gemini-1.5-flash"

# --- 4. 核心分析邏輯 (MCDM 定向 Prompt) ---
def run_focused_mcdm(text, key, model_name, topic, count):
    url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent?key={key}"
    headers = {'Content-Type': 'application/json'}
    
    # 這是最強的 MCDM 定向指令
    prompt = f"""
    你是一個 MCDM（多準則決策分析）的研究專家。
    【使用者研究題目】：{topic}
    【目標】：請從提供的文獻中，歸納出最適合該題目的 {count} 個評估準則。

    【任務流程】：
    1. 閱讀所有文獻摘要。
    2. 根據「研究題目」，篩選出最相關的 {count} 個評估準則 (Criteria)。準則必須是名詞（如：建置成本、個資安全性、介面易用性）。
    3. 整理每一篇文獻的 APA 引用格式。
    4. 建立對照關係：這 {count} 個準則分別在哪幾篇文獻中被提到？

    【輸出格式】：
    請直接回傳純 JSON 格式 (不要 Markdown)，結構如下：
    {{
      "master_criteria": ["準則1", "準則2", ... "準則{count}"],
      "papers": [
        {{
          "apa": "作者 (年份). 文獻標題...",
          "matched_criteria": ["準則1", "準則3"] 
        }},
        ...
      ]
    }}
    注意：master_criteria 的數量必須盡量接近 {count} 個。matched_criteria 必須只包含 master_criteria 裡面的項目。

    【原始文獻資料】：
    {text[:13000]}
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
st.title("🎯 MCDM 準則定向分析工作區")

if not thesis_topic:
    st.warning("👈 請先在左側輸入您的「論文題目」，這樣 AI 才能抓得準！")

raw_text = st.text_area("請在此貼上文獻摘要：", height=300)

if st.button("🚀 開始分析 (依照題目與數量)", type="primary"):
    if not api_key:
        st.error("❌ 請先貼上 API Key！")
    elif not thesis_topic:
        st.error("❌ 請輸入論文題目！")
    elif not raw_text:
        st.warning("⚠️ 請輸入文獻資料！")
    else:
        with st.spinner(f"🔍 AI 正在根據題目「{thesis_topic}」歸納出 {criteria_count} 個準則..."):
            valid_model = get_best_model(api_key)
            
            # 呼叫 AI
            status, result_data = run_focused_mcdm(raw_text, api_key, valid_model, thesis_topic, criteria_count)
            
            if status == "OK":
                st.success("✅ 分析完成！")
                
                # 解析 JSON
                # result_data 結構: { "master_criteria": [...], "papers": [...] }
                
                try:
                    master_criteria = result_data.get("master_criteria", [])
                    papers = result_data.get("papers", [])
                    
                    if not master_criteria:
                        st.warning("AI 沒能抓到準則，請檢查文獻內容是否足夠豐富。")
                    else:
                        # 1. 顯示 AI 抓到的 Master List
                        st.subheader(f"🎯 AI 為您歸納的 {len(master_criteria)} 個關鍵準則")
                        final_criteria = st.multiselect("您可以手動微調 (刪減)：", options=master_criteria, default=master_criteria)
                        
                        if final_criteria:
                            # 2. 建立矩陣
                            matrix = {}
                            labels = []
                            apa_list = []
                            
                            for i, paper in enumerate(papers):
                                lbl = string.ascii_uppercase[i % 26]
                                labels.append(lbl)
                                apa_list.append(paper["apa"])
                                
                                # 檢查這篇論文是否包含選定的準則
                                paper_crits = paper.get("matched_criteria", [])
                                matrix[lbl] = ["●" if c in paper_crits else "" for c in final_criteria]
                            
                            # 3. 轉 DataFrame
                            df_matrix = pd.DataFrame(matrix, index=final_criteria)
                            df_legend = pd.DataFrame({"代號": labels, "文獻來源 (APA)": apa_list})
                            
                            st.divider()
                            
                            # 4. 顯示結果
                            c1, c2 = st.columns([1.5, 2.5])
                            with c1:
                                st.subheader("📊 準則檢核矩陣")
                                st.dataframe(df_matrix, use_container_width=True)
                            with c2:
                                st.subheader("📝 文獻對照表")
                                st.dataframe(df_legend, hide_index=True, use_container_width=True)
                            
                            # 5. 下載
                            output = BytesIO()
                            try:
                                import xlsxwriter
                                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                    df_matrix.to_excel(writer, sheet_name='MCDM矩陣')
                                    df_legend.to_excel(writer, sheet_name='文獻來源')
                                st.download_button("📥 下載 Excel 報表", output.getvalue(), "mcdm_thesis_analysis.xlsx", type="primary")
                            except:
                                st.download_button("📥 下載 CSV", df_matrix.to_csv().encode('utf-8-sig'), "mcdm_analysis.csv")

                except Exception as parse_err:
                    st.error("資料解析錯誤，可能是 AI 回傳格式不符。請重試一次。")
                    st.json(result_data)
            else:
                st.error("分析失敗")
                st.code(result_data)
