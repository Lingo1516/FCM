import streamlit as st
from groq import Groq
import google.generativeai as genai
import time
import pandas as pd

# --- 系統設定 ---
st.set_page_config(page_title="論文寫作助手 (安全無警報版)", layout="wide", page_icon="🛡️")

# --- 側邊欄：引擎與金鑰設定 ---
with st.sidebar:
    st.header("⚙️ 引擎設定")
    
    # 選擇引擎
    engine_choice = st.radio("選擇 AI 模型", ["Groq (Llama 3)", "Google (Gemini)"])
    
    api_key = ""
    if engine_choice == "Groq (Llama 3)":
        st.info("推薦！速度快，適合處理大量文字。")
        # ⬇️ 這裡改為空字串，避免觸發 Secret Scanning 警告
        api_key = st.text_input("請輸入 Groq Key (gsk_...)", type="password")
        st.markdown("[👉 點此免費申請 Groq Key](https://console.groq.com/keys)")
    else:
        st.info("備用。Google 引擎。")
        api_key = st.text_input("請輸入 Google Key", type="password")
        st.markdown("[👉 點此免費申請 Google Key](https://aistudio.google.com/app/apikey)")

    st.divider()

    # 關鍵字
    business_keywords = ["策略管理", "ESG", "CSR", "消費者行為", "滿意度", "供應鏈", "FinTech", "數位轉型"]
    selected_kws = st.multiselect("選擇關鍵字：", business_keywords)
    custom_kw = st.text_input("自訂關鍵字：")
    final_kws = selected_kws + ([custom_kw] if custom_kw else [])
    keywords_str = ", ".join(final_kws)

    # 方法
    method_category = st.selectbox("方法分類", ["MCDM", "量化", "質性", "混合"])
    final_method = method_category
    if "MCDM" in method_category:
        mcdm_tools = st.multiselect("工具：", 
            ["Delphi", "Fuzzy Delphi", "AHP", "Fuzzy AHP", "ANP", "FCM (模糊認知圖)", "TOPSIS"],
            default=["Delphi", "FCM (模糊認知圖)"]
        )
        final_method = f"MCDM ({' + '.join(mcdm_tools)})" if mcdm_tools else "MCDM"

    # 格式
    paper_type = st.radio("類型", ["學位論文", "期刊論文"])
    if paper_type == "學位論文":
        CHAPTERS = [
            {"key": "ch1", "name": "第一章 緒論"},
            {"key": "ch2", "name": "第二章 文獻探討"},
            {"key": "ch3", "name": "第三章 研究方法"},
            {"key": "ch4", "name": "第四章 分析結果"},
            {"key": "ch5", "name": "第五章 結論"}
        ]
    else:
        CHAPTERS = [{"key": f"ch{i}", "name": n} for i, n in enumerate(["前言", "文獻", "方法", "結果", "結論"], 1)]
    
    # 規則
    if 'global_rules' not in st.session_state: 
        st.session_state.global_rules = "1. 必須使用繁體中文\n2. 數學公式與模型必須完整\n3. 數據結果必須引用文獻佐證"
    rules = st.text_area("寫作規則", value=st.session_state.global_rules, height=100)
    st.session_state.global_rules = rules

# --- 核心函數：呼叫 AI ---
def call_ai_api(prompt, sys_role="你是一位學術專家。"):
    if not api_key:
        return "⚠️ 請在左側輸入 API Key 才能開始運作。"

    try:
        if engine_choice == "Groq (Llama 3)":
            client = Groq(api_key=api_key)
            completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": sys_role},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.5,
                max_tokens=4000,
            )
            return completion.choices[0].message.content

        elif engine_choice == "Google (Gemini)":
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(
                f"{sys_role}\n\n{prompt}",
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=4000,
                    temperature=0.5
                )
            )
            return response.text

    except Exception as e:
        error_msg = str(e)
        if "413" in error_msg:
            return "❌ 錯誤 413：內容太長！請使用下方的「分批解析」。"
        elif "429" in error_msg:
            return "❌ 錯誤 429：額度已滿，請更換 Key。"
        else:
            return f"❌ 連線錯誤: {error_msg}"

# --- 核心函數：智慧分批處理 ---
def smart_batch_process(long_text, method_name):
    if not api_key: return "⚠️ 請先輸入 API Key"
    
    chunk_size = 3000
    chunks = [long_text[i:i+chunk_size] for i in range(0, len(long_text), chunk_size)]
    total_chunks = len(chunks)
    
    progress_bar = st.progress(0, text="準備開始分批閱讀...")
    combined_notes = ""
    
    for i, chunk in enumerate(chunks):
        progress_bar.progress((i / total_chunks) * 0.8, text=f"正在研讀第 {i+1}/{total_chunks} 部分...")
        prompt = f"這是一份文獻回顧的一部分。請提取：1.學者年份 2.變數 3.與{method_name}的關聯。\n文獻片段：\n{chunk}"
        note = call_ai_api(prompt, sys_role="你是一位速讀助理。")
        if "❌" in note: return note
        combined_notes += f"\n\n--- Part {i+1} ---\n{note}"
        time.sleep(1)
        
    progress_bar.progress(0.9, text="正在統整...")
    final_prompt = f"請將這些筆記整合成完整的學術文獻回顧表(Markdown)：\n{combined_notes}"
    final_result = call_ai_api(final_prompt, sys_role="你是一位博學的教授。")
    progress_bar.progress(1.0, text="完成！")
    return final_result

# --- 核心函數：生成 FCM 圖表數據 ---
def generate_fcm_data():
    data = {
        '時間週期': ['t=0 (初始)', 't=1 (投入)', 't=2 (轉化)', 't=3 (產出)', 't=4 (穩定)'],
        'C1 經費投入': [0.20, 0.90, 0.90, 0.90, 0.90],
        'C7 員工生產力': [0.50, 0.50, 0.60, 0.90, 1.00],
        'C9 離職率': [0.70, 0.70, 0.55, 0.20, 0.05]
    }
    return pd.DataFrame(data).set_index('時間週期')

# --- 初始化 ---
if 'step' not in st.session_state: st.session_state.step = 0
if 'final_title' not in st.session_state: st.session_state.final_title = ""
if 'refs' not in st.session_state: st.session_state.refs = ""
if 'parsed
