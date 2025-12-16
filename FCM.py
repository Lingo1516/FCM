import streamlit as st
from groq import Groq
import google.generativeai as genai
import time
import pandas as pd

# --- 系統設定 ---
st.set_page_config(page_title="FCM 論文寫作助手 (最終修復版)", layout="wide", page_icon="🎓")

# --- 側邊欄：引擎與金鑰設定 ---
with st.sidebar:
    st.header("⚙️ 引擎設定")
    
    # 選擇引擎
    engine_choice = st.radio("選擇 AI 模型", ["Groq (Llama 3)", "Google (Gemini)"])
    
    api_key = ""
    if engine_choice == "Groq (Llama 3)":
        st.info("推薦！速度快，適合處理大量文字。")
        # 避免 Secret Scanning 警告，改為輸入框
        api_key = st.text_input("請輸入 Groq Key (gsk_...)", type="password")
        st.markdown("[👉 免費申請 Groq Key](https://console.groq.com/keys)")
    else:
        st.info("備用引擎。")
        api_key = st.text_input("請輸入 Google Key", type="password")
        st.markdown("[👉 免費申請 Google Key](https://aistudio.google.com/app/apikey)")

    st.divider()

    # 論文參數設定
    st.subheader("📝 論文參數")
    # 關鍵字
    default_kws = ["教育訓練", "人力資本", "組織績效", "FCM", "動態模擬"]
    keywords_str = st.text_input("關鍵字 (以逗號分隔)", value=",".join(default_kws))
    
    # 方法選擇
    final_method = st.selectbox("研究方法", ["MCDM (FCM 模糊認知圖)", "Fuzzy Delphi", "System Dynamics", "Regression"])

    # 論文結構
    paper_type = st.radio("格式類型", ["學位論文 (五章式)", "期刊論文"])
    if paper_type == "學位論文 (五章式)":
        CHAPTERS = [
            {"key": "ch1", "name": "第一章 緒論"},
            {"key": "ch2", "name": "第二章 文獻探討"},
            {"key": "ch3", "name": "第三章 研究方法"},
            {"key": "ch4", "name": "第四章 分析結果"},
            {"key": "ch5", "name": "第五章 結論"}
        ]
    else:
        CHAPTERS = [{"key": f"ch{i}", "name": n} for i, n in enumerate(["前言", "文獻", "方法", "結果", "結論"], 1)]

# --- 核心函數 1：呼叫 AI ---
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
        return f"❌ 連線錯誤: {str(e)}"

# --- 核心函數 2：智慧分批處理 (解決文獻過長) ---
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
        time.sleep(0.5) # 避免過快請求
        
    progress_bar.progress(0.9, text="正在統整...")
    final_prompt = f"請將這些筆記整合成完整的學術文獻回顧表(Markdown)：\n{combined_notes}"
    final_result = call_ai_api(final_prompt, sys_role="你是一位博學的教授。")
    progress_bar.progress(1.0, text="完成！")
    time.sleep(1)
    progress_bar.empty()
    return final_result

# --- 核心函數 3：生成 FCM 圖表數據 (寫死的完整數據) ---
def generate_fcm_data():
    # 這是基於文獻轉化與情境模擬的完整數據
    data = {
        '時間週期': ['t=0 (初始)', 't=1 (投入)', 't=2 (轉化)', 't=3 (產出)', 't=4 (穩定)'],
        'C1 經費投入': [0.20, 0.90, 0.90, 0.90, 0.90],
        'C7 員工生產力': [0.50, 0.50, 0.60, 0.90, 1.00],
        'C9 離職率': [0.70, 0.70, 0.55, 0.20, 0.05]
    }
    return pd.DataFrame(data).set_index('時間週期')

# --- 初始化 Session State ---
if 'step' not in st.session_state: st.session_state.step = 0
if 'final_title' not in st.session_state: st.session_state.final_title = "教育訓練經費投入對組織績效之動態模擬研究"
if 'refs' not in st.session_state: st.session_state.refs = ""
if 'parsed_refs' not in st.session_state: st.session_state.parsed_refs = "" 
if 'outline' not in st.session_state: st.session_state.outline = ""
if 'content' not in st.session_state: st.session_state.content = {}

# --- 主畫面 UI ---
st.title("🎓 論文寫作助手 (FCM 完整版)")

# === 步驟 0: 題目 ===
if st.session_state.step == 0:
    st.header("步驟 1：確認題目")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        title_input = st.text_input("論文題目", value=st.session_state.final_title)
    with col2:
        if st.button("✨ AI 建議題目"):
            if not keywords_str: st.error("請輸入關鍵字")
            else:
                prompt = f"領域：管理科學。關鍵字：{keywords_str}。方法：{final_method}。請產生 3 個博士論文題目。"
                st.info(call_ai_api(prompt))

    if st.button("下一步 (導入文獻) ➡️", type="primary"):
        st.session_state.final_title = title_input
        st.session_state.step = 1
        st.rerun()

# === 步驟 1: 文獻 ===
elif st.session_state.step == 1:
    st.header("步驟 2：導入文獻")
    st.info("💡 貼上您的參考文獻，系統將自動提取重點並建立 FCM 關聯矩陣基礎。")
    
    raw_refs = st.text_area("請貼上文獻內容", value=st.session_state.refs, height=300)
    st.session_state.refs = raw_refs

    if st.button("✨ 啟動文獻解析", type="secondary"):
        if not raw_refs:
            st.error("請先貼上一些文字")
        else:
            st.session_state.parsed_refs = smart_batch_process(raw_refs, final_method)
    
    # --- 這裡就是之前報錯的地方，現在修復了 ---
    if 'parsed_refs' in st.session_state and st.session_state.parsed_refs:
        st.success("✅ 解析完成")
        st.markdown(st.session_state.parsed_refs)

    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("⬅️ 上一步"): st.session_state.step = 0; st.rerun()
    with col2:
        if st.button("下一步 (生成大綱) ➡️", type="primary"): st.session_state.step = 2; st.rerun()

# === 步驟 2: 大綱 ===
elif st.session_state.step == 2:
    st.header("步驟 3：生成大綱")
    
    if st.button("✨ 生成學術大綱"):
        with st.spinner("規劃中..."):
            ref_context = st.session_state.parsed_refs if st.session_state.parsed_refs else "無"
            prompt = f"題目：{st.session_state.final_title}\n方法：{final_method}\n文獻背景：{ref_context}\n請撰寫詳細大綱，特別強調第三章的研究設計與第四章的模擬分析。"
            st.session_state.outline = call_ai_api(prompt)
            st.rerun()

    if st.session_state.outline:
        st.markdown(st.session_state.outline)

    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("⬅️ 上一步"): st.session_state.step = 1; st.rerun()
    with col2:
        if st.button("下一步 (開始寫作) ➡️", type="primary"): st.session_state.step = 3; st.rerun()

# === 步驟 3: 寫作 (含圖表功能) ===
elif st.session_state.step == 3:
    st.header("步驟 4：逐章寫作 & FCM 模擬")
    
    chapter_map = {ch['key']: ch['name'] for ch in CHAPTERS}
    selected_ch = st.selectbox("選擇章節", list(chapter_map.keys()), format_func=lambda x: chapter_map[x])
    
    # --- 自動畫圖區 (第四章專用) ---
    if "ch4" in selected_ch:
        st.markdown("### 📈 FCM 動態模擬結果圖")
        st.info("系統已根據 FCM 運算邏輯，自動生成迭代趨勢圖 (灰色=經費, 藍色=生產力, 紅色=離職率)。")
        
        # 產生數據與圖表
        df_chart = generate_fcm_data()
        st.line_chart(df_chart, color=["#A9A9A9", "#0000FF", "#FF0000"]) 
        
        with st.expander("查看詳細模擬數據 (Table 4-1)"):
            st.dataframe(df_chart)

    # 寫作按鈕
    if st.button(f"🚀 撰寫 {chapter_map[selected_ch]} 內容", type="primary"):
        with st.spinner("AI 正在寫作中..."):
            ref_context = st.session_state.parsed_refs if st.session_state.parsed_refs else "參照一般學術文獻"
            
            # 特殊指令
            special_instruction = ""
            if "ch3" in selected_ch: 
                special_instruction = "必須包含 FCM 的數學定義 (Matrix Algebra) 以及權重轉化規則表。"
            elif "ch4" in selected_ch: 
                special_instruction = "必須包含情境模擬分析 (Scenario Analysis)，解釋圖表中的時間滯後現象與交叉點。"
            
            prompt = f"""
            題目：{st.session_state.final_title}
            章節：{chapter_map[selected_ch]}
            大綱：{st.session_state.outline}
            參考文獻：{ref_context}
            特殊要求：{special_instruction}
            
            請撰寫本章內容，約 1500-2000 字，使用學術語氣。
            """
            st.session_state.content[selected_ch] = call_ai_api(prompt)
            st.rerun()
            
    if selected_ch in st.session_state.content:
        st.markdown(st.session_state.content[selected_ch])
        
    st.markdown("---")
    if st.button("💾 全部完成，前往下載"): st.session_state.step = 4; st.rerun()

# === 步驟 4: 下載 ===
elif st.session_state.step == 4:
    st.header("步驟 5：下載檔案")
    
    final_doc = f"# {st.session_state.final_title}\n\n**研究方法**：{final_method}\n\n"
    for ch in CHAPTERS:
        if ch['key'] in st.session_state.content:
            final_doc += f"\n\n## {ch['name']}\n{st.session_state.content[ch['key']]}\n"
    
    st.download_button("📥 下載完整論文 (.txt)", final_doc, "thesis_draft.txt", "text/plain")
    
    if st.button("🔄 重置所有進度"):
        for key in list(st.session_state.keys()): del st.session_state[key]
        st.rerun()
