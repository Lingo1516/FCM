import streamlit as st
import pandas as pd
import time

# 嘗試匯入 AI 庫，如果使用者沒有安裝也不會直接報錯，只是 API 功能不能用
try:
    from groq import Groq
    import google.generativeai as genai
except ImportError:
    st.error("請先安裝相關套件: pip install groq google-generativeai pandas streamlit")

# --- 1. 系統設定 ---
st.set_page_config(page_title="論文寫作助手 (含FCM動態模擬圖)", layout="wide", page_icon="📊")

# --- 2. 初始化 Session State (確保變數存在) ---
if 'step' not in st.session_state: st.session_state.step = 0
if 'final_title' not in st.session_state: st.session_state.final_title = "FCM 動態模擬研究"
if 'content' not in st.session_state: st.session_state.content = {}

# --- 3. 側邊欄：引擎與金鑰設定 ---
with st.sidebar:
    st.header("⚙️ 引擎設定")
    engine_choice = st.radio("選擇 AI 模型", ["Groq (Llama 3)", "Google (Gemini)"])
    
    api_key = ""
    if engine_choice == "Groq (Llama 3)":
        api_key = st.text_input("Groq Key", type="password", help="輸入您的 Groq API Key")
    else:
        api_key = st.text_input("Google Key", type="password", help="輸入您的 Gemini API Key")
    st.divider()

# --- 4. 核心函數：呼叫 AI ---
def call_ai_api(prompt, sys_role="你是一位學術專家。"):
    if not api_key: 
        # 如果沒輸入 Key，回傳模擬文字方便測試 UI
        time.sleep(1)
        return "⚠️ (模擬回應) 請先在側邊欄輸入 API Key 才能產生真實內容。\n\n這是模擬的段落內容..."
    
    try:
        if engine_choice == "Groq (Llama 3)":
            client = Groq(api_key=api_key)
            completion = client.chat.completions.create(
                messages=[{"role": "system", "content": sys_role}, {"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile", temperature=0.5, max_tokens=4000,
            )
            return completion.choices[0].message.content
        elif engine_choice == "Google (Gemini)":
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(f"{sys_role}\n\n{prompt}")
            return response.text
    except Exception as e:
        return f"❌ 錯誤: {str(e)}"

# --- 5. 核心函數：生成 FCM 迭代數據 ---
def generate_fcm_data():
    """
    產生模擬用的 DataFrame 數據
    """
    data = {
        # '時間週期' 會被設為 X 軸
        '時間週期': ['t=0 (初始)', 't=1 (投入)', 't=2 (轉化)', 't=3 (產出)', 't=4 (穩定)'],
        'C1 經費投入': [0.20, 0.90, 0.90, 0.90, 0.90],
        'C7 員工生產力': [0.50, 0.50, 0.60, 0.90, 1.00],
        'C9 離職率': [0.70, 0.70, 0.55, 0.20, 0.05]
    }
    # 將數據轉為 DataFrame 並設定索引，這對 st.line_chart 很重要
    df = pd.DataFrame(data)
    df = df.set_index('時間週期')
    return df

# --- 6. 主畫面邏輯 ---

st.title("📊 論文寫作助手 (含圖表生成)")

# === 步驟 0-2 (快速跳轉區) ===
if st.session_state.step < 3:
    st.info("👇 這是測試模式，點擊下方按鈕直接進入「第四章：分析結果」查看圖表功能")
    if st.button("🚀 直接跳轉至步驟 3 (測試圖表功能)"):
        st.session_state.step = 3
        st.rerun()

# === 步驟 3: 寫作與畫圖 ===
elif st.session_state.step == 3:
    st.header("步驟 4：逐章寫作 & 數據模擬")
    
    chapter_list = ["第一章 緒論", "第二章 文獻探討", "第三章 研究方法", "第四章 分析結果", "第五章 結論"]
    # 預設選中第四章，方便你直接看圖
    default_index = 3 
    selected_ch = st.selectbox("選擇章節", chapter_list, index=default_index)
    
    # --- 🔥 重點：自動畫圖區域 ---
    if "第四章" in selected_ch:
        st.markdown("### 📈 FCM 動態模擬結果圖")
        st.success("系統已自動生成「迭代趨勢圖」。使用 Streamlit 原生圖表，無需擔心字型亂碼。")
        
        # 1. 取得數據
        df_chart = generate_fcm_data()
        
        # 2. 畫出互動式折線圖
        # 這裡會自動讀取 DataFrame 的 columns 作為線條，index 作為 X 軸
        st.line_chart(df_chart, color=["#A9A9A9", "#0000FF", "#FF0000"]) 
        # 色碼對應: 灰色(C1), 藍色(C7), 紅色(C9) 
        # 注意：顏色順序是對應欄位字母順序或 DataFrame 欄位順序
        
        st.caption("圖 4-1：教育訓練經費投入後之各項指標動態變化趨勢")
        
        # 顯示詳細數據表
        with st.expander("點擊查看詳細數據表 (Table 4-1)"):
            st.dataframe(df_chart)

    # --- 寫作功能區 ---
    st.divider()
    if st.button(f"📝 讓 AI 撰寫 {selected_ch} 內容", type="primary"):
        with st.spinner("AI 正在思考與寫作中..."):
            prompt = f"請撰寫學術論文的 {selected_ch}，題目為：{st.session_state.final_title}。請包含相關的數據分析描述。"
            # 呼叫 API 並儲存結果
            result_text = call_ai_api(prompt)
            st.session_state.content[selected_ch] = result_text
            st.rerun()
            
    # 顯示 AI 寫好的內容
    if selected_ch in st.session_state.content:
        st.markdown("### 📄 草稿預覽")
        st.markdown(st.session_state.content[selected_ch])
        
    st.markdown("---")
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("⬅️ 返回"):
            st.session_state.step = 0
            st.rerun()
    with col2:
        if st.button("💾 全部完成，進入下載頁"):
            st.session_state.step = 4
            st.rerun()

# === 步驟 4: 下載 ===
elif st.session_state.step == 4:
    st.header("步驟 5：下載檔案")
    
    final_doc = f"# {st.session_state.final_title}\n\n"
    for ch in st.session_state.content:
        final_doc += f"\n\n## {ch}\n{st.session_state.content[ch]}\n"
    
    st.text_area("全文預覽", final_doc, height=300)
    
    st.download_button("📥 下載完整論文 (.txt)", final_doc, "thesis_draft.txt")
    
    if st.button("🔄 重頭來過"):
        st.session_state.clear()
        st.rerun()
