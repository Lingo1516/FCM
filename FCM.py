import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from io import StringIO

# --- 1. 頁面設定 ---
st.set_page_config(page_title="超級 Python 工作台", layout="wide", page_icon="💻")

# --- CSS 美化 (讓介面看起來更專業) ---
st.markdown("""
<style>
    .reportview-container { margin-top: -2em; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stTextArea textarea { font-family: 'Consolas', 'Courier New', monospace; background-color: #f0f2f6; }
</style>
""", unsafe_allow_html=True)

# --- 2. 側邊欄：檔案上傳區 ---
with st.sidebar:
    st.header("📂 檔案上傳區")
    st.markdown("上傳 CSV 或 Excel，變數名稱會自動設為 `df`")
    uploaded_file = st.file_uploader("選擇檔案", type=["csv", "xlsx"])
    
    df = None
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.success(f"✅ 成功讀取: {uploaded_file.name}")
            st.write("資料預覽 (前 5 筆):")
            st.dataframe(df.head())
            st.info("💡 在右邊程式碼中，直接使用變數 `df` 即可操作此資料！")
        except Exception as e:
            st.error(f"檔案讀取失敗: {e}")

    st.markdown("---")
    st.markdown("### 📝 常用指令小抄")
    st.code("st.write(data) # 顯示文字或變數", language="python")
    st.code("st.dataframe(df) # 顯示表格", language="python")
    st.code("st.bar_chart(data) # 快速長條圖", language="python")
    st.code("st.pyplot(fig) # 顯示 Matplotlib 圖", language="python")

# --- 3. 主畫面區 ---
st.title("🚀 超級 Python 線上工作台")
st.markdown("### 👉 在這裡輸入程式碼，把這裡當作你的畫布")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("💻 程式碼輸入")
    
    # 預設程式碼 (根據是否有上傳檔案給不同範例)
    if df is not None:
        default_code = """# 範例：分析上傳的資料
st.write("📊 資料統計摘要：")
st.write(df.describe())

st.write("📈 畫個簡單的圖：")
# 假設資料全是數值，直接畫圖 (你可以修改欄位)
st.line_chart(df.select_dtypes(include=['number']))
"""
    else:
        default_code = """import numpy as np
import pandas as pd

# 1. 建立假資料
st.write("正在產生隨機資料...")
data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['A', 'B', 'C']
)

# 2. 顯示表格
st.subheader("我的資料表")
st.dataframe(data)

# 3. 畫圖
st.subheader("折線圖分析")
st.line_chart(data)
"""

    code_input = st.text_area("Python Code", value=default_code, height=500)
    run_btn = st.button("▶️ 執行程式 (Run)", type="primary")

with col2:
    st.subheader("🖥️ 執行結果")
    
    # 這裡是用來捕捉輸出的容器
    output_container = st.container()

    if run_btn:
        with output_container:
            # 重新導向 stdout 以捕捉 print 的內容
            old_stdout = sys.stdout
            redirected_output = sys.stdout = StringIO()

            # 建立執行環境的變數字典 (讓 exec 認識 st, pd, plt, df)
            local_env = {
                "st": st,
                "pd": pd,
                "plt": plt,
                "sns": sns,
                "df": df  # 如果有上傳檔案，這裡會有 df
            }

            try:
                # --- 核心執行區 ---
                exec(code_input, {}, local_env)
                # ------------------
                
                # 顯示 print() 的內容
                printed_text = redirected_output.getvalue()
                if printed_text:
                    st.text("📝 文字輸出 (Terminal Output):")
                    st.code(printed_text)
                
            except Exception as e:
                st.error("❌ 程式發生錯誤 (Error):")
                st.exception(e)
            finally:
                sys.stdout = old_stdout
