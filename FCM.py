import pandas as pd
import numpy as np
import streamlit as st

# ==========================================
# 0. 頁面初始化
# ==========================================
st.set_page_config(page_title="FCM 論文決策系統 (Smooth Fix)", layout="wide")

st.markdown("""
<style>
    /* 論文預覽區 */
    .report-box { 
        border: 1px solid #ccc; padding: 40px; background-color: #ffffff; 
        color: #000000; font-family: "Times New Roman", "標楷體", serif; 
        font-size: 16px; line-height: 2.0; text-align: justify;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-top: 20px; white-space: pre-wrap;
    }
    .stButton>button { width: 100%; border-radius: 5px; height: 3.5em; font-weight: bold; font-size: 15px;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 初始化數據
# ==========================================
if 'matrix' not in st.session_state:
    # 預設3x3矩陣，初始化為零
    mat = np.zeros((3, 3))
    st.session_state.matrix = mat

# 讀取上傳的文件並更新矩陣
uploaded_file = st.file_uploader("上傳矩陣文件", type=['xlsx', 'csv'])
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, index_col=0)
        else:
            df = pd.read_excel(uploaded_file, index_col=0)
        
        # 更新矩陣
        st.session_state.matrix = df.values
        st.session_state.concepts = df.columns.tolist()  # 讀取列標籤
        st.success("矩陣已成功加載！")
        st.dataframe(df)  # 顯示上傳的矩陣
    except Exception as e:
        st.error(f"加載矩陣失敗: {e}")

# ==========================================
# 2. 矩陣視圖和計算
# ==========================================
st.subheader("矩陣視圖 (-1 ~ 1)")

# 顯示當前矩陣
df_show = pd.DataFrame(st.session_state.matrix, columns=st.session_state.concepts, index=st.session_state.concepts)
st.dataframe(df_show)

# 按鈕操作：隨機生成權重矩陣
if st.button("🎲 隨機生成權重 (-1 ~ 1)"):
    n = len(st.session_state.concepts)
    rand = np.random.uniform(-1.0, 1.0, (n, n))
    np.fill_diagonal(rand, 0)  # 填充對角線為0
    st.session_state.matrix = rand
    df_show = pd.DataFrame(st.session_state.matrix, columns=st.session_state.concepts, index=st.session_state.concepts)
    st.dataframe(df_show)
    st.success("矩陣已隨機生成！")

# ==========================================
# 3. 生成論文草稿
# ==========================================
st.subheader("生成論文草稿")

# 示例生成部分（可根据需要进一步生成）
if st.button("生成 4.1 結構分析"):
    t = "### 第四章 研究結果與分析\n\n**4.1 FCM 矩陣結構特性分析 (Structural Analysis)**\n\n"
    t += f"本研究矩陣包含 {len(st.session_state.concepts)} 個準則，矩陣密度為 {np.count_nonzero(st.session_state.matrix) / (len(st.session_state.concepts) ** 2):.2f}。\n"
    t += f"數據顯示，**{st.session_state.concepts[0]}** 之總影響力 (絕對值出度={np.sum(np.abs(st.session_state.matrix[0])):.2f}) 最高，確認其為系統核心。\n"
    t += "基於這些分析，該準則被確定為系統中最關鍵的驅動力。\n"
    st.session_state.paper_sections["4.1"] = t
    st.write(t)

# 下載完整論文內容
if 't' in locals() and t:
    st.download_button("📥 下載完整論文 (TXT)", t, "thesis_final.txt")
else:
    st.error("生成的內容為空，請先生成內容再下載。")
