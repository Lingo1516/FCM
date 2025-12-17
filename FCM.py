import streamlit as st
import pandas as pd
import re
import string
from io import BytesIO

# 嘗試匯入結巴斷詞 (中文分析專用)
try:
    import jieba
    import jieba.analyse
except ImportError:
    st.error("請先安裝 jieba 套件: pip install jieba")
    st.stop()

# --- 頁面設定 ---
st.set_page_config(page_title="AI 統計分析文獻工具", layout="wide", page_icon="🧮")

st.title("🧮 智慧統計文獻分析器 (演算法版)")
st.markdown("""
### 原理說明
這個版本不使用預設字庫，也不需要 AI Key。
它使用 **TF-IDF 演算法**，現場計算您貼上的文字中，哪些詞彙的**權重最高**，自動抓出來當作分析準則。
""")

# --- 1. 輸入區 ---
st.info("💡 請貼上您的文獻資料，系統會自動算出最常出現的關鍵字。")
raw_text = st.text_area("文獻資料輸入區：", height=300, placeholder="直接把整篇論文摘要或筆記貼進來...")

# --- 2. 核心邏輯 ---

def analyze_data(text):
    # 步驟 A: 切割文獻 (抓年份)
    pattern = r'([^\n\r。]+?[\(\[\{](?:19|20)\d{2}[\)\]\}])'
    segments = re.split(pattern, text)
    
    literature_list = []
    current_author = None
    all_abstracts_text = "" # 用來給演算法分析的大池子
    
    for segment in segments:
        segment = segment.strip()
        if not segment: continue
        
        # 判斷是否為作者 (包含年份)
        if re.search(r'[\(\[\{](?:19|20)\d{2}[\)\]\}]', segment):
            current_author = segment[-50:] # 截斷過長的誤判
        else:
            # 這是摘要內容
            if current_author:
                literature_list.append({"author": current_author, "abstract": segment})
                all_abstracts_text += segment + "\n" # 累積所有摘要文字

    # 步驟 B: 使用 Jieba 演算法抓關鍵字
    if not all_abstracts_text:
        return None, None

    # 使用 extract_tags (基於 TF-IDF 演算法) 抓取前 20 個關鍵詞
    # allowPOS 指定詞性：n=名詞, v=動詞, vn=名動詞 (過濾掉 '的', '是' 這種廢話)
    keywords = jieba.analyse.extract_tags(all_abstracts_text, topK=20, allowPOS=('n', 'vn', 'v'))
    
    return literature_list, keywords

# --- 3. 操作介面 ---

if st.button("🚀 開始運算與分析", type="primary"):
    if not raw_text:
        st.warning("請先貼上資料！")
    else:
        with st.spinner("正在進行斷詞與權重計算..."):
            lit_data, auto_keywords = analyze_data(raw_text)
        
        if not lit_data:
            st.error("無法辨識文獻格式，請確認內容包含年份 (例如: 2023)。")
        else:
            # --- 顯示自動抓到的關鍵字 ---
            st.success(f"✅ 分析完成！演算法算出這篇文章最重要的 {len(auto_keywords)} 個詞：")
            
            # 讓使用者可以刪減
            final_keywords = st.multiselect(
                "系統自動抓到的關鍵字 (您可以刪除不喜歡的)",
                options=auto_keywords,
                default=auto_keywords
            )
            
            if not final_keywords:
                st.warning("請至少保留一個關鍵字以生成圖表。")
            else:
                # --- 生成矩陣 ---
                matrix = {}
                labels = []
                authors = []
                
                # 生成代號
                def get_label(index):
                    if index < 26: return string.ascii_uppercase[index]
                    else: return f"{string.ascii_uppercase[index // 26 - 1]}{string.ascii_uppercase[index % 26]}"

                for i, item in enumerate(lit_data):
                    label = get_label(i)
                    labels.append(label)
                    authors.append(item['author'])
                    
                    col_res = []
                    for kw in final_keywords:
                        if kw in item['abstract']:
                            col_res.append("●")
                        else:
                            col_res.append("")
                    matrix[label] = col_res
                
                # --- 顯示表格 ---
                df_matrix = pd.DataFrame(matrix, index=final_keywords)
                df_legend = pd.DataFrame({"代號": labels, "作者": authors})
                
                st.divider()
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("📊 關鍵字分析矩陣")
                    st.dataframe(df_matrix, use_container_width=True, height=500)
                
                with col2:
                    st.subheader("📝 文獻對照表")
                    st.dataframe(df_legend, hide_index=True, use_container_width=True)
                
                # --- 下載 ---
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df_matrix.to_excel(writer, sheet_name='矩陣分析')
                    df_legend.to_excel(writer, sheet_name='文獻對照')
                
                st.download_button("📥 下載 Excel 報表", output.getvalue(), "analysis_report.xlsx")
