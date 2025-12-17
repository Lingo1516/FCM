import streamlit as st
import pandas as pd
import string
import re
from collections import Counter
from io import BytesIO

# --- 頁面設定 ---
st.set_page_config(page_title="萬能文獻分析 (容錯版)", layout="wide", page_icon="🛡️")

st.title("🛡️ 萬能文獻分析器 (容錯 + 自動統計)")
st.markdown("### 只要貼上文字並換行，系統自動統計關鍵字並製表")

# --- 1. 輸入區 ---
st.info("👇 請貼上文獻資料。**重要：每一篇不同的文獻，請記得按 Enter 換行！**")
raw_text = st.text_area("資料輸入區", height=250, placeholder="第一篇文獻內容...\n第二篇文獻內容...\n(不需要年份，只要換行就好)")

# --- 2. 核心：寬鬆切割邏輯 (解決找不到年份的問題) ---
def loose_parse(text):
    # 直接用「換行符號」來切割
    lines = text.strip().split('\n')
    literature_list = []
    
    for line in lines:
        line = line.strip()
        if len(line) < 4: continue # 過濾掉太短的空行
        
        # 自動給一個代號名稱 (取前15個字)
        name_guess = line[:15] + "..." if len(line) > 15 else line
        
        literature_list.append({
            "author": name_guess, 
            "abstract": line
        })
    return literature_list

# --- 3. 核心：自動抓關鍵字 (免安裝版) ---
def simple_keyword_extraction(text_list, top_n=10):
    # 把所有內容接在一起分析
    full_text = " ".join([item['abstract'] for item in text_list])
    
    # 只留中文
    chinese_only = re.sub(r'[^\u4e00-\u9fa5]', '', full_text)
    
    words = []
    # 排除廢話
    stopwords = set(['研究', '探討', '分析', '結果', '顯示', '發現', '提出', '認為', '使用', '進行', '影響', '我們', '這些', '不同', '以及', '透過', '對於', '文獻', '本文', '摘要', '方法'])

    # 簡單的 N-gram (抓 2~3 個字)
    for i in range(len(chinese_only)):
        if i + 2 <= len(chinese_only):
            w = chinese_only[i:i+2]
            if w not in stopwords: words.append(w)
        if i + 3 <= len(chinese_only):
            w = chinese_only[i:i+3]
            if w not in stopwords: words.append(w)
            
    # 統計頻率
    counter = Counter(words)
    return [w for w, c in counter.most_common(top_n)]

# --- 4. 執行與顯示 ---
if st.button("🚀 開始分析", type="primary"):
    if not raw_text:
        st.warning("沒資料無法分析，請先貼上文字。")
    else:
        # A. 切割
        lit_data = loose_parse(raw_text)
        
        if not lit_data:
            st.error("無法切割資料，請確認你有按 Enter 換行。")
        else:
            st.success(f"✅ 成功辨識 {len(lit_data)} 行資料！")
            
            # B. 自動抓關鍵字
            auto_keywords = simple_keyword_extraction(lit_data)
            
            # 讓使用者篩選關鍵字
            st.subheader("1. 確認分析準則")
            final_keywords = st.multiselect(
                "系統統計出最常出現的詞 (可手動增刪)：",
                options=auto_keywords,
                default=auto_keywords
            )
            
            # 手動增加功能
            manual_add = st.text_input("想手動增加關鍵字？(用空白隔開)", placeholder="例如：ESG 獲利能力")
            if manual_add:
                final_keywords.extend(manual_add.split())

            if final_keywords:
                # C. 製表
                matrix = {}
                labels = []
                full_names = []
                
                # 代號生成器
                def get_label(index):
                    if index < 26: return string.ascii_uppercase[index]
                    else: return f"{string.ascii_uppercase[index // 26 - 1]}{string.ascii_uppercase[index % 26]}"

                for i, item in enumerate(lit_data):
                    label = get_label(i)
                    labels.append(label)
                    full_names.append(item['author'])
                    
                    col_res = []
                    for kw in final_keywords:
                        if kw in item['abstract']:
                            col_res.append("○")
                        else:
                            col_res.append("")
                    matrix[label] = col_res
                
                # D. 顯示結果
                df_matrix = pd.DataFrame(matrix, index=final_keywords)
                df_legend = pd.DataFrame({"代號": labels, "對應內容": full_names})
                
                st.divider()
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("📊 分析矩陣")
                    st.dataframe(df_matrix, use_container_width=True)
                with col2:
                    st.subheader("📝 對照表")
                    st.dataframe(df_legend, hide_index=True, use_container_width=True)
                
                # E. 下載 (包含防呆機制)
                output = BytesIO()
                download_ready = False
                file_name = "matrix.csv"
                mime_type = "text/csv"
                
                try:
                    # 優先嘗試存成 Excel (最漂亮)
                    import xlsxwriter
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df_matrix.to_excel(writer, sheet_name='矩陣')
                        df_legend.to_excel(writer, sheet_name='對照表')
                    file_name = "analysis.xlsx"
                    mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    download_ready = True
                except ImportError:
                    # 如果沒裝 xlsxwriter，改存 CSV 避免當機
                    st.warning("⚠️ 偵測到未安裝 xlsxwriter，將改為下載 CSV 格式。")
                    output = BytesIO()
                    df_matrix.to_csv(output, encoding='utf-8-sig')
                    download_ready = True

                if download_ready:
                    st.download_button(
                        label=f"📥 下載結果 ({file_name.split('.')[-1]})", 
                        data=output.getvalue(), 
                        file_name=file_name,
                        mime=mime_type
                    )
