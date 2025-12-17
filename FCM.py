import streamlit as st
import pandas as pd
import google.generativeai as genai
import string
from io import BytesIO

# --- 1. 設定您的 API Key (已內建) ---
# ⚠️ 安全警告：這把鑰匙是您的私密資訊，請勿將此程式碼發布到公開網路 (GitHub/論壇)
USER_API_KEY = "AIzaSyBlj24gBVr3RJhkukS9p6yo5s2-WVBH2H0"

# 設定 Google Gemini
genai.configure(api_key=USER_API_KEY)

# --- 2. 頁面設定 ---
st.set_page_config(page_title="AI 文獻分析器 (自動版)", layout="wide", page_icon="🤖")

st.title("🤖 AI 智慧文獻分析器")
st.markdown("### 已內建金鑰，直接貼上文獻即可開始分析")

# --- 3. 輸入區 ---
st.info("👇 請將文獻資料貼在下方 (每一篇請記得 **按 Enter 換行**)")
raw_text = st.text_area("文獻輸入區", height=300, placeholder="直接把亂亂的文字貼進來...\n記得每一篇要換行...\n程式會自動幫你抓重點...")

# --- 4. 核心邏輯：呼叫 Google AI ---
def get_ai_analysis(text):
    # 使用免費快速的 Flash 模型
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
    你是一位學術研究專家。請閱讀以下文獻內容，幫我歸納出 10 到 15 個最重要的「研究構面」或「評估準則」。
    
    【嚴格規則】：
    1. 排除所有無關詞彙（如：日期、下午、報告、作者名、研究方法）。
    2. 只保留具備學術價值的名詞（例如：績效管理、數位轉型、供應鏈韌性、ESG、獲利能力）。
    3. 直接輸出名詞，用「、」頓號隔開。不要有任何開場白或結尾。
    
    【文獻內容】：
    {text[:10000]} 
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# --- 5. 輔助邏輯：切割文獻 ---
def parse_text(text):
    lines = text.strip().split('\n')
    literature_list = []
    
    for line in lines:
        line = line.strip()
        if len(line) < 5: continue
        
        # 自動取前 15 字當標題
        title = line[:15] + "..." if len(line) > 15 else line
        literature_list.append({"title": title, "content": line})
        
    return literature_list

# --- 6. 執行按鈕 ---
if st.button("🚀 開始分析", type="primary"):
    if not raw_text:
        st.warning("請先貼上資料！")
    else:
        with st.spinner("🤖 AI 正在閱讀您的文獻並歸納重點..."):
            # A. 切割資料
            lit_data = parse_text(raw_text)
            
            if not lit_data:
                st.error("無法切割資料，請確認每篇文獻有換行。")
            else:
                # B. 呼叫 AI
                ai_result = get_ai_analysis(raw_text)
                
                if "Error" in ai_result:
                    st.error(f"連線錯誤：{ai_result} (可能是額度已滿或 Key 被停用)")
                else:
                    st.success("✅ AI 分析完成！")
                    
                    # C. 整理關鍵字
                    keywords = [k.strip() for k in ai_result.replace("\n", "、").split("、") if k.strip()]
                    # 去除重複
                    keywords = list(dict.fromkeys(keywords))
                    
                    # D. 讓使用者篩選
                    st.subheader("1️⃣ AI 建議的準則 (可刪減)")
                    final_keywords = st.multiselect(
                        "請勾選要保留的準則：",
                        options=keywords,
                        default=keywords
                    )
                    
                    # 手動補充
                    manual_add = st.text_input("手動補充準則 (用空白隔開)：", placeholder="例如：創新能力 組織文化")
                    if manual_add:
                        final_keywords = manual_add.split() + final_keywords

                    if final_keywords:
                        # E. 建立矩陣
                        matrix = {}
                        labels = []
                        titles = []
                        
                        def get_label(idx):
                            if idx < 26: return string.ascii_uppercase[idx]
                            else: return f"{string.ascii_uppercase[idx // 26 - 1]}{string.ascii_uppercase[idx % 26]}"

                        for i, item in enumerate(lit_data):
                            label = get_label(i)
                            labels.append(label)
                            titles.append(item['title'])
                            
                            col_res = []
                            for kw in final_keywords:
                                if kw in item['content']:
                                    col_res.append("○")
                                else:
                                    col_res.append("")
                            matrix[label] = col_res
                        
                        # F. 顯示結果
                        df_matrix = pd.DataFrame(matrix, index=final_keywords)
                        df_legend = pd.DataFrame({"代號": labels, "對應文獻": titles})
                        
                        st.divider()
                        c1, c2 = st.columns([2, 1])
                        with c1:
                            st.subheader("📊 分析矩陣")
                            st.dataframe(df_matrix, use_container_width=True)
                        with c2:
                            st.subheader("📝 對照表")
                            st.dataframe(df_legend, hide_index=True, use_container_width=True)
                        
                        # G. 下載
                        output = BytesIO()
                        try:
                            import xlsxwriter
                            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                df_matrix.to_excel(writer, sheet_name='矩陣')
                                df_legend.to_excel(writer, sheet_name='對照表')
                            file_name = "ai_analysis.xlsx"
                            mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        except ImportError:
                            df_matrix.to_csv(output, encoding='utf-8-sig')
                            file_name = "ai_analysis.csv"
                            mime = "text/csv"
                            
                        st.download_button(f"📥 下載報表 ({file_name})", output.getvalue(), file_name, mime, type="primary")
