import streamlit as st
import pandas as pd
import string
from io import BytesIO

# --- 嘗試匯入必要的套件 (防呆機制) ---
try:
    import google.generativeai as genai
    import xlsxwriter # 檢查是否已安裝
except ImportError as e:
    st.error(f"❌ 系統偵測到缺少套件：{e.name}")
    st.warning("⚠️ 請務必在 GitHub 專案中建立 `requirements.txt` 檔案，並填入必要套件名稱。")
    st.stop() # 停止執行，避免後面報一堆錯

# --- 1. 設定您的 API Key ---
# ⚠️ 請在下方引號內貼上你的 AIza 開頭金鑰
USER_API_KEY = "AIzaSyBlj24gBVr3RJhkukS9p6yo5s2-WVBH2H0" 

# 設定 Google Gemini
if USER_API_KEY and "AIza" in USER_API_KEY:
    genai.configure(api_key=USER_API_KEY)

# --- 2. 頁面設定 ---
st.set_page_config(page_title="AI 文獻分析器 (最終修復版)", layout="wide", page_icon="🛠️")
st.title("🛠️ AI 文獻分析器 (Gemini 1.5 Flash)")

# --- 3. 測試連線按鈕 ---
if st.button("📡 測試連線與版本"):
    if "AIza" not in USER_API_KEY:
        st.error("❌ 金鑰格式錯誤！請檢查第 16 行。")
    else:
        with st.spinner("正在檢查 Google連線..."):
            try:
                # 列出可用模型，確認帳號權限
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content("Hello")
                st.success(f"✅ 連線成功！Google 回應：{response.text}")
                st.caption(f"目前使用的套件版本：google-generativeai (最新版)")
            except Exception as e:
                st.error(f"❌ 連線失敗：{str(e)}")
                if "404" in str(e):
                    st.warning("💡 若出現 404 錯誤，代表您的 `requirements.txt` 沒有設定 `google-generativeai>=0.8.3`，請去更新檔案。")

# --- 4. 文獻輸入與處理 ---
st.info("👇 請貼上文獻資料 (每篇請換行)")
raw_text = st.text_area("文獻輸入區", height=200)

def get_ai_analysis(text):
    # 使用目前最穩定的 Flash 模型
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""
    任務：歸納 10 個學術研究構面關鍵字。
    規則：只列出名詞，用頓號隔開。排除無關詞彙(如日期、下午)。
    內容：{text[:8000]}
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# 切割文字邏輯
def parse_text(text):
    lines = text.strip().split('\n')
    return [{"title": line[:15], "content": line} for line in lines if len(line) > 5]

# --- 5. 執行分析 ---
if st.button("🚀 開始分析", type="primary"):
    if not raw_text:
        st.warning("請先貼上資料！")
    else:
        with st.spinner("🤖 AI 正在閱讀與分析中..."):
            lit_data = parse_text(raw_text)
            ai_result = get_ai_analysis(raw_text)
            
            if "Error" in ai_result:
                st.error(f"分析失敗：{ai_result}")
            else:
                st.success("✅ 分析完成！")
                
                # 處理關鍵字
                keywords = [k.strip() for k in ai_result.replace("\n", "、").split("、") if k.strip()]
                final_keywords = st.multiselect("AI 抓到的準則", options=keywords, default=keywords)
                
                if final_keywords:
                    # 建表
                    matrix = {}
                    labels = []
                    titles = []
                    for i, item in enumerate(lit_data):
                        lbl = string.ascii_uppercase[i % 26]
                        labels.append(lbl)
                        titles.append(item['title'])
                        col_res = []
                        for kw in final_keywords:
                            if kw in item['content']: col_res.append("○")
                            else: col_res.append("")
                        matrix[lbl] = col_res
                    
                    # 顯示
                    df = pd.DataFrame(matrix, index=final_keywords)
                    df_legend = pd.DataFrame({"代號": labels, "對應文獻": titles})
                    
                    c1, c2 = st.columns([2, 1])
                    with c1: st.dataframe(df, use_container_width=True)
                    with c2: st.dataframe(df_legend, hide_index=True)
                    
                    # 下載 Excel
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df.to_excel(writer, sheet_name='矩陣')
                        df_legend.to_excel(writer, sheet_name='對照表')
                    st.download_button("📥 下載 Excel", output.getvalue(), "analysis.xlsx")
