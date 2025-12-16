import streamlit as st
import sys
from io import StringIO

# 設定網頁標題
st.set_page_config(page_title="我的 Python 執行器", layout="wide")

st.title("🐍 Python 線上執行沙盒")
st.markdown("把你的 Python 程式碼貼在下面，按下執行即可查看結果。")

# 左邊是輸入區，右邊是輸出區
col1, col2 = st.columns(2)

with col1:
    st.subheader("輸入程式碼")
    # 預設一些範例程式碼
    default_code = "print('Hello, World!')\nfor i in range(5):\n    print(f'Counting: {i}')"
    code_input = st.text_area("Code Area", value=default_code, height=400)
    run_button = st.button("🚀 執行程式碼", type="primary")

with col2:
    st.subheader("執行結果")
    output_container = st.empty()

    if run_button:
        # 這是為了攔截 print() 的輸出結果
        old_stdout = sys.stdout
        redirected_output = sys.stdout = StringIO()

        try:
            # 警告：exec() 有資安風險，僅建議在本地或受信任環境使用
            exec(code_input)
            result = redirected_output.getvalue()
            if result:
                st.code(result, language="text")
            else:
                st.info("程式執行成功，但沒有輸出 (No Output)。")
        except Exception as e:
            st.error(f"程式發生錯誤：\n{e}")
        finally:
            # 恢復標準輸出，避免影響後續程式
            sys.stdout = old_stdout
