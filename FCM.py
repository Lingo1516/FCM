import streamlit as st
from pptx import Presentation
from pptx.util import Pt
from io import BytesIO  # 這是關鍵：讓我們在記憶體中處理檔案

def create_ppt_in_memory():
    prs = Presentation()

    # 輔助函式：新增投影片
    def add_slide(title, content_list):
        slide_layout = prs.slide_layouts[1] 
        slide = prs.slides.add_slide(slide_layout)
        slide.shapes.title.text = title
        body_shape = slide.placeholders[1]
        tf = body_shape.text_frame
        for item in content_list:
            p = tf.add_paragraph()
            p.text = item
            p.font.size = Pt(20)
            p.space_after = Pt(10)

    # --- 投影片內容 (精簡版範例) ---
    # 封面
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = "管理組織的生存環境"
    slide.placeholders[1].text = "Chapter 06: 生態與體制理論"

    # 內容頁範例
    add_slide("組織生態理論重點", [
        "核心觀點：自然選擇 (Natural Selection)",
        "環境負載力：池塘能養多少魚有上限",
        "結構慣性：大象難跳舞，組織難轉型"
    ])
    
    add_slide("體制理論重點", [
        "同形化：為何大家長得越來越像？",
        "法規性：不得不做 (政府規定)",
        "規範性：應該要做 (職業道德)",
        "認知性：大家都做 (模仿成功者)"
    ])

    # --- 關鍵修改：存入記憶體而非硬碟 ---
    binary_output = BytesIO()
    prs.save(binary_output)
    binary_output.seek(0) # 指針回到開頭
    return binary_output

# --- Streamlit 介面 ---
st.title("投影片生成器 📊")
st.write("點擊下方按鈕，將「組織生態與體制理論」課程內容轉為 PPT。")

# 產生檔案
if st.button('🚀 開始生成 PPT'):
    ppt_file = create_ppt_in_memory()
    
    # 下載按鈕
    st.download_button(
        label="📥 點此下載 PPTX 檔案",
        data=ppt_file,
        file_name="Ch06_組織生態與體制理論.pptx",
        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
    )
    st.success("生成完畢！請點擊上方按鈕下載。")
