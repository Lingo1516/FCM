from flask import Flask, render_template_string
import pandas as pd

app = Flask(__name__)

# 用戶提供的文獻資料（包含文獻摘要）
literature_data = [
    {"author": "A", "abstract": "人力資源管理、績效管理、新資福利、員工安全、教育訓練"},
    {"author": "B", "abstract": "績效管理、新資福利、員工安全、教育訓練"},
    {"author": "C", "abstract": "績效管理、員工安全"},
    {"author": "D", "abstract": "新資福利、員工安全"},
    {"author": "E", "abstract": "績效管理、員工安全、教育訓練"},
    {"author": "F", "abstract": "人力資源管理、績效管理、教育訓練"},
    {"author": "G", "abstract": "人力資源管理、績效管理、新資福利、員工安全、教育訓練"}
]

# 預設的準則（可以根據實際情況進行修改或擴展）
criteria = [
    "人力資源管理", "績效管理", "新資福利", "員工安全", "教育訓練", "招聘策略", "員工福利", "員工滿意度", "工作環境", "團隊合作",
    "領導力", "員工參與", "績效評估", "獎勳與激勳", "員工健康", "工作生活平衡", "員工離職率", "專業發展", "知識管理", "創新文化",
    "內部溝通", "目標設定", "員工自我發展", "培訓與發展", "員工敬業度", "績效改善", "工作安全", "團隊發展", "薪資結構", "獎勳制度",
    "員工回饋", "技能提升", "工作壓力", "職場歧視", "勞動法規", "福利規劃", "心理健康", "領導風格", "員工流動性", "內部招聘",
    "工作滿意度", "員工參與度", "勞動關係", "企業文化", "社會責任", "業務管理", "風險管理", "財務管理", "經濟激勳", "員工關係"
]

@app.route('/')
def index():
    # 創建表格內容
    table_html = ""

    # 遍歷每個準則
    for criterion in criteria:
        row_html = f"<tr><td>{criterion}</td>"

        # 檢查每篇文獻是否提到該準則
        for data in literature_data:
            if criterion in data["abstract"]:
                row_html += "<td>&#x25CB;</td>"  # 代表有提到，畫圓圈
            else:
                row_html += "<td></td>"  # 沒有提到，留空
        row_html += "</tr>"
        table_html += row_html

    # 使用模板渲染表格
    return render_template_string('''
        <!DOCTYPE html>
        <html lang="zh-TW">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>文獻關鍵字分析表格生成</title>
            <style>
                table {
                    width: 100%;
                    border-collapse: collapse;
                }
                table, th, td {
                    border: 1px solid black;
                }
                th, td {
                    padding: 10px;
                    text-align: center;
                }
                th {
                    background-color: #f2f2f2;
                }
            </style>
        </head>
        <body>
            <h2>文獻關鍵字分析表格</h2>
            <table>
                <thead>
                    <tr>
                        <th>構面/準則</th>
                        <th>A</th>
                        <th>B</th>
                        <th>C</th>
                        <th>D</th>
                        <th>E</th>
                        <th>F</th>
                        <th>G</th>
                    </tr>
                </thead>
                <tbody>
                    {{ table_html|safe }}
                </tbody>
            </table>
        </body>
        </html>
    ''', table_html=table_html)

if __name__ == '__main__':
    app.run(debug=True)
