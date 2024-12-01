import requests
from bs4 import BeautifulSoup
import json
import re

def clean_text(text):
    # 替换掉常见的非标准空格和特殊字符，包括 \u00a0、\u200b 等
    text = (text
            .replace('\u00a0', ' ')  # 替换不间断空格
            .replace('\u2013', '-')  # 替换短横线
            .replace('\u2014', '-')  # 替换长横线
            .replace('\u200b', '')   # 替换零宽空格
           )
    
    # 使用正则表达式去除不必要的 " | " 并修复格式
    text = re.sub(r'\s*\|\s*', ' ', text)  # 将 | 两边的空格去除并替换成一个空格
    text = re.sub(r'\(\s*\|\s*', '(', text)  # 去掉括号内多余的 | 和空格
    text = re.sub(r'\|\s*\)', ')', text)  # 去掉右括号前的 | 符号
    text = re.sub(r'\s+', ' ', text).strip()  # 去除多余空格
    
    return text

def extract_infobox(url):
    # 发送 HTTP 请求获取页面内容
    response = requests.get(url)
    
    # 检查请求是否成功
    if response.status_code != 200:
        print(f"Failed to retrieve page. Status code: {response.status_code}")
        return None
    
    # 解析页面内容
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # 查找 class="infobox" 的表格
    infobox = soup.find('table', {'class': 'infobox'})
    
    if infobox is None:
        print("No infobox found on this page.")
        return None
    
    # 提取表格的所有行
    rows = infobox.find_all('tr')
    
    # 保存提取的 infobox 信息
    infobox_data = {}
    current_section = None
    
    for row in rows:
        header = row.find('th')
        data = row.find('td')
        
        if header and data:
            # 处理表格中的换行符，使用 "|" 连接多行内容并清理文本
            header_text = clean_text(header.text.strip())
            data_text = clean_text(' | '.join([line.strip() for line in data.stripped_strings]))
            
            # 如果表头和数据都存在，且表头中含有子标题，则视为新的一节
            if header_text:
                if current_section is None:
                    current_section = header_text
                    infobox_data[current_section] = {header_text: data_text}
                else:
                    infobox_data[current_section][header_text] = data_text
        elif header: 
            # 更新当前的 section
            current_section = clean_text(header.text.strip())
            infobox_data[current_section] = {}
    
    return infobox_data

def save_infobox_to_json(url, category, output_file):
    infobox_data = extract_infobox(url)
    
    if infobox_data is None:
        return
    
    # 以指定格式创建数据结构
    result = {
        "gold_evidences": {
            "category": category,
            list(infobox_data.keys())[0]: infobox_data  # 第一个键作为主要条目名
        }
    }
    
    # 保存到 JSON 文件
    with open(output_file, 'w') as json_file:
        json.dump(result, json_file, indent=4)

# 使用示例，提供 Wikipedia 页面 URL 和类别
url = "https://en.wikipedia.org/wiki/Vijay Singh"
category = "golf"
output_file = "infobox_data_cleaned_with_formatting.json"
save_infobox_to_json(url, category, output_file)
