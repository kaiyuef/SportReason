import os
import requests

# 提取标题
def extract_title_from_html_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if '<title>' in line:
                start = line.find('<title>') + len('<title>')
                end = line.find('</title>')
                title = line[start:end]
                return title.split(' - Wikipedia')[0]  # 去掉 " - Wikipedia" 后缀
    return None

# 获取相关标题
def get_related_titles(title, max_titles=10):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        'action': 'query',
        'format': 'json',
        'titles': title,
        'prop': 'links',
        'pllimit': 'max'
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"Failed to fetch related titles for {title}")
        return []

    data = response.json()
    pages = data.get('query', {}).get('pages', {})
    related_titles = []
    for page_id, page_data in pages.items():
        links = page_data.get('links', [])
        related_titles.extend(link['title'] for link in links if ':' not in link['title'])
    return related_titles[:max_titles]

# 下载 HTML 页面
def fetch_html_by_title(title):
    url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    print(f"Failed to fetch HTML for {title}")
    return None

# 数据扩充
def expand_dataset(input_folder, output_folder, multiplier=10):
    processed_titles = set()

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    for file_name in os.listdir(input_folder):
        if not file_name.endswith('.html'):
            continue  # 跳过非 HTML 文件

        file_path = os.path.join(input_folder, file_name)
        title = extract_title_from_html_file(file_path)
        if not title:
            print(f"Skipping file {file_name}: Unable to extract title")
            continue

        if title in processed_titles:
            print(f"Skipping already processed title: {title}")
            continue
        
        print(f"Processing title: {title}")
        processed_titles.add(title)

        # 获取相关标题
        related_titles = get_related_titles(title, max_titles=multiplier)
        for i, related_title in enumerate(related_titles):
            if related_title in processed_titles:
                continue

            new_html_content = fetch_html_by_title(related_title)
            if new_html_content:
                new_file_name = f"{file_name.split('.')[0]}_related_{i}.html"
                new_file_path = os.path.join(output_folder, new_file_name)

                with open(new_file_path, 'w', encoding='utf-8') as out_file:
                    out_file.write(new_html_content)
                processed_titles.add(related_title)

# 主函数
def main():
    input_folder = "wikipedia_html"  # 输入 HTML 文件夹路径
    output_folder = "wikipedia_expanded_html"  # 输出 HTML 文件夹路径
    multiplier = 10  # 每个标题扩充的相关页面数量

    expand_dataset(input_folder, output_folder, multiplier)

if __name__ == "__main__":
    main()
