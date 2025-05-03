import requests
import json
import logging
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

# 配置日志记录
logging.basicConfig(filename='download_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def sanitize_filename(filename):
    """
    清理文件名中的非法字符，确保文件名合法。

    参数：
        filename (str): 原始文件名。

    返回：
        str: 清理后的合法文件名。
    """
    # 定义非法字符的正则表达式模式
    invalid_chars = r'[<>:"/\\|?*\n\t]'
    # 替换非法字符为下划线
    sanitized = re.sub(invalid_chars, '_', filename)
    # 移除文件名开头和结尾的空白字符
    sanitized = sanitized.strip()
    return sanitized

def download_html(url, title, save_dir):
    """
    下载给定URL的HTML内容并保存到指定目录，文件名为对应的标题。

    参数：
        url (str): 维基百科页面的URL。
        title (str): 维基百科页面的标题。
        save_dir (str): 保存HTML文件的目录。

    返回：
        None
    """
    # 清理标题，生成合法的文件名
    valid_filename = sanitize_filename(title)
    valid_filename = valid_filename.replace(' ', '_') + '.html'
    file_path = os.path.join(save_dir, valid_filename)

    # 检查文件是否已存在，避免覆盖
    if os.path.exists(file_path):
        logging.info(f"File already exists for {url} with title '{title}'. Skipping download.")
        return

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        logging.info(f"Successfully saved HTML for {url} as {valid_filename}")
    except requests.RequestException as e:
        logging.error(f"Failed to download {url}: {e}")

def process_dataset_for_html(input_file, save_dir, max_workers=10):
    """
    处理数据集，提取所有维基百科链接并保存对应的HTML文件，文件名为页面标题。

    参数：
        input_file (str): 输入数据集的文件路径（JSON格式）。
        save_dir (str): 保存HTML文件的目录。
        max_workers (int): 最大并行线程数。

    返回：
        None
    """
    # 创建保存目录（如果不存在）
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # 使用字典来存储URL到标题的映射，确保每个URL只下载一次
    url_title_map = {}

    for element in tqdm(dataset, desc="Collecting URLs"):
        for evidence in element.get('gold_evidences', []):
            # 直接从gold_evidences中获取title和url
            url = evidence.get('meta', {}).get('url', '')
            title = evidence.get('title', '')
            if url and title:
                # 如果URL未被处理过，添加到映射中
                if url not in url_title_map:
                    url_title_map[url] = title
                else:
                    # 如果URL已经存在，但标题不同，记录警告日志
                    if url_title_map[url] != title:
                        logging.warning(f"Different titles for the same URL {url}: '{url_title_map[url]}' and '{title}'")
    
    logging.info(f"Total unique URLs to download: {len(url_title_map)}")

    # 使用线程池并行下载HTML文件
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_html, url, title, save_dir): url for url, title in url_title_map.items()}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading HTML files"):
            url = futures[future]
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error downloading {url}: {e}")

    logging.info("All HTML files have been downloaded.")

# 使用示例
input_file = "merged_dataset.json"  # 替换为你的输入数据集文件
html_save_directory = "wikipedia_html"  # 保存HTML文件的目录
process_dataset_for_html(input_file, html_save_directory, max_workers=10)
