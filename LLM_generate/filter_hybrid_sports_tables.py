"""
过滤 HybridQA 的 WikiTables，将“体育”相关表格复制到
  sports_tables_tok/  和  sports_request_tok/  两个缓存目录。

用法：
    python filter_sports_tables.py
"""

import os, json, shutil, asyncio
from typing import Dict
from dotenv import load_dotenv
from openai import AsyncOpenAI

# ---------- 常量 ----------
TABLE_DIR        = "hybridqa/WikiTables-WithLinks/tables_tok"
HYPERLINK_DIR    = "hybridqa/WikiTables-WithLinks/request_tok"
SPORTS_TABLE_DIR = "hybridqa/WikiTables-WithLinks/sports_tables_tok"
SPORTS_LINK_DIR  = "hybridqa/WikiTables-WithLinks/sports_request_tok"
CONCURRENCY      = 3            # 同时判定 3 张表，避免触发速率限制

# ---------- 初始化 ----------
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    raise RuntimeError("缺少 OPENAI_API_KEY！请放到环境变量或 .env 文件中")

client = AsyncOpenAI(api_key=OPENAI_KEY, base_url="https://api.openai.com/v1")
os.makedirs(SPORTS_TABLE_DIR, exist_ok=True)
os.makedirs(SPORTS_LINK_DIR,  exist_ok=True)
sema = asyncio.Semaphore(CONCURRENCY)

# ---------- 判定函数 ----------
async def is_sports_table(table_struct: Dict, file_name: str) -> bool:
    """
    使用 gpt‑4o‑mini 判断表格是否与体育相关。
    将文件名也包含在 prompt 中，供模型参考。
    """
    header = table_struct["columns"]
    rows   = table_struct["rows"][:3]

    prompt = (
        "Respond with only 'yes' or 'no'. "
        "Is the following table primarily about sports (teams, leagues, matches, players, etc.)?\n"
        f"File name: {file_name}\n"
        f"Header: {header}\n"
        f"Rows (first 3): {rows}"
    )

    resp = await client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=1
    )
    return resp.choices[0].message.content.strip().lower().startswith("y")

# ---------- 处理单个文件 ----------
async def process_file(file_name: str):
    async with sema:
        dst_table_path = os.path.join(SPORTS_TABLE_DIR, file_name)
        if os.path.exists(dst_table_path):
            print(f"✔ 已缓存 {file_name}，跳过判定")
            return

        table_path     = os.path.join(TABLE_DIR, file_name)
        hyperlink_path = os.path.join(HYPERLINK_DIR, file_name)

        # 载入表格
        try:
            table_json = json.load(open(table_path, encoding="utf-8"))
            full_table = {
                "columns": table_json["header"],
                "rows": [[cell[0] for cell in row] for row in table_json["data"]]
            }
        except Exception as e:
            print(f"❌ 读取失败 {file_name}: {e}")
            return

        # 判定
        try:
            if await is_sports_table(full_table, file_name):
                shutil.copyfile(table_path, dst_table_path)
                if os.path.exists(hyperlink_path):
                    shutil.copyfile(
                        hyperlink_path,
                        os.path.join(SPORTS_LINK_DIR, file_name)
                    )
                print(f"✅ 体育表格 {file_name} → 已缓存")
            else:
                print(f"⏩ 非体育表格 {file_name}")
        except Exception as e:
            print(f"❌ 判定出错 {file_name}: {e}")

# ---------- 主入口 ----------
async def main():
    files = [f for f in os.listdir(TABLE_DIR) if f.endswith(".json")]
    await asyncio.gather(*(process_file(f) for f in files))
    print("🎉 预处理完成！")

if __name__ == "__main__":
    asyncio.run(main())
