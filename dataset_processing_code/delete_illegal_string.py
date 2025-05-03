import json
import re

def fix_invalid_json(input_file, output_file):
    """
    修复 JSONL 文件中的非法字符，并将修复后的内容保存到新的文件中。
    """
    def fix_line(line):
        # 替换未转义的反斜杠
        line = re.sub(r'\\([^"\\/bfnrtu])', r'\\\\\1', line)  # 非合法转义字符后添加反斜杠
        # 转义未关闭的字符串（如以反斜杠结尾的字符串）
        line = line.replace("\\", "\\\\")
        return line

    with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue  # 跳过空行
            try:
                json.loads(line)  # 尝试解析
                fout.write(line + "\n")  # 如果解析成功，直接写入
            except json.JSONDecodeError:
                print(f"Fixing line {i}: {line}")
                fixed_line = fix_line(line)  # 修复非法字符
                try:
                    json.loads(fixed_line)  # 再次尝试解析
                    fout.write(fixed_line + "\n")  # 修复成功后写入
                except json.JSONDecodeError:
                    print(f"Failed to fix line {i}: {line}")

    print(f"Processed file saved to: {output_file}")

# 输入和输出文件路径
input_file = "corpus/expanded_tables_content.jsonl"
output_file = "corpus/fix_expanded_tables_content.jsonl"

fix_invalid_json(input_file, output_file)
