import os
import subprocess

def configure_java_env(java_home, jvm_path):
    """
    配置 Java 环境变量以支持 Pyserini。
    """
    os.environ['JAVA_HOME'] = java_home
    os.environ['JVM_PATH'] = jvm_path
    print(f"Configured JAVA_HOME: {java_home}")
    print(f"Configured JVM_PATH: {jvm_path}")

def build_index(corpus_path, index_path, threads=4):
    """
    构建 Pyserini 索引。
    :param corpus_path: JSON 文档集合的路径
    :param index_path: 索引输出路径
    :param threads: 并行线程数
    """
    command = [
        "python", "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", corpus_path,
        "--index", index_path,
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", str(threads),
        "--storeRaw"
    ]
    print("Building index with command:")
    print(" ".join(command))
    
    try:
        subprocess.run(command, check=True)
        print("索引已成功构建！")
    except subprocess.CalledProcessError as e:
        print(f"索引构建失败: {e}")
