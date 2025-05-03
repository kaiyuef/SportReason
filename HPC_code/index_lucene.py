import os

# 测试 JVM 是否加载正确
from jnius import autoclass
System = autoclass('java.lang.System')
print(f"Java version: {System.getProperty('java.version')}")

# 构建索引
os.system(
    "python -m pyserini.index.lucene "
    "--collection JsonCollection "
    "--input numsports/corpus "
    "--index numsports/indexes/lucene-index "
    "--generator DefaultLuceneDocumentGenerator "
    "--threads 4 "
    "--storeRaw"
)

print("索引已构建完成！")
