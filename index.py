import os

# 配置环境变量
os.environ['JAVA_HOME'] = '/Library/Java/JavaVirtualMachines/zulu-23.jdk/Contents/Home'
os.environ['JVM_PATH'] = '/Library/Java/JavaVirtualMachines/zulu-23.jdk/Contents/Home/lib/server/libjvm.dylib'

# 测试 JVM 是否加载正确
from jnius import autoclass
System = autoclass('java.lang.System')
print(f"Java version: {System.getProperty('java.version')}")

# 构建索引
os.system(
    "python -m pyserini.index.lucene "
    "--collection JsonCollection "
    "--input /Users/kevinfeng/numsports/corpus "
    "--index /Users/kevinfeng/numsports/indexes/lucene-index "
    "--generator DefaultLuceneDocumentGenerator "
    "--threads 4 "
    "--storeRaw"
)

print("索引已构建完成！")
