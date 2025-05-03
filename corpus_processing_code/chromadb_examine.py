import chromadb

# ==== 配置 ====
PERSIST_DIR     = "numsports/indexes/table_index"
COLLECTION_NAME = "numsports_table_corpus"

# ==== 打开本地持久化 Client ====
client     = chromadb.PersistentClient(path=PERSIST_DIR)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

# ==== 取回一条记录（含 embedding） ====
result = collection.get(
    limit=1,
    include=["documents", "metadatas", "embeddings"]
)

# ==== 打印结果 ====
doc_id    = result["ids"][0]         # IDs 总是会返回
doc_text  = result["documents"][0]
meta      = result["metadatas"][0]
embed_vec = result["embeddings"][0]

print(f"ID       : {doc_id}")
print(f"Meta     : {meta}")
print(f"Content  : {doc_text[:200]}{'...' if len(doc_text)>200 else ''}")
print(f"Vector 长度: {len(embed_vec)}")
print(f"Embedding : {embed_vec!r}")
