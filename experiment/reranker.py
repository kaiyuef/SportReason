from FlagEmbedding import FlagReranker

def bge(query, bm25_results, normalize=False):
        """
        对 BM25 检索结果进行重排序。
        
        Args:
            query (str): 查询文本。
            bm25_results (list[dict]): BM25 检索结果列表，每个结果包含文档内容。
            normalize (bool): 是否将分数归一化到 [0, 1]。
        
        Returns:
            list[dict]: 重排序后的结果列表。
        """
        reranker = FlagReranker(
                    'BAAI/bge-reranker-v2-m3', 
                    query_max_length=256,
                    passage_max_length=512,
                    use_fp16=True,
                    devices=['cuda:1']
                ) # Setting use_fp16 to True speeds up computation with a slight performance degradation
        
        passages = [result["content"] for result in bm25_results]
        input_pairs = [(query, passage) for passage in passages]
        scores = reranker.compute_score(input_pairs, normalize=normalize)

        # 将得分加入 BM25 结果并重新排序
        for i, score in enumerate(scores):
            bm25_results[i]["rerank_score"] = score
        bm25_results.sort(key=lambda x: x["rerank_score"], reverse=True)

        return bm25_results