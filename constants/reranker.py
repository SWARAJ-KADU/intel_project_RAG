from langchain_community.document_compressors.openvino_rerank import OpenVINOReranker
from langchain.retrievers import ContextualCompressionRetriever


def reranker_(retriever):
    rerank_model_name = "BAAI/bge-reranker-base"
    rerank_model_kwargs = {"device": "CPU"}
    rerank_top_n = 2

    reranker = OpenVINOReranker(
        model_name_or_path=rerank_model_name,
        model_kwargs=rerank_model_kwargs,
        top_n=rerank_top_n,
    )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker, base_retriever=retriever
    )
    return compression_retriever