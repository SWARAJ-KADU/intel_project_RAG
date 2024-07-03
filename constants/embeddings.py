import openvino as ov
from langchain_community.embeddings import OpenVINOEmbeddings

core = ov.Core()
devices = core.available_devices
def embeddings_():
    embedding_model_name = "BAAI/bge-large-en-v1.5"
    embedding_model_kwargs = {"device": devices[1]}
    encode_kwargs = {
        "normalize_embeddings": True,
    }

    embedding = OpenVINOEmbeddings(
        model_name_or_path=embedding_model_name,
        model_kwargs=embedding_model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    return embedding
