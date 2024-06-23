from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

def retriver_(text, embedding):
    temp_text_path = "data/file.txt"
    with open(temp_text_path, "w", encoding="utf-8") as f:
        f.write(text)

    documents = TextLoader(
        temp_text_path,
        encoding="utf-8",
    ).load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    for idx, text in enumerate(texts):
        text.metadata["id"] = idx

    retriever = FAISS.from_documents(texts, embedding).as_retriever(search_kwargs={"k": 2})

    return retriever