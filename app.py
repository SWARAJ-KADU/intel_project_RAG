import streamlit as st
from constants.embeddings import embeddings_
from constants.reranker import reranker_
from constants.retriver import retriver_
from utils import extract_text_from_pdf, format_docs
from summarization import summarize_text
from io import BytesIO

def main(embedding):
    st.title("Tasks with OpenVINO")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt"])
    
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type == "text/plain":
            text = uploaded_file.read().decode("utf-8")
        else:
            st.error("Unsupported file type")
        
        option = st.selectbox("Choose an option", ["Summarization", "Question Answering"])
        if option == "Summarization":
            summarize_ = summarize_text(text)
            st.text_area("POSSIBLE ANSWER", summarize_, height=200)
            summary_file = BytesIO(summarize_.encode('utf-8'))
            st.download_button(
                label="Download Summary",
                data=summary_file,
                file_name="summary.txt",
                mime="text/plain",
            )
        else:
            st.error("Please enter some text to summarize")
            retriver = retriver_(text, embedding)
            reranker = reranker_(retriver)
            query = st.text_input("Enter your query:")
            ans = format_docs(reranker.invoke(query))
            st.text_area("POSSIBLE ANSWER", ans, height=300)

if __name__ == "__main__":
    emb = embeddings_()
    main(emb)