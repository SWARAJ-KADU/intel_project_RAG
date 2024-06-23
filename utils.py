from PyPDF2 import PdfReader
def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [
                f"Document {i+1}:\n\n{d.page_content}\nMetadata: {d.metadata}"
                for i, d in enumerate(docs)
            ]
        )
    )

def format_docs(docs):
    formatted_docs = "\n".join(
        [
            f"{'-' * 100}\nDocument {i+1}:\n\n{d.page_content}\nMetadata: {d.metadata}"
            for i, d in enumerate(docs)
        ]
    )
    return formatted_docs

def extract_text_from_pdf(file):
    text = ""
    reader = PdfReader(file)
    for page in reader.pages:
        text += page.extract_text()
    return text


