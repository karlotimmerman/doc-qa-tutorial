import streamlit as st # 
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.vectorstores.faiss import FAISS

def main():
    st.title("PDF Question Answering")

    # File upload
    st.header("Upload a PDF file")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf"])

    if uploaded_file is not None:
        # Parse PDF
        st.header("Parsing PDF...")
        text = parse_pdf(uploaded_file)

        # Embed text
        st.header("Embedding text...")
        index = embed_text(text)

        if index is not None:
            # Query answering
            st.header("Ask a question")
            question = st.text_input("Type your question here")
            if question:
                answer = get_answer(index, question)
                st.header("Answer")
                st.write(answer)

def parse_pdf(file):
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        output.append(text)
    return "\n\n".join(output)

@st.cache
def embed_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=0, separators=["\n\n", ".", "?", "!", " ", ""]
    )
    texts = text_splitter.split_text(text)

    try:
        embeddings = OpenAIEmbeddings()
        index = FAISS.from_texts(texts, embeddings)
    except Exception as e:
        st.error(f"Error creating embeddings: {e}")
        return None

    return index

def get_answer(index, query):
    docs = index.similarity_search(query)

    chain = load_qa_chain(OpenAI(temperature=0))
    answer = chain.run(input_documents=docs, question=query)

    return answer

if __name__ == "__main__":
    main()
