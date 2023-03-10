import streamlit as st
from utils import parse_pdf, embed_text, get_answer

def main():
    st.header("I-Doc")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file is not None:
        index = embed_text(parse_pdf(uploaded_file))
        query = st.text_area("Ask a question about the document")
        button = st.button("Submit")

        if button:
            answer = get_answer(index, query)
            st.write(answer)

if __name__ == '__main__':
    main()
