import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Sidebar contents
with st.sidebar:
    st.title('Interviewer')
    st.markdown('''
    ## About
    This app is an LLM-powered Interviwer built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
    ''')

    st.write('Making interviews accessible')


def main():
    st.write("Interview Buddy")

    load_dotenv()

    # upload a pdf file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    # st.write(pdf.name)

    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        # embeddings
        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        store_name = pdf.name[:-4]
        with open(f"{store_name}.pk1, wb") as f:
            pickle.dump(VectorStore, f)


if __name__ == '__main__':
    main()
