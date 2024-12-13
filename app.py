from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain

from langchain_community.document_loaders import PyPDFDirectoryLoader 
from langchain_community.vectorstores import FAISS
import os

groq_api_key = os.getenv('GROQ_API_KEY')

llm = ChatGroq(groq_api_key=groq_api_key, model="Llama3-8b-8192")

prompt=ChatPromptTemplate.from_template(
    """"
    Answer the question based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context} 
    </context>
    Question: {input}
    """ # Here contect means the research papers and input means user question
)

def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embedding = OllamaEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader('research_papers')
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_docs = st.session_state.splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_docs, st.session_state.embedding)

st.title("RAG Document Q&A with Groq and Llama3")

user_prompt = st.text_input("Please enter your query from research paper")

if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("Vector embedding is ready")

if user_prompt:
    document_chain = create_stuff_documents_chain(llm,prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = retrieval_chain.invoke({'input':user_prompt})
    st.write(response['answer'])

    with st.expander("Document similarity search"):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content) 
            st.write('-----')
