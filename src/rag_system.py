from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv, find_dotenv
import os
import openai
from langchain_community.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']


def load_to_chroma(pdf_paths: [str, ...], chroma_dir: str, llm_name: str, embedding_model_name: str):
    """
    Loads PDF documents into a Chroma vector database after splitting and embedding the text within.
    :param pdf_paths: A list of paths to PDF files to be loaded.
    :param chroma_dir: The directory where the Chroma database will be persisted.
    :param llm_name: The name of the language model that will be used during inference, to ensure fitting splitting.
    :param embedding_model_name: The name of the model used to generate text embeddings.
    :return:
    """

    loaders = [PyPDFLoader(i) for i in pdf_paths]
    docs = []
    for loader in loaders:
        docs.extend(loader.load())

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=2000, chunk_overlap=200, model_name=llm_name, separator="\n")
    splits = text_splitter.split_documents(docs)

    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=OpenAIEmbeddings(model=embedding_model_name),
        persist_directory=chroma_dir
    )
    vectordb.persist()


def RAG(question: str, chroma_dir: str, llm_name: str, embedding_model_name: str) -> str:
    """
    Performs Retrieval-Augmented Generation (RAG) to answer a question using a Chroma vector database.
    :param question: The input question to be answered.
    :param chroma_dir: The directory where the Chroma database is stored.
    :param llm_name: The name of the language model used for generating the answer.
    :param embedding_model_name: The name of the model used to generate text embeddings.
    :return: The model's response.
    """

    template = """
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Context: {context}
    Question: {question}
    Answer:
    """
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

    vectordb = Chroma(persist_directory=chroma_dir, embedding_function=OpenAIEmbeddings(model=embedding_model_name))
    qa_chain = RetrievalQA.from_chain_type(
        ChatOpenAI(model_name=llm_name, temperature=0),
        retriever=vectordb.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    # TODO text only
    return qa_chain({"query": question})
