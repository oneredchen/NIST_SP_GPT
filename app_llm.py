import os
import time
import traceback
import logging
import streamlit as st
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.vectorstores import Chroma
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo,
)
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

logging.basicConfig(level=logging.INFO)


def setup_paths():
    """Sets up necessary paths and returns them. Creates the db directory if it doesn't exist."""
    cwd = os.getcwd()
    data_path = os.path.join(cwd, "raw_data")
    db_dir = os.path.join(cwd, "db")
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
    db_path = os.path.join(db_dir, "nist.db")
    return data_path, db_path


def setup_vectorstore_agent(data_path, db_path):
    """Sets up the Langchain LLM to be used and returns a vectorstore agent."""
    llm = OpenAI(temperature=0.1, verbose=True, model="text-davinci-003")
    embeddings = OpenAIEmbeddings()
    memory = ConversationBufferMemory(memory_key="chat_history")

    if not os.path.exists(db_path):
        logging.info("Existing NIST Database not found, generating NIST Database ...")
        loader = PyPDFDirectoryLoader(data_path)
        pages = loader.load_and_split()
        store = Chroma.from_documents(pages, embeddings, persist_directory=db_path)
        store.persist()
    else:
        logging.info("Existing NIST Database found, loading NIST Database ...")
        store = Chroma(embedding_function=embeddings, persist_directory=db_path)

    vectorstore_info = VectorStoreInfo(
        name="nist_db",
        description="NIST SP Framework",
        vectorstore=store,
    )
    toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)
    print(toolkit.get_tools())
    agent_executor = create_vectorstore_agent(
        llm=llm, toolkit=toolkit, verbose=True, memory=memory
    )

    return agent_executor, store


def main():
    startTime = time.time()

    data_path, db_path = setup_paths()
    agent_executor, store = setup_vectorstore_agent(data_path, db_path)

    st.set_page_config(page_title="NIST GPT - Your NIST AI Consultant")
    st.title("🦜🔗 NIST SP GPT")

    prompt = st.text_input("Input your prompt here")

    if prompt:
        try:
            response = agent_executor.run(prompt)
        except Exception as e:
            response = f"Looks like the following error took place: {e}"
            logging.error(traceback.format_exc())
        st.write(response)

    executionTime = time.time() - startTime
    logging.info(f"Execution time in seconds: {executionTime}")


if __name__ == "__main__":
    main()
