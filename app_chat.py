import os
import time
import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.vectorstores import Chroma
from langchain.agents.agent_toolkits import (
    VectorStoreToolkit,
    VectorStoreInfo,
)
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


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
    llm = ChatOpenAI(
        temperature=0,
        verbose=True,
        model="gpt-3.5-turbo-16k",
        openai_api_key=OPENAI_API_KEY,
    )
    embeddings = OpenAIEmbeddings()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    if not os.path.exists(db_path):
        loader = PyPDFDirectoryLoader(data_path)
        pages = loader.load_and_split()
        store = Chroma.from_documents(pages, embeddings, persist_directory=db_path)
        store.persist()
    else:
        store = Chroma(embedding_function=embeddings, persist_directory=db_path)

    vectorstore_info = VectorStoreInfo(
        name="nist_db",
        description="NIST SP Framework",
        vectorstore=store,
    )
    toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)
    agent_executor = initialize_agent(
        tools=toolkit.get_tools(),
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
    )

    return agent_executor, store


# User input
## Function for taking user provided prompt as input
def get_text():
    input_text = st.text_input("You: ", "", key="input")
    return input_text


# Response output
## Function for taking user prompt as input followed by producing AI generated responses
def generate_response(prompt, agent):
    response = agent.run(input=prompt)
    return response


def main():
    startTime = time.time()

    data_path, db_path = setup_paths()
    agent_executor, store = setup_vectorstore_agent(data_path, db_path)

    st.set_page_config(page_title="NIST GPT - Your NIST AI Consultant")
    st.title("🦜🔗 NIST SP GPT")

    if "generated" not in st.session_state:
        st.session_state["generated"] = ["I'm NIST GPT, How may I help you?"]
    if "past" not in st.session_state:
        st.session_state["past"] = ["Hi!"]

    input_container = st.container()
    colored_header(label="", description="", color_name="blue-30")
    response_container = st.container()

    ## Applying the user input box
    with input_container:
        user_input = get_text()

    ## Conditional display of AI generated responses as a function of user provided prompts
    with response_container:
        if user_input:
            response = generate_response(user_input, agent_executor)
            st.session_state.past.append(user_input)
            st.session_state.generated.append(response)

        if st.session_state["generated"]:
            for i in range(len(st.session_state["generated"])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
                message(st.session_state["generated"][i], key=str(i))

    executionTime = time.time() - startTime
    print(f"Time taken: {executionTime}")


if __name__ == "__main__":
    main()
