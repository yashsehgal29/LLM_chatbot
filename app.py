import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import ctransformers, replicate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import faiss
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os
from dotenv import load_dotenv
import tempfile

load_dotenv()


def initial_session():
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "generated" not in st.session_state:
        st.session_state["generated"] = ["Helloo!! Ask Me anything"]
    if "past" not in st.session_state:
        st.session_state["past"] = ["Hey!!"]


def conversational_chain(vectorstore):
    load_dotenv()
    llm = replicate.Replicate(streaming=True, model="meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
                              callbacks={StreamingStdOutCallbackHandler()},
                              input={"temperature": 0.01, "max_length": 500, "top_1": 1})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever(search_kwargs={"k": 2}), memory=memory)
    return chain


def conversation_chat(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]


def display_history(chain):
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key="my_form", clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your documents", key="input")
            submit_but = st.form_submit_button(label="Send")

        if submit_but and user_input:
            with st.spinner("Generating Response ...."):
                output = conversation_chat(user_input, chain, st.session_state['history'])
                st.session_state["past"].append(user_input)
                st.session_state["generated"].append(output)

    if st.session_state["generated"]:
        with reply_container:
            for i in range(len(st.session_state["generated"])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + "_user", avatar_style="adventurer")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")


def main():
    initial_session()
    st.title("Multi-Docs Chatbot")
    # st.sidebar.title('Documents Processing')

    # uploader_files = st.sidebar.file_uploader("Uploaded files", accept_multiple_files=True)
    file=os.path("Panchatantra-.pdf")
    # if uploader_files:
    text = []

        # for file in uploader_files:
    file_ext = os.path.splitext(file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(file.read())
        temp_file_path = temp_file.name
    loader = PyPDFLoader(temp_file_path)
    # loader = None
    # if file_ext == ".pdf":
    
    # elif file_ext == ".docx" or file_ext == ".doc":
    #     loader = Docx2txtLoader(temp_file_path)
    # elif file_ext == ".txt":
    #     loader = TextLoader(temp_file_path)

    if loader:
        text.extend(loader.load())
        os.remove(temp_file_path)

    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100, length_function=len)
    text_chunks = text_splitter.split_documents(text)

    # Creating embeddings
    embedds = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})

    # Creating a vector store
    vector_store = faiss.FAISS.from_documents(text_chunks, embedding=embedds)

    # Chain the objects
    chain = conversational_chain(vectorstore=vector_store)
    display_history(chain)


if __name__ == "__main__":
    main()
