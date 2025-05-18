import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from docx import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import functools
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json

# File to store user data
USER_DATA_FILE = "users.json"

# Load existing user data from the file
def load_user_data():
    try:
        with open(USER_DATA_FILE, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return {"usernames": [], "hashed_passwords": []}

# Save user data to the file
def save_user_data(user_data):
    with open(USER_DATA_FILE, "w") as file:
        json.dump(user_data, file)

# Initialize user data
user_data = load_user_data()
usernames = user_data["usernames"]
hashed_passwords = user_data["hashed_passwords"]

# Hash passwords using SHA256
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Check if the input password matches the stored hash
def authenticate(username, password):
    if username in usernames:
        user_index = usernames.index(username)
        return hashed_passwords[user_index] == hash_password(password)
    return False

# Register a new user
def register(username, password):
    if username in usernames:
        return False  # User already exists
    usernames.append(username)
    hashed_passwords.append(hash_password(password))
    # Save the updated user data
    save_user_data({"usernames": usernames, "hashed_passwords": hashed_passwords})
    return True

def process_chunk(chunk):
    return chunk

@functools.cache
def cached_query(question):
    return st.session_state.conversation({'question': question})

def get_txt_text(txt_file):
    return txt_file.read().decode("utf-8")

def get_docx_text(docx_file):
    doc = Document(docx_file)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])

def extract_text_from_files(files):
    text = ""
    for file in files:
        if file.name.endswith('.pdf'):
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        elif file.name.endswith('.txt'):
            text += get_txt_text(file)
        elif file.name.endswith('.docx'):
            text += get_docx_text(file)
        else:
            st.error(f"Unsupported file type: {file.name}")
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    with ThreadPoolExecutor() as executor:
        processed_chunks = list(executor.map(process_chunk, chunks))

    return processed_chunks

def get_vectorstore(text_chunks):
    if not text_chunks:
        raise ValueError("Text chunks are empty. Check your document processing.")

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

    if vectorstore is None:
        raise ValueError("Vectorstore creation failed.")

    return vectorstore

def get_conversation_chain(vectorstore):
    # llm = ChatOpenAI()
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
    )
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)

    if vectorstore is None:
        raise ValueError("Vectorstore is None. Check your embeddings and vectorstore creation.")

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    if conversation_chain is None:
        raise ValueError("Conversation chain creation failed.")

    return conversation_chain

def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.error("The conversation chain has not been initialized. Please upload and process documents first.")
        return

    with st.spinner("Thinking..."):
        response = cached_query(user_question)

    st.session_state.chat_history = response['chat_history']


    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            with st.chat_message("user"):
                st.write(message.content)
        else:
            with st.chat_message("ai"):
                st.write(message.content)

def main():
    load_dotenv()  
    st.set_page_config(page_title="Chat with your Documents", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = None

    if st.session_state.authenticated:
        st.sidebar.success(f"Welcome, {st.session_state.username}!")
        logout_button = st.sidebar.button("Logout")
        if logout_button:
            st.session_state.authenticated = False
            st.session_state.username = None
            st.sidebar.warning("You have been logged out.")
    else:
        tab = st.sidebar.radio("Account", ["Login", "Register"])

        if tab == "Login":
            st.sidebar.subheader("Login")
            username = st.sidebar.text_input("Username")
            password = st.sidebar.text_input("Password", type="password")
            login_button = st.sidebar.button("Login")

            if login_button:
                if authenticate(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.sidebar.success(f"Welcome, {username}!")
                else:
                    st.sidebar.error("Invalid username or password")
        elif tab == "Register":
            st.sidebar.subheader("Register")
            new_username = st.sidebar.text_input("New Username")
            new_password = st.sidebar.text_input("New Password", type="password")
            register_button = st.sidebar.button("Register")

            if register_button:
                if register(new_username, new_password):
                    st.sidebar.success("Registration successful! You can now log in.")
                else:
                    st.sidebar.error("Username already exists. Please try a different one.")

    if st.session_state.authenticated:
        st.header("Chat with your Documents :books:")
        user_question = st.chat_input("Ask a question about your documents:")
        if user_question:
            handle_userinput(user_question)

        with st.sidebar:
            st.subheader("Your documents")
            uploaded_files = st.file_uploader(
                "Upload your documents here and click on 'Process'", 
                accept_multiple_files=True, 
                type=['pdf', 'txt', 'docx']
            )
            if st.button("Process"):
                with st.spinner("Processing"):
                    raw_text = extract_text_from_files(uploaded_files)
                    if not raw_text:
                        st.error("No text extracted from the uploaded documents. Please check your files.")
                        return

                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
    else:
        st.warning("Please log in to use the application.")

if __name__ == '__main__':
    main()











