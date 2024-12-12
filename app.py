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

# User authentication setup
names = ["User1", "User2"]
usernames = ["user1", "user2"]
passwords = ["pass1", "pass2"]

# Hash passwords using SHA256
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

hashed_passwords = [hash_password(password) for password in passwords]

# Check if the input password matches the stored hash
def authenticate(username, password):
    if username in usernames:
        user_index = usernames.index(username)
        return hashed_passwords[user_index] == hash_password(password)
    return False

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
    llm = ChatOpenAI()
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
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()  # Ensure this loads your .env file if needed
    st.set_page_config(page_title="Chat with your Documents", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = None

    # Update sidebar title based on authentication status
    if st.session_state.authenticated:
        st.sidebar.title(f"Account ({st.session_state.username})")
    else:
        st.sidebar.title("Account")

    if st.session_state.authenticated:
        st.sidebar.success(f"Welcome, {st.session_state.username}!")
        logout_button = st.sidebar.button("Logout")
        if logout_button:
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.conversation = None
            st.session_state.chat_history = []
            st.sidebar.warning("You have been logged out.")
    else:
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

    if st.session_state.authenticated:
        if "conversation" not in st.session_state:
            st.session_state.conversation = None
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        st.header("Chat with your Documents :books:")
        user_question = st.text_input("Ask a question about your documents:")
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

                    st.write("Raw text extracted from documents:", raw_text)

                    text_chunks = get_text_chunks(raw_text)
                    if not text_chunks:
                        st.error("No text chunks created. Check the text processing logic.")
                        return

                    st.write("Text chunks created:", text_chunks)

                    vectorstore = get_vectorstore(text_chunks)
                    st.write("Vectorstore created successfully.")

                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.write("Conversation chain initialized successfully.")
    else:
        st.warning("Please log in to use the application.")

if __name__ == '__main__':
    main()
