import time
import tempfile
import os
from io import BytesIO

import streamlit as st
import pylatexenc
import pylatexenc.latex2text
import textract
import docx
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from textract import process
from bs4 import BeautifulSoup
from PIL import Image
import pytesseract

user_template = "<div style='background-color: #2D3250; padding: 10px; border-radius: 5px; margin-bottom: 5px;'>User: {{MSG}}</div>"
bot_template = "<div style='background-color: #5C8374; padding: 10px; border-radius: 5px; margin-bottom: 5px;'>Bot: {{MSG}}</div>"

def send_request_with_rate_limiting():
    # Introduce a delay between requests
    time.sleep(5)  # Adjust the delay (in seconds) as needed

# Modify the get_vectorstore function to include rate limiting
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    send_request_with_rate_limiting()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Modify the get_conversation_chain function to include rate limiting
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    send_request_with_rate_limiting()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_tex_text(tex_files):
    text = ""
    for tex_file in tex_files:
        # Save the uploaded TEX file to a temporary directory
        temp_dir = tempfile.TemporaryDirectory()
        tex_path = os.path.join(temp_dir.name, tex_file.name)
        with open(tex_path, 'wb') as f:
            f.write(tex_file.read())
        
        # Parse TEX file and extract text
        with open(tex_path, 'r', encoding='utf-8') as f:
            tex_content = f.read()
            text += pylatexenc.latex2text.latex2text(tex_content)
        
        # Clean up temporary directory
        temp_dir.cleanup()
    return text

def get_docx_text(docx_files):
    text = ""
    for docx_file in docx_files:
        doc = docx.Document(docx_file)
        for paragraph in doc.paragraphs:
            text += paragraph.text
    return text


def get_html_text(html_files):
    text = ""
    for html_file in html_files:
        # Convert UploadedFile object to bytes
        content = html_file.getvalue()
        
        # Parse HTML content using BeautifulSoup
        soup = BeautifulSoup(content, 'html.parser')
        
        # Extract text from HTML
        text += soup.get_text()
        
    return text

def get_image_text(image_files):
    text = ""
    for image_file in image_files:
        img = Image.open(image_file)
        extracted_text = pytesseract.image_to_string(img)
        text += extracted_text + "\n"  # Append extracted text from each image
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=512,
        chunk_overlap=24,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    st.header("Chat with Scientific Documents")
    user_question = st.text_input("Ask a question:")
    
    # Initialize conversation if not already initialized
    if 'conversation' not in st.session_state:
        st.session_state.conversation = {}
    
    if user_question:
        handle_userinput(user_question)

    # file_types = st.multiselect("Select file types:", ["PDF", "TeX", "DOCX", "HTML", "Image"])
    file_types = st.selectbox("Select file type:", ["PDF", "TeX", "DOCX", "HTML", "Image"])

    if "PDF" in file_types:
        pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True)

    if "TeX" in file_types:
        tex_files = st.file_uploader("Upload TeX files", accept_multiple_files=True)

    if "DOCX" in file_types:
        docx_files = st.file_uploader("Upload DOCX files", accept_multiple_files=True)

    if "HTML" in file_types:
        html_files = st.file_uploader("Upload HTML files", accept_multiple_files=True)

    if "Image" in file_types:
        image_files = st.file_uploader("Upload Images", accept_multiple_files=True)

    if st.button("Process"):
        with st.spinner("Processing"):
            text = ""
            if "PDF" in file_types:
                text += get_pdf_text(pdf_docs)
            if "TeX" in file_types:
                text += get_tex_text(tex_files)
            if "DOCX" in file_types:
                text += get_docx_text(docx_files)
            if "HTML" in file_types:
                text += get_html_text(html_files)
            if "Image" in file_types:
                text += get_image_text(image_files)
            
            text_chunks = get_text_chunks(text)
            vectorstore = get_vectorstore(text_chunks)
            st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()
