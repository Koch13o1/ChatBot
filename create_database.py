from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os
import shutil
from typing import List  # Import List for type hinting in Python 3.8

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()

CHROMA_PATH = "chroma"
DATA_PATH = "data/books"  # This is where your markdown file (e.g., alice_in_wonderland.md) is located

# Use a Hugging Face model for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # You can replace this with your preferred Hugging Face model

def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    embeddings = embed_documents(chunks)
    save_to_chroma(chunks, embeddings)

def load_documents():
    # This will load the .md file from the specified directory
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")
    return documents

def split_text(documents: List[Document]):  # Use List from typing for type hinting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    # Display a sample chunk for inspection
    document = chunks[10]  # This is just to check a chunk's content
    print(f"Sample chunk:\n{document.page_content}")
    print(f"Metadata: {document.metadata}")

    return chunks

def embed_documents(chunks: List[Document]):  # Use List from typing for type hinting
    # Extract text from the chunks and generate embeddings using SentenceTransformer
    texts = [chunk.page_content for chunk in chunks]
    embeddings = embedding_model.encode(texts, show_progress_bar=True)
    return embeddings

def save_to_chroma(chunks: List[Document], embeddings):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents and their embeddings
    db = Chroma.from_documents(
        chunks, embeddings, persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

if __name__ == "__main__":
    main()
