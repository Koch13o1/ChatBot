import argparse
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer  # Hugging Face for embeddings
from dotenv import load_dotenv
import os
from typing import List  # Import List for type hints


# from huggingface_hub import InferenceApi

# # Initialize the Hugging Face Inference API
# hf_api = InferenceApi(repo_id="gpt2", token=os.getenv("HUGGINGFACE_API_KEY"))

# # Generate a response using Hugging Face
# response = hf_api(inputs=prompt, max_length=100)
# response_text = response[0]['generated_text']

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

load_dotenv()
print("]]]]]]]]", os.getenv("OPENAI_API_KEY"))

# Defining a wrapper class for Hugging Face embeddings
class HuggingFaceEmbeddings:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Generate embeddings using Hugging Face's SentenceTransformer
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        # Generate embedding for a single query
        embedding = self.model.encode(text, show_progress_bar=False)
        return embedding.tolist()

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Prepare the DB with Hugging Face embeddings
    embedding_function = HuggingFaceEmbeddings()  # Use Hugging Face for embeddings
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    print("//////lenresults:", len(results))
    if len(results) == 0 or results[0][1] < 0.5:
        print(f"Unable to find matching results.")
        return

    # Construct the context text from the retrieved chunks
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    # Use OpenAI to generate the response
    model = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
    response_text = model.predict(prompt)

    # Extract sources from the metadata
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)

if __name__ == "__main__":
    main()