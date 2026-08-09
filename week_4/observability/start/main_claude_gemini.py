import os
import glob
from anthropic import Anthropic
from anthropic.types import TextBlockParam, Message, TextBlock
from pinecone import Pinecone
from typing import List
from langsmith import Client, traceable
from langsmith.wrappers import wrap_anthropic
import dotenv
from google import genai

dotenv.load_dotenv()

# Initialize OpenAI and Pinecone clients
client = Anthropic()
emb = genai.Client()
pc = Pinecone()

# LangSmith client setup
#langsmith_api_key = os.environ.get("LANGCHAIN_API_KEY")
langsmith_project = "rag-observability"
langsmith_client = wrap_anthropic(client)

# Constants
INDEX_NAME = "test"
EMBEDDING_MODEL = "gemini-embedding-2"
CHAT_MODEL = "claude-sonnet-5"

@traceable(name="load_documents")
def load_documents():
    """Load all text documents from the letters directory."""
    documents = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, "letters/*.txt")
    files = glob.glob(path)
    
    for file_path in files:
        with open(file_path, 'r') as file:
            content = file.read()
            documents.append({"content": content, "metadata": {"source": file_path}})
    
    print(f"Found {len(documents)} letters")
    return documents

@traceable(name="chunk_documents")
def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into smaller chunks for better processing."""
    chunks = []
    
    for doc in documents:
        content = doc["content"]
        metadata = doc["metadata"]
        
        # Simple text splitting - you can implement more sophisticated chunking if needed
        for i in range(0, len(content), chunk_size - chunk_overlap):
            if i > 0:
                start = i - chunk_overlap
            else:
                start = 0
                
            chunk_content = content[start:start + chunk_size]
            if chunk_content:
                chunks.append({"content": chunk_content, "metadata": metadata})
    
    return chunks

@traceable(name="get_embeddings")
def get_embeddings(texts: List[str]):
    """Generate embeddings for a list of texts using OpenAI."""
    response = emb.models.embed_content(
        contents=texts,
        model=EMBEDDING_MODEL,
        config=genai.types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY", output_dimensionality=1024)
    )
    return [embedding.values for embedding in response.embeddings]

@traceable(name="embed_documents")
def embed_documents(chunks, namespace):
    """Embed documents and store them in Pinecone."""
    # Get Pinecone index
    index = pc.Index(INDEX_NAME)
    
    # Prepare batches (Pinecone usually works well with batches of ~100)
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        chunk_batch = chunks[i:i+batch_size]
        
        # Get text from each chunk
        texts = [chunk["content"] for chunk in chunk_batch]
        
        # Get embeddings
        embeddings = get_embeddings(texts)
        
        # Prepare data for Pinecone
        vectors = []
        for j, embedding in enumerate(embeddings):
            vectors.append({
                "id": f"chunk_{i+j}",
                "values": embedding,
                "metadata": chunk_batch[j]["metadata"]
            })
        
        # Upsert to Pinecone
        index.upsert(vectors=vectors, namespace=namespace)

def search_documents(query, namespace, top_k=5):
    """Search the vector store with the user query."""
    # Get query embedding
    query_embedding = get_embeddings([query])[0]
    
    # Search Pinecone
    index = pc.Index(INDEX_NAME)
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True,
        include_values=False
    )
    
    # Get matched documents
    docs_with_scores = []
    for match in results["matches"]:
        # Load the document content based on the source file
        with open(match["metadata"]["source"], 'r') as f:
            content = f.read()
        docs_with_scores.append((content, match["score"]))
    
    return docs_with_scores

# TODO: Add traceable decorator to track this function in LangSmith
# Example: https://docs.smith.langchain.com/observability/how_to_guides/log_traces_to_project
def ask_openai(query, documents):
    """Ask OpenAI a question with context from the documents."""
    # Join all documents into a single context string
    context = "\n\n".join([doc for doc, _ in documents])

    system_messages: List[TextBlockParam] = [
        {
            "type":"text",
            "text":  "Provide an answer to the user's query about Berkshire Hathaway."
                              "Documents from the Berkshire Hathaway shareholder meetings will be provided."
                              "Use those documents to best answer the question."
        },
        {
            "type": "text",
            "text" : f"Documents: {context}"
        }
    ]



    # Create messages for ClaudeAI
    messages = [
        {"role": "user", "content": query}
    ]
    
    response = client.messages.create(
        model=CHAT_MODEL,
        max_tokens=1000,
        system=system_messages,
        messages=messages
    )
    
    return response

if __name__ == "__main__":

    # Step1 - Embed Documents
    # documents = load_documents()
    # embed_documents(documents, "chunks")

    # Step 2: Write a query
    user_query = "When did Berkshire Hathaway purchase it's first coke stock?" # Year: 1988
    #user_query = "Can you explain about some phenomenon about bacteria?"

    # Step 3: Check Pinecone for similar chunks
    docs_and_scores = search_documents(query=user_query, namespace="chunks")
    
    # Step 4: Put docs into prompt and send to OpenAI
    response = ask_openai(user_query, docs_and_scores)


    for text in response.content:
        if type(text) is TextBlock:
            print(f"{text.text}")

# Query: "When did Berkshire Hathaway purchase it's first coke stock?"
# Output Generated:
# Based on the 1988 shareholder letter, Berkshire Hathaway made its major purchase of Coca-Cola stock **during 1988**. The letter states:
#
# > "In 1988 we made major purchases of Federal Home Loan Mortgage Pfd. ('Freddie Mac') and Coca Cola. We expect to hold these securities for a long time."
#
# At that time, Berkshire held 14,172,500 shares of The Coca-Cola Company, with a cost basis of $592,540,000 and a market value of $632,448,000.
#
# Buffett also noted this reflected his general investment philosophy: "when we own portions of outstanding businesses with outstanding managements, our favorite holding period is forever." This suggests 1988 marked the beginning of what would become one of Berkshire's most famous and enduring investment positions.
