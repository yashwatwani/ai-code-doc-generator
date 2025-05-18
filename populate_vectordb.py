import chromadb
from sentence_transformers import SentenceTransformer
import json

# --- Configuration ---
CHROMA_DB_PATH = "./chroma_db"  # Path to store the local ChromaDB files
COLLECTION_NAME = "code_documentation_store"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' # A good, small, and fast model

# --- Initialize ChromaDB Client and Embedding Model ---
print(f"Initializing ChromaDB client with path: {CHROMA_DB_PATH}")
# Persistent client stores data on disk
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

print(f"Loading sentence transformer model: {EMBEDDING_MODEL_NAME}")
# Load a pre-trained sentence embedding model
# The first time you run this, it might download the model, which can take a few minutes.
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
print("Embedding model loaded.")

# --- Get or Create Collection ---
try:
    print(f"Getting or creating collection: {COLLECTION_NAME}")
    # You can specify your own embedding function if needed, but for SentenceTransformer,
    # ChromaDB can often handle it if you provide embeddings directly, or use its default.
    # For more control or different models, you might use `collection.add(embeddings=..., documents=..., metadatas=..., ids=...)`
    # Let's use ChromaDB's default embedding function for now, which uses SentenceTransformer by default for text.
    # Alternatively, we can explicitly pass our model as the embedding function.
    from chromadb.utils import embedding_functions
    # Using an explicit embedding function tied to our chosen sentence-transformer model
    # This ensures consistency if we use the same model for querying later.
    st_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=st_ef # Use our specific model
        # metadata={"hnsw:space": "cosine"} # Optional: specify distance metric
    )
    print(f"Collection '{COLLECTION_NAME}' ready.")
except Exception as e:
    print(f"Error with ChromaDB collection: {e}")
    exit()

# --- Load Data from JSON ---
try:
    with open("knowledge_base_data.json", 'r') as f:
        knowledge_base = json.load(f)
    print(f"Loaded {len(knowledge_base)} items from knowledge_base_data.json")
except FileNotFoundError:
    print("Error: knowledge_base_data.json not found. Please create it.")
    exit()
except json.JSONDecodeError:
    print("Error: Could not decode knowledge_base_data.json. Please check its format.")
    exit()

# --- Prepare Data for ChromaDB ---
documents_to_add = [] # The text content we want to embed and search against
metadatas_to_add = [] # Additional data associated with each document
ids_to_add = []       # Unique IDs for each document

for item in knowledge_base:
    # We will embed the code snippet for retrieval.
    # We store the golden_documentation in metadata to retrieve it alongside the code.
    documents_to_add.append(item["code_snippet"]) 
    metadatas_to_add.append({
        "language": item["language"],
        "golden_doc": item["golden_documentation"] 
        # Add any other relevant metadata, e.g., source, complexity_rating
    })
    ids_to_add.append(item["id"]) # Must be unique strings

# --- Add Data to Collection (Upsert allows adding or updating) ---
if documents_to_add:
    try:
        print(f"Adding/updating {len(documents_to_add)} documents to the collection...")
        collection.upsert(
            documents=documents_to_add,
            metadatas=metadatas_to_add,
            ids=ids_to_add
            # Embeddings will be generated automatically by the collection's embedding_function (st_ef)
        )
        print("Data successfully added/updated in ChromaDB.")
        
        # Verify count
        count = collection.count()
        print(f"Total documents in collection '{COLLECTION_NAME}': {count}")

    except Exception as e:
        print(f"Error adding data to ChromaDB: {e}")
else:
    print("No documents to add.")

print("Script finished.")

# --- Example Query (Optional, for testing) ---
if collection.count() > 0:
    print("\n--- Example Query ---")
    query_code = "def subtract(x, y):\n    return x - y" # Example new code snippet
    
    # No need to manually embed for querying if collection has embedding_function
    results = collection.query(
        query_texts=[query_code], # Text to find similar documents for
        n_results=1,             # Number of results to return
        include=['documents', 'metadatas', 'distances'] # What to include in results
    )
    
    print("Query Results for:", query_code)
    if results and results['ids'][0]:
        for i in range(len(results['ids'][0])):
            print(f"  ID: {results['ids'][0][i]}")
            print(f"  Distance: {results['distances'][0][i]}")
            print(f"  Retrieved Code: {results['documents'][0][i]}")
            print(f"  Retrieved Metadata: {results['metadatas'][0][i]}")
    else:
        print("  No results found or error in query structure.")