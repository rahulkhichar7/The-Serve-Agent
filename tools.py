import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List

def collect_logs(path: str) -> List[str]:
    logs = []
    with open(path, 'r') as file:
        for line in file:
            logs.append(line.strip())
    return logs

def create_or_get_cromadb(logs: List[str]): #create_cromadb always tries to create a new collection. If the DB already exists, that call will throw an error because the collection name is taken

    client = chromadb.PersistentClient(path="./vector_databases/chroma_db")
    try:
        collection = client.get_collection(name = "monitoring_logs")
    except:
        collection = client.create_collection(name = "monitoring_logs")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(logs)

        for i, log in enumerate(logs):
            collection.add(
                ids = str(i),
                documents = [log],
                embeddings = [embeddings[i]]
            )
    return collection

"""
def load_cromadb():
    client = chromadb.PersistentClient(path="./vector_databases/chroma_db")
    collection = client.get_collection(name="monitoring_logs")
    return collection 
        """

def search_query(query: str, collection, n_results: int = 3):
    result = collection.query(
        query_texts = [query],
        n_results = n_results
    )
    return result
    
    



