
# qdrant_setup_partial.py
from pathlib import Path
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import uuid
import re

# Configuration
QDRANT_URL = "https://754e68dd-c297-4ab2-9833-c81cbfbfb75c.eu-west-2-0.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.kRiEKHQ5s4KUWoYJqhQ29tbmbgfqFT2jAAfgrPTshSM"
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

# New files to process
# IMPORTANT: Added the configuration for readme_blogs_latest here
NEW_COLLECTIONS = [
    {
        "file_path": "/home/ubuntu/nilesh-workspace-backup-20250707/Blog_generation/internal-llama-cookbook/end-to-end-use-cases/technical_blogger/Blog_generation/cookbook_metadata/mdfiles_latest.txt",
        "collection_name": "readme_blogs_latest"
    },
    {
        "file_path": "/home/ubuntu/nilesh-workspace-backup-20250707/Blog_generation/internal-llama-cookbook/end-to-end-use-cases/technical_blogger/Blog_generation/cookbook_metadata/3rd_party_integrations.txt",
        "collection_name": "3rd_party_integrations"
    },
    {
        "file_path": "/home/ubuntu/nilesh-workspace-backup-20250707/Blog_generation/internal-llama-cookbook/end-to-end-use-cases/technical_blogger/Blog_generation/cookbook_metadata/Getting_started_files.txt",
        "collection_name": "getting_started_files"
    }
]

def markdown_splitter(text, max_chunk=800):
    sections = re.split(r'(?=^#+ .*)', text, flags=re.MULTILINE)
    chunks = []
    current_chunk = []
    
    for section in sections:
        if len(''.join(current_chunk)) + len(section) > max_chunk:
            chunks.append(''.join(current_chunk))
            current_chunk = [section]
        else:
            current_chunk.append(section)
    
    if current_chunk:
        chunks.append(''.join(current_chunk))
    
    return [{"text": chunk, "header": f"section_{i}"} for i, chunk in enumerate(chunks)]

def get_qdrant_client():
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)

def process_file(config):
    client = get_qdrant_client()
    embedding_model = get_embedding_model()
    
    # Create collection if not exists
    if not client.collection_exists(config["collection_name"]):
        client.create_collection(
            collection_name=config["collection_name"],
            vectors_config=models.VectorParams(
                size=384,
                distance=models.Distance.COSINE
            )
        )
    
    # Process and store documents
    try:
        text = Path(config["file_path"]).read_text(encoding='utf-8')
        chunks = markdown_splitter(text)
        
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            points = []
            for chunk in batch:
                embedding = embedding_model.encode(chunk["text"]).tolist()
                points.append(
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embedding,
                        payload=chunk
                    )
                )
            client.upsert(collection_name=config["collection_name"], points=points)
        
        print(f"Processed {len(chunks)} chunks for {config['collection_name']}")
    except FileNotFoundError:
        print(f"Error: The file at {config['file_path']} was not found. Skipping collection setup.")

def setup_all_collections():
    for config in NEW_COLLECTIONS:
        process_file(config)
    print("All collections created and populated successfully!")

if __name__ == "__main__":
    setup_all_collections()