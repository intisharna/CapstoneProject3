import json
import os
import pandas as pd
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

load_dotenv()

QDRANT_COLLECTION_NAME = "imdb_movies"

data = pd.read_csv("./imdb_top_1000.csv")

docs: list[Document] = []

for _, row in data.iterrows():
    page_content = f"Title: {row.get('Series_Title', '')}\nOverview: {row.get('Overview', '')}"
    metadata = {
        "title": str(row.get("Series_Title", "")), 
        "year": int(row["Released_Year"]) if pd.notna(row.get("Released_Year")) and str(row.get("Released_Year")).isdigit() else None,
        "genre": str(row.get("Genre", "")).split(', '),
        "rating": float(row["IMDB_Rating"]) if pd.notna(row.get("IMDB_Rating")) else None,
        "director": str(row.get("Director", "")),
        "star1": str(row.get("Star1", "")),
    }
    metadata = {k: v for k, v in metadata.items() if v is not None}

    docs.append(
        Document(
            page_content=page_content,
            metadata=metadata,
        )
    )

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""],
)
docs = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", 
    api_key=os.getenv("OPENAI_API_KEY"),
)
vector_client = QdrantClient(
    api_key=os.getenv("QDRANT_API_KEY"),
    url=os.getenv("QDRANT_URL"),
)

vector_client.delete_collection(collection_name=QDRANT_COLLECTION_NAME)

vector_client.create_collection(
    collection_name=QDRANT_COLLECTION_NAME,
    vectors_config=qm.VectorParams(
        size=1536,
        distance=qm.Distance.COSINE
    ),
)

vector_client.create_payload_index(
    collection_name=QDRANT_COLLECTION_NAME,
    field_name="metadata.year",
    field_schema="integer",
)

vector_client.create_payload_index(
    collection_name=QDRANT_COLLECTION_NAME,
    field_name="metadata.genre",
    field_schema="keyword", 
)

vector_client.create_payload_index(
    collection_name=QDRANT_COLLECTION_NAME,
    field_name="metadata.rating",
    field_schema="float",
)

vector_client.create_payload_index(
    collection_name=QDRANT_COLLECTION_NAME,
    field_name="metadata.director",
    field_schema="text", 
)

vector_db = QdrantVectorStore(
    client=vector_client,
    embedding=embeddings,
    collection_name=QDRANT_COLLECTION_NAME,
)
print(f"Adding {len(docs)} documents to collection '{QDRANT_COLLECTION_NAME}'...")
vector_db.add_documents(docs)
print("Done adding documents.")