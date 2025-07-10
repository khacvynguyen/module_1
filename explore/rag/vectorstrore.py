import chromadb
from typing import List, Dict, Any
from uuid import uuid4
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document


def get_collection_names(db_path: str) -> List[str]:
    persistent_client = chromadb.PersistentClient(
        path=db_path
    )

    # Retrieve and list all collection names
    collections = persistent_client.list_collections()
    
    coll_names = [
        collection if isinstance(collection, str) else
        collection.name for collection in collections
    ]

    return coll_names


def load_vectorstore(
    db_path: str,
    coll_name: str,
    embeddings: GoogleGenerativeAIEmbeddings | Any
) -> Chroma:
    """ Load chroma vectorstore """
    vectorstore = Chroma(
        persist_directory=db_path, 
        collection_name=coll_name, 
        embedding_function=embeddings
    )

    return vectorstore



def initialize_vectorstore(documents, embeddings, db_path, collection_name, force_create=False, **kwargs):
    # Check if the collection already exists

    existing_collections = get_collection_names(db_path=db_path)

    if collection_name in existing_collections and not force_create:
        print(f"Loaded existing collection: {collection_name} from {db_path}")
        return load_vectorstore(db_path, collection_name, embeddings)

    print(f"Creating new collection: {collection_name} in {db_path}")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=db_path,
        collection_name=collection_name,
        **kwargs
    )

    return vectorstore
