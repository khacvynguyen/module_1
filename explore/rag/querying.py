import random
from typing import List, Dict, Any
from langchain_chroma import Chroma
from langchain_core.documents import Document
from sklearn.cluster import KMeans
import numpy as np


def get_random_documents(vectorstore, num_docs=5):
    try:
        # Get all document IDs from the vector store
        doc_ids = vectorstore._collection.get()['ids']

        # Randomly sample k document IDs
        sampled_doc_ids = random.sample(doc_ids, num_docs)

        # Retrieve the documents corresponding to the sampled IDs
        random_documents = vectorstore._collection.get(ids=sampled_doc_ids)['documents']
        sampled_docs = [Document(page_content=doc) for doc in random_documents]

    except Exception as e:
        print(f"An error occurred: {e}")
        return []

    return sampled_docs

