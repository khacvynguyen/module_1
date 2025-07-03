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


def random_sampling_with_retriver(query, retriever, num_samples=3):
    # Use the retriever to fetch more documents than needed
    docs = retriever.get_relevant_documents(query)  # "random" is a dummy query
    # Randomly sample from retrieved documents
    sampled_docs = random.sample(docs, min(num_samples, len(docs)))
    return sampled_docs


def retrieve_cluster_and_resampling(query_str: str, vectorstore: Chroma, **kwargs):

    NUM_DOCS = kwargs.get("num_docs", 30)
    NUM_CLUSTERS = kwargs.get("num_cluster", 3)
    SAMPLES_PER_CLUSTER = kwargs.get("sample_per_cluster", 2)
    RANDOM_STATE = kwargs.get("random_state", 3)

    # # query rewriter
    # rewriter = rewrite_prompt | log_prompt | llm | StrOutputParser()

    # retrieve
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": NUM_DOCS}
    )
    relevant_docs = retriever.invoke(query_str)

    doc_ids = [doc.id for doc in relevant_docs]
    doc_embeddings = vectorstore.get(
        ids=doc_ids,
        include=['embeddings']
    )['embeddings']

    # print(doc_embeddings[0].shape)
    # cluster
    print("Perform K-Means clustering")
    kmeans = KMeans(
        n_clusters=NUM_CLUSTERS,
        random_state=RANDOM_STATE
    )

    kmeans.fit(doc_embeddings)
    labels = kmeans.labels_

    # resampling
    sampled_doc_ids = []
    for cluster_id in range(NUM_CLUSTERS):
        cluster_indices = np.where(labels == cluster_id)[0]
        if len(cluster_indices) > 0:
            sampled_indices = np.random.choice(
                cluster_indices,
                size=min(SAMPLES_PER_CLUSTER, len(cluster_indices)),
                replace=False
            )
            sampled_doc_ids.extend([doc_ids[i] for i in sampled_indices])

    print("Sampled doc IDs:", sampled_doc_ids)

    resampled_docs = vectorstore.get(ids=sampled_doc_ids)["documents"]

    return resampled_docs