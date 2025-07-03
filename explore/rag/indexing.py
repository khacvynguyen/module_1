from typing import List, Dict, Any
from uuid import uuid4
from llm_inference.llm_inference_base import get_length
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter


def create_document(content:str, metadata: Dict[str, Any] = {}) -> Document:  
    return Document(page_content=content, metadata=metadata)


def create_documents(data: List[Dict[str, Any]]) -> List[Document]:
    return [create_document(doc["content"], doc["metadata"]) for doc in data]


def split_document(document: Document, chunk_size: int = 4000, chunk_overlap: int = 400) -> List[Document]:
    # TODO: use token split instead of character split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    chunked_documents = []

    splits = text_splitter.split_text(document.page_content)
    for i, split in enumerate(splits):
        new_doc = Document(
            page_content=split,
            metadata={
                **document.metadata,  # Copy original metadata
                "chunk_idx": i  # Add chunk number to metadata
            }
        )

        # re-init doc_id
        new_doc.metadata["doc_id"] = str(uuid4())
        new_doc.metadata["num_tokens"] = get_length(split)

        chunked_documents.append(new_doc)

    print(f"Number of chunked documents: {len(chunked_documents)}")

    return chunked_documents


def split_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    chunked_documents = []
    for doc in documents:
        splits = text_splitter.split_text(doc.page_content)
        for i, split in enumerate(splits):
            new_doc = Document(
                page_content=split,
                metadata={
                    **doc.metadata,  # Copy original metadata
                    "chunk_idx": i  # Add chunk number to metadata
                }
            )
            
            # re-init doc_id
            new_doc.metadata["parent_id"] = new_doc.metadata["doc_id"].get("doc_id", "")
            new_doc.metadata["doc_id"] = str(uuid4())
            
            chunked_documents.append(new_doc)

    print(f"Number of chunked documents: {len(chunked_documents)}")

    return chunked_documents
