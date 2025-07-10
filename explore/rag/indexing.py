from typing import List, Dict, Any, Tuple
from uuid import uuid4
from llm_inference.llm_inference_base import get_length
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter


def create_document(content:str, metadata: Dict[str, Any] = {}) -> Document:  
    return Document(page_content=content, metadata=metadata)


def create_documents(data: List[Dict[str, Any]]) -> List[Document]:
    return [create_document(doc["content"], doc["metadata"]) for doc in data]


def split_document(
    document: Document,
    split_on: List[Tuple[str, str]] = [("#", "Header 1"), ("##", "Header 2")],
    **kwargs
) -> List[Document]:
    # TODO: use token split instead of character split
    text_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=split_on,
        **kwargs
    )

    chunked_documents = text_splitter.split_text(document.page_content)

    print(f"Number of chunked documents: {len(chunked_documents)}")

    return chunked_documents


def split_documents(
    documents: List[Document],
    split_on: List[Tuple[str, str]] = [("#", "Header 1"), ("##", "Header 2")],
    **kwargs
) -> List[Document]:
    text_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=split_on,
        **kwargs
    )

    chunked_documents = []
    for doc in documents:
        splits = text_splitter.split_text(doc.page_content)

        chunked_documents.extend(splits)

    print(f"Number of chunked documents: {len(chunked_documents)}")

    return chunked_documents
