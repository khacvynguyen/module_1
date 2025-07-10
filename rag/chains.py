from typing import List, Dict, Any
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser, StructuredOutputParser, PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_chroma import Chroma

from pydantic import BaseModel, Field

from utils import clean_text
from querying import retrieve_cluster_and_resampling

def format_docs(docs: List[Document|str]) -> str:
    for doc in docs:
        if isinstance(doc, Document):
            context += "\n\n" + doc.page_content
        elif isinstance(doc, str):
            context += "\n\n" + doc
        
    return context.strip()


def log_prompt(prompt):
    print("\n--- Prompt Fed to LLM ---")
    try:
        print(prompt.messages[0].content)
    except Exception as e:
        print(e)
        print(prompt)
    print("\n--- End Prompt ---")
    # print(prompt)
    return prompt


def query_rewrite_chain(llm: GoogleGenerativeAI | Any, rewrite_template: str):

    rewrite_prompt = ChatPromptTemplate.from_template(rewrite_template)

    query_rewriter = rewrite_prompt | log_prompt | llm | StrOutputParser() | clean_text

    return query_rewriter


def rewrite_and_resampling_chain(
    llm: GoogleGenerativeAI | Any,
    vectorstore: Chroma,
    rewrite_template: str,
    response_template: str,
    **kwargs
):
    
    query_rewriter = query_rewrite_chain(llm, rewrite_template)
    
    prompt_resp = ChatPromptTemplate.from_template(response_template)
    
    query_func = lambda query_str: retrieve_cluster_and_resampling(query_str, vectorstore, **kwargs)
    
    chain = (
        {
            "context": {"x": RunnablePassthrough()} | query_rewriter | query_func | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt_resp
        | log_prompt
        | llm
        | StrOutputParser()
    )

    return chain 