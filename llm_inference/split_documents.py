import uuid
import tiktoken
import pandas as pd
from pandarallel import pandarallel 
from langchain_text_splitters import TokenTextSplitter


def split_single_document(
    document: str,
    chunk_size: int = 4000,
    chunk_overlap: int = 1000,
    model_name: str = "gpt-4o",
    token_splitter: TokenTextSplitter = None
) -> list:
    """
    Splits a single document string into token chunks.

    Args:
        document (str): The document content to split.
        chunk_size (int): Token chunk size.
        chunk_overlap (int): Token chunk overlap.
        model_name (str): Model name for tiktoken encoding.
        token_splitter (TokenTextSplitter, optional): Pre-initialized TokenTextSplitter.

    Returns:
        list: List of text chunks.
    """
    if token_splitter is None:
        encoding_name = tiktoken.encoding_name_for_model(model_name)
        token_splitter = TokenTextSplitter(
            encoding_name=encoding_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    return token_splitter.split_text(document)


def split_documents(
    input_csv_path: str,
    output_csv_path: str = "",
    chunk_size: int = 4000,
    chunk_overlap: int = 1000,
    model_name: str = "gpt-4o",
    col_id: str = "doc_id",
    col_content: str = "crawled_markdown",
    num_workers: int = 8
) -> pd.DataFrame:
    """
    Reads a crawled data CSV, splits the 'crawled_markdown' column into token chunks,
    and returns a DataFrame with exploded chunks and new doc_id for each chunk.

    Args:
        csv_path (str): Path to the crawled data CSV.
        chunk_size (int): Token chunk size.
        chunk_overlap (int): Token chunk overlap.
        model_name (str): Model name for tiktoken encoding.
        col_content (str): Column name for content to be split.
        col_id (str): Column name for document ID.
        nb_workers (int): Number of workers for parallel processing.

    Returns:
        pd.DataFrame: DataFrame with split chunks and new doc_id.
    """
    pandarallel.initialize(progress_bar=True, nb_workers=num_workers)

    # Initialize the token splitter with the specified model's encoding
    encoding_name = tiktoken.encoding_name_for_model(model_name)
    token_splitter = TokenTextSplitter(
        encoding_name=encoding_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    df = pd.read_csv(input_csv_path)
    if col_id in df.columns:
        df = df.rename(columns={col_id: "org_id"})

    # Use split_single_document for each row
    df.rename(columns={col_content: f"org_{col_content}"}, inplace=True)
    df[col_content] = df[f"org_{col_content}"].parallel_apply(
        lambda doc: split_single_document(
            doc,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            model_name=model_name,
            token_splitter=token_splitter
        )
    )
    df_chunks = df.explode(col_content).reset_index(drop=True)
    df_chunks[col_id] = [str(uuid.uuid4()) for _ in range(len(df_chunks))]

    if output_csv_path:
        df_chunks.to_csv(output_csv_path, index=False)
        print(f"Split documents saved to {output_csv_path}")

    else:
        return df_chunks


if __name__ == "__main__":
    import fire
    fire.Fire(split_documents)