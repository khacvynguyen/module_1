import os
import logging
import pandas as pd
from pandarallel import pandarallel

from llm_inference.llm_inference_base import ai_runall
from llm_inference.llm_response_parser import parse_llm_response

from data_pipeline.configs import GEMINI_MODEL, TEMPERATURE, MAX_TOKENS
from data_pipeline.prompts import PROMPT_TO_CLEAN_DOC

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("preprocess_doc")


def clean_document(document_text: str, model: str = GEMINI_MODEL) -> str:
    """Use LLM to clean document."""
    try:
        response = ai_runall(
            model=model,
            messages=[
                {"role": "system", "content": PROMPT_TO_CLEAN_DOC},
                {"role": "user", "content": document_text}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        
        parsed_response = parse_llm_response(response, delimiter="<output>")
        
        return parsed_response
    except Exception as e:
        logger.error(f"Error cleaning document: {e}")
        raise


def preprocess_document(
    crawled_data_path: str,
    cleaned_csv_output: str,
    model: str = GEMINI_MODEL,
    num_workers: int = 8,
    cleaned_md_dir: str = "",
    debug: bool = False
):
    """ Preprocess crawled data using LLM.

    Args:
        crawled_data_path (str): Path to crawled data.
        cleaned_csv_output (str): Path to save cleaned data as CSV file.
        model (str, optional): LLM model to use. Defaults to GEMINI_MODEL.
        num_workers (int, optional): Number of workers to use. Defaults to 8.
        cleaned_md_dir (str, optional): Directory to save cleaned markdown files. Defaults to "".
        debug (bool, optional): Debug mode. Defaults to False.

    Raises:
        ValueError: If 'crawled_markdown' column not found in input data.
    """
    
    pandarallel.initialize(nb_workers=num_workers, progress_bar=True)
    
    # Load raw data
    df_crawled_data = pd.read_csv(crawled_data_path)
    
    if debug:
        df_crawled_data = df_crawled_data.sample(4)
    
    # Use LLM to clean document further
    if "crawled_markdown" not in df_crawled_data.columns:
        raise ValueError("Column 'crawled_markdown' not found in input data.")
    
    df_crawled_data["cleaned_markdown"] = df_crawled_data["crawled_markdown"].parallel_apply(
        lambda x: clean_document(x, model)
    )
    df_crawled_data.to_csv(cleaned_csv_output, index=False)
    
    print("Saved cleaned data to:", cleaned_csv_output)

    if cleaned_md_dir:
        os.makedirs(cleaned_md_dir, exist_ok=True)
        for _, row in df_crawled_data.iterrows():
            doc_id = row["doc_id"]
            llm_cleaned_md = row["cleaned_markdown"]
            with open(f"{cleaned_md_dir}/{doc_id}.md", "w") as f:
                f.write(llm_cleaned_md)
        

if __name__ == "__main__":
    import fire
    fire.Fire(preprocess_document)
    