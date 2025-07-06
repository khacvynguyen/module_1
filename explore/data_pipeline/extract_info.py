import os
import json
import logging
import json_repair
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm_inference.llm_inference_base import ai_completion_with_backoff, ai_runall
from llm_inference.llm_response_parser import parse_llm_response

from data_pipeline.configs import GEMINI_MODEL, TEMPERATURE, MAX_TOKENS
from data_pipeline.prompts import (
    PROMPT_TO_SUMMARIZE_DOC, 
    PROMPT_TO_EXTRACT_KEY_INFO
)

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("extract_info")


def ensure_directories_exist(
    directories: List[str],
) -> None:
    """Create required directories if they don't exist."""
    for directory in directories:
        if not directory:
            continue
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Directory ready: {directory}")


def summarize_document(document_text: str, model: str = GEMINI_MODEL) -> str:
    """Generate a summary of the document."""
    try:
        prompt = PROMPT_TO_SUMMARIZE_DOC.format(company_context=document_text)
        response = ai_runall(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        return parse_llm_response(response, delimiter="<output>")

    except Exception as e:
        logger.error(f"Error summarizing document: {e}")
        raise


def extract_key_info(document_text: str, model: str = GEMINI_MODEL) -> str:
    """Extract key information from document."""
    try:
        # Fixed typo from conpany_context to company_context
        prompt = PROMPT_TO_EXTRACT_KEY_INFO.format(company_context=document_text)
        response = ai_completion_with_backoff(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        return parse_llm_response(response, delimiter="<output>")
    except Exception as e:
        logger.error(f"Error extracting key info: {e}")
        raise


def enrich_data(
    structured_data: dict,
    model: str,
    enriched_data_dir: str = "",
) -> Dict[str, Any]:
    """ Process a single raw crawled data file through the entire pipeline and save cleaned data as JSON.
    
    Args:
        structured_data (dict): Structured data from raw crawled data
        model (str): Model to use for inference
        enriched_data_dir (str, optional): Directory to save enriched data
        
    Returns:
        dict: Processed data and metadata
    """

    doc_id = structured_data.get("doc_id", "")
    logger.info(f"Processing {doc_id}...")

    try:
        cleaned_md = structured_data.get("cleaned_markdown", "")    
        if not cleaned_md:
            logger.warning(f"Empty or missing markdown")
            return {"success": False, "output": {"error": "Empty or missing markdown"}}

        # Summarize content
        summary = summarize_document(cleaned_md, model)

        # Extract key information
        key_info = extract_key_info(cleaned_md, model)

        # Step 6: Store cleaned document and metadata
        cleaned_metadata = dict(
            url=structured_data.get("url", ""),
            title=structured_data.get("metadata", {}).get("title", ""),
            description=structured_data.get("metadata", {}).get("description", ""),
            keywords=structured_data.get("metadata", {}).get("keywords", "")
        )

        doc_to_save = {
            "doc_id": doc_id,
            "url": structured_data.get("url", ""),
            "cleaned_markdown": cleaned_md,
            "metadata": cleaned_metadata,
            "summary": summary,
            "key_info": key_info
        }

        if enriched_data_dir and os.path.exists(enriched_data_dir):
            with open(f"{enriched_data_dir}/{doc_id}.json", "w") as f:
                json.dump(doc_to_save, f, indent=2)
            
        return {"success": True, "output": doc_to_save}
    
    except Exception as e:
        logger.error(f"Error processing {doc_id}: {e}", exc_info=True)
        return {"success": False, "output": {"error": str(e)}}


def parallel_enrich_data(
    cleaned_csv_path: str,
    enriched_csv_output: str,
    model: str = GEMINI_MODEL,
    batch_size: int = 2, 
    num_workers: int = 2,
    enriched_data_dir: str = "",
    debug: bool = False
) -> List[Dict[str, Any]]:
    """
    Process raw crawled data files in parallel batches to manage memory usage.
    
    Args:
        cleaned_csv_path: Path to cleaned data CSV file
        enriched_csv_output: Path to save enriched data CSV file
        model: (Optional) Model to use for inference
        batch_size: (Optional) Number of files to process in each batch
        num_workers: (Optional) Number of parallel workers to use
        enriched_data_dir: (Optional) Directory to save enriched data files
        debug: (Optional) Flag to run the pipeline only few samples for debugging
    
    Output:
        Saved extracted information to enriched_csv and local dir
        
    """
    
    # Check output directories
    ensure_directories_exist([enriched_data_dir])

    df_cleaned_data = pd.read_csv(cleaned_csv_path)
    df_cleaned_data["metadata"] = df_cleaned_data["metadata"].apply(json_repair.loads)
    doc_ids = df_cleaned_data["doc_id"].tolist()
    
    if debug:
        doc_ids = doc_ids[:4]
    
    total_files = len(doc_ids)
    logger.info(f"Found {total_files} files to process")
    
    if total_files == 0:
        logger.info("No files to process")
        return []
    
    all_results = []
    
    # Process in batches
    for i in range(0, total_files, batch_size):
        batch = doc_ids[i:i+batch_size]
        list_structured_data = df_cleaned_data[df_cleaned_data["doc_id"].isin(batch)].to_dict(orient="records")
        
        batch_num = i//batch_size + 1
        total_batches = (total_files + batch_size - 1) // batch_size
        logger.info(f"Processing batch {batch_num}/{total_batches}: {len(batch)} files with {num_workers} workers")
        
        # Process current batch in parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks for this batch
            future_to_enriched_data = {
                executor.submit(enrich_data, structured_data, model, enriched_data_dir): structured_data for structured_data in list_structured_data
            }
            
            # Process as they complete with progress bar
            batch_results = []
            with tqdm(total=len(batch), desc=f"Batch {batch_num}/{total_batches}") as progress:
                for future in as_completed(future_to_enriched_data):
                    structured_data = future_to_enriched_data[future]
                    doc_id = structured_data.get("doc_id", "")
                    try:
                        result = future.result()
                        batch_results.append(result)
                    except Exception as e:
                        logger.error(f"Exception processing {doc_id}: {e}", exc_info=True)
                        batch_results.append({
                            "success": False,  
                            "error": str(e)
                        })
                    finally:
                        progress.update(1)
        
        # Add batch results to overall results
        all_results.extend(batch_results)
        
        # Summarize batch results
        batch_success = sum(1 for r in batch_results if r.get("success", False))
        logger.info(f"Batch {batch_num}: Successfully processed {batch_success} out of {len(batch)} files")
    
    df_output = pd.DataFrame([r.get("output", {}) for r in all_results])
    df_output.to_csv(enriched_csv_output, index=False)
    
    # Summarize overall results
    success_count = sum(1 for r in all_results if r.get("success", False))
    logger.info(f"Overall: Successfully processed {success_count} out of {total_files} files")
    
    return None

if __name__ == "__main__":
    import fire
    fire.Fire(parallel_enrich_data)