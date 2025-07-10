import glob
import os
import time
import uuid
import json
import pandas as pd
import google.generativeai as genai
from google.generativeai.types import file_types
from google.generativeai import types
from pypdf import PdfReader, PdfWriter
from dotenv import load_dotenv
from typing import List
from pandarallel import pandarallel
from llm_inference.llm_inference_base import llm_completion_with_backoff, get_length
from data_pipeline.prompts import PROMPT_TO_CONVERT_DOC_TO_MARKDOWN

load_dotenv()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

DEFAULT_TEMPERATURE = 0.95
DEFAULT_MAX_TOKENS = 60000


def upload_to_gemini(path, mime_type=None) -> file_types.File:
    """Uploads the given file to Gemini.

    See https://ai.google.dev/gemini-api/docs/prompting_with_media
    """
    file = genai.upload_file(path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file


def wait_for_files_active(files: List[file_types.File]):
    """Waits for the given files to be active.

    Some files uploaded to the Gemini API need to be processed before they can be
    used as prompt inputs. The status can be seen by querying the file's "state"
    field.

    This implementation uses a simple blocking polling loop. Production code
    should probably employ a more sophisticated approach.
    """
    print("Waiting for file processing...")
    for name in (file.name for file in files):
        file = genai.get_file(name)
        while file.state.name == "PROCESSING":
            print(".", end="", flush=True)
            time.sleep(10)

        file = genai.get_file(name)
        if file.state.name != "ACTIVE":
            raise Exception(f"File {file.name} failed to process")

    print("...all files ready")
    print()


def split_pdf(input_pdf, output_folder, num_pages=5, overlap=1):
    """
    Splits a PDF file into smaller chunks with a specified number of pages and overlap.
    Args:
        input_pdf (str): Path to the input PDF file to be split.
        output_folder (str): Directory where the split PDF files will be saved.
        num_pages (int, optional): Number of pages per chunk. Defaults to 5.
        overlap (int, optional): Number of overlapping pages between consecutive chunks. Defaults to 1.
    Returns:
        list: A list of file paths to the generated PDF chunks.
    Raises:
        FileNotFoundError: If the input PDF file does not exist.
        ValueError: If `num_pages` is less than or equal to 0, or if `overlap` is negative.
    Notes:
        - The function ensures that the output folder exists by creating it if necessary.
        - The last chunk may contain fewer pages if the total number of pages is not evenly divisible.
        - Overlap ensures that the last `overlap` pages of one chunk are included at the beginning of the next chunk.
    """
    
    
    # Check if the output folder exists, if not, create it
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    
    # Open the input PDF file
    pdf_reader = PdfReader(input_pdf)
    total_pages = len(pdf_reader.pages)
    base_name = os.path.splitext(os.path.basename(input_pdf))[0]
    processed_pdf_paths = []
    
    # Iterate through the pages in chunks, accounting for overlap
    chunk_num = 1
    for start_page in range(0, total_pages, num_pages - overlap):
        pdf_writer = PdfWriter()
        # Ensure end_page doesn't exceed total_pages
        end_page = min(start_page + num_pages, total_pages)
        
        # If this would create an empty document or one with only overlap pages, break
        if start_page >= total_pages:
            break
        
        # Add pages to this chunk
        for page_num in range(start_page, end_page):
            pdf_writer.add_page(pdf_reader.pages[page_num])

        # Save the chunk to a new PDF file
        output_pdf = f"{output_folder}/{base_name}-chunk_{chunk_num:02d}.pdf"
        with open(output_pdf, 'wb') as output_file:
            pdf_writer.write(output_file)

        processed_pdf_paths.append(output_pdf)
        print(f"Document {chunk_num} saved as {output_pdf} (pages {start_page+1}-{end_page})")

        chunk_num += 1
    
    return processed_pdf_paths


def process_single_file(input_pdf_path: str, output_path: str):
    """
    Processes a single PDF file by uploading it to a remote service, converting it to markdown format
    using an AI model, and saving the resulting markdown content to a specified output file.
    Args:
        input_pdf_path (str): The file path of the input PDF to be processed.
        output_path (str): The file path where the resulting markdown content will be saved.
    Steps:
        1. Uploads the input PDF file to the Gemini service.
        2. Waits for the uploaded file to become active.
        3. Uses an AI model to convert the document to markdown format.
        4. Writes the markdown content to the specified output file.
    Raises:
        Any exceptions raised during file upload, AI processing, or file writing will propagate to the caller.
    Note:
        - The AI model used for conversion is specified as "gemini-2.0-flash".
        - The conversion process relies on a predefined prompt and configuration parameters.
    """
    

    uploaded_file = upload_to_gemini(input_pdf_path)
    
    wait_for_files_active([uploaded_file])

    try:
        # Calling the AI model to convert the document to markdown
        markdown_content = llm_completion_with_backoff(
            messages=[
                {"role": "user", "content": [
                    uploaded_file,
                    PROMPT_TO_CONVERT_DOC_TO_MARKDOWN
                ]}
            ],
            model="gemini-2.0-flash",
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS,
        )["content"]
        
        try:
            with open(output_path, "w") as f:
                f.write(markdown_content)
        except FileNotFoundError:
            print(f"File {output_path} not found")
            file_id = str(uuid.uuid4())
            print(f"Saving to current directory with file id {file_id}")
            with open(f"{file_id}.md", "w") as f:
                f.write(markdown_content)
                
            print(f"Markdown content saved to {output_path}")
    except Exception as e:
        print("fail to process file")
        print("save uploaded file to json file")
        with open(f"{str(uuid.uuid4())}.json", "w") as f:
            json.dump(uploaded_file.to_dict(), f)


def process_files(file_paths: List[str], output_csv_path: str, num_workers: int = 4):
    """
    Processes a list of file paths, uploads them to Google Gemini, converts their content 
    to cleaned markdown using AI, and saves the results to a CSV file.
    Args:
        file_paths (List[str]): A list of file paths to be processed.
        output_csv_path (str): The path to save the output CSV file containing the processed data.
        num_workers (int, optional): The number of parallel workers to use for processing. Defaults to 4.
    Returns:
        None
    """
    
    pandarallel.initialize(nb_workers=num_workers, progress_bar=True)
    
    # Create a DataFrame to hold the file paths and their corresponding markdown content
    df = pd.DataFrame({"local_path": file_paths})
    
    # Create a unique ID for each file
    df["doc_id"] = df["local_path"].apply(lambda x: str(uuid.uuid4()))
    
    # upload the file to Google Gemini
    df["goolge_file"] = df["local_path"].apply(lambda x: upload_to_gemini(x))
    
    # Wait for the files to be processed
    wait_for_files_active(df["goolge_file"].tolist())

    df["url"] = df["goolge_file"].apply(lambda x: x.uri)
    df["metadata"] = df["goolge_file"].apply(lambda x: x.to_dict())
    
    df["cleaned_markdown"] = df.parallel_apply(
        lambda row: llm_completion_with_backoff(
            messages=[
                {"role": "user", "content": [
                    row["goolge_file"],
                    PROMPT_TO_CONVERT_DOC_TO_MARKDOWN
                ]}
            ],
            model="gemini-2.0-flash",
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS,
        )["content"],
        axis=1,
    )
    
    # Save the cleaned markdown content to a CSV file
    df.drop(columns=["goolge_file"], inplace=True)
    
    df.to_csv(output_csv_path, index=False)
    
    return None


def segment_and_process_file(input_pdf_path: str, output_csv_path: str, output_folder: str, num_pages: int = 5, overlap: int = 1):
    """ 
    Splits a PDF file into smaller chunks and processes each chunk to extract
    
    Args:
        input_pdf_path (str): Path to the input PDF file.
        output_csv_path (str): Path to save the output CSV file.
        output_folder (str): Folder to save the split PDF chunks.
        num_pages (int): Number of pages per chunk.
        overlap (int): Number of overlapping pages between chunks.
    """

    # Split the PDF into smaller chunks
    chunk_paths = split_pdf(input_pdf_path, output_folder, num_pages, overlap)
    
    # Process each chunk and save the markdown content
    process_files(
        file_paths=chunk_paths,
        output_csv_path=output_csv_path,
        num_workers=4
    )


if __name__ == "__main__":
    import fire
    fire.Fire()