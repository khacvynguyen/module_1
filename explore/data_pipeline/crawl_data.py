import asyncio
import json
import logging
import os
import uuid
from typing import List, Dict, Optional, Union, Any

import pandas as pd
from tqdm import tqdm

from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_url(url: str) -> str:
    """Clean a URL by removing query parameters, anchors, and trailing slashes."""
    return url.split('?')[0].split('#')[0].rstrip('/')


def get_crawler_config(**kwargs) -> CrawlerRunConfig:
    """Create a crawler configuration with sensible defaults."""
    return CrawlerRunConfig(
        cache_mode=CacheMode.DISABLED,
        exclude_external_images=True,
        exclude_external_links=True,
        excluded_tags=["nav", "footer", "header"],
        markdown_generator=DefaultMarkdownGenerator(
            content_filter=PruningContentFilter(threshold=0.4),
            options={"ignore_links": True, "ignore_images": True}
        ),
        **kwargs
    )


async def markdown_crawl(url: str, outfile: Optional[str] = None) -> Dict[str, Any]:
    """Crawl a URL and convert the content to markdown."""
    if outfile:
        os.makedirs(os.path.dirname(outfile), exist_ok=True)

    config = get_crawler_config()
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url, config=config)

    if outfile:
        # Use a single file operation for the base JSON
        with open(outfile, 'w', encoding="utf-8") as f:
            f.write(result.model_dump_json(indent=2))

        # Save the markdown versions
        base_path = outfile.replace('.json', '')
        with open(f"{base_path}.md", 'w', encoding="utf-8") as f:
            f.write(result.markdown)

        with open(f"{base_path}_cleaned.md", 'w', encoding="utf-8") as f:
            f.write(result.markdown_v2.fit_markdown)

        logger.info(f"Crawled data saved to {outfile}")

    else:
        return result


async def simple_crawl(url: str) -> Any:
    """Perform a simple crawl without markdown conversion."""
    crawler_run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
    async with AsyncWebCrawler() as crawler:
        return await crawler.arun(url=url, config=crawler_run_config)


async def parallel_markdown_crawl(
    urls: List[str], 
    workers: int = 8, 
    crawler_config: Optional[CrawlerRunConfig] = None
) -> List[Any]:
    """Crawl multiple URLs in parallel with a limited number of workers."""
    if not crawler_config:
        crawler_config = get_crawler_config()

    total_batches = (len(urls) + workers - 1) // workers
    results = []

    async with AsyncWebCrawler() as crawler:
        for i in tqdm(range(0, len(urls), workers), desc="Crawling batches"):
            batch_num = i // workers + 1
            logger.info(f'Crawling batch {batch_num}/{total_batches}')

            # Get batch of URLs and create crawl tasks
            batch = urls[i:i + workers]
            tasks = [crawler.arun(url, config=crawler_config) for url in batch]

            # Run crawl tasks and collect results
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out exceptions
            valid_results = [r for r in batch_results if not isinstance(r, Exception)]
            if len(valid_results) != len(batch_results):
                logger.warning(f"Failed to crawl {len(batch_results) - len(valid_results)} URLs in batch {batch_num}")
    
            results.extend(valid_results)
            await asyncio.sleep(1)  # Rate limiting

    return results


def craw_data(
    input_urls_file: str,
    output_file: str,
    num_workers: int = 8,
    folder_to_save: str = "",
    debug: bool = False
) -> None:
    """
    Run the crawler on a list of URLs from a file and save the results.
    Args:
        input_urls_file: Path to a JSON or CSV file with a list of URLs to crawl.
        num_workers: Number of workers to use for parallel crawling.
        output_file: Path to save the output CSV file with the crawled data.
        folder_to_save: Directory to save the crawled data.
        
    Output:
        A CSV file with the crawled data and a JSON file for each crawled URL.

    """
    
    # Create output directory if it doesn't exist
    if folder_to_save:
        os.makedirs(folder_to_save, exist_ok=True)

    # Load URLs from file
    if input_urls_file.endswith('.json'):
        with open(input_urls_file, 'r') as f:
            links_dict = json.load(f)
        urls = list(links_dict.keys())

    elif input_urls_file.endswith('.csv'):
        input_csv = pd.read_csv(input_urls_file)
        urls = input_csv['href'].tolist()

    else:
        raise ValueError(f"Unsupported file format: {input_urls_file}")

    if debug:
        urls = urls[:10]
    
    logger.info(f'Number of URLs to crawl: {len(urls)}')
    results = asyncio.run(parallel_markdown_crawl(urls, workers=num_workers))

    # Save results
    output_data = []
    
    for idx, result in enumerate(results):
        # convert CrawlResult to dict
        result_dict = result.model_dump(mode="json")

        if idx == 0 and debug:
            logger.info(f"Example result: {result_dict}")

        doc_id = str(uuid.uuid4())
        file_name = f"{doc_id}.json"
        file_path = os.path.join(folder_to_save, file_name)
        result_dict["doc_id"] = doc_id
        result_dict["local_path"] = file_path
        output_data.append(dict(
            doc_id=doc_id,
            local_path=file_path,
            url=result_dict.get("url", ""),
            crawled_markdown=result_dict.get("markdown_v2", {}).get("fit_markdown", ""),
            metadata=result_dict.get("metadata", {})
        ))

        if folder_to_save:
            with open(file_path, 'w', encoding="utf-8") as f:
                f.write(json.dumps(result_dict, indent=2))
            
    # Save output CSV
    output_csv = pd.DataFrame(output_data)
    output_csv.to_csv(output_file, index=False)

if __name__ == '__main__':
    import fire
    fire.Fire(craw_data)