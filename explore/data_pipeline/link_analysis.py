import asyncio
import json
import logging
import os
from typing import Dict, List, Any, Optional

from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig
from tqdm import tqdm

# Configure logging properly
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def clean_url(url: str) -> str:
    """Clean a URL by removing query parameters, anchors, and trailing slashes."""
    return url.split('?')[0].split('#')[0].rstrip('/')


async def link_analysis(url: str) -> Any:
    """Analyze links from a given URL."""
    config = CrawlerRunConfig(
        cache_mode=CacheMode.ENABLED,
        exclude_external_links=False,
        exclude_social_media_links=True,
    )
    
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url, config=config)
    return result


async def process_link(
    link: Dict[str, Any], 
    urls_dict: Dict[str, Any], 
    crawler: AsyncWebCrawler, 
    config: CrawlerRunConfig
) -> List[Dict[str, Any]]:
    """Process a single link and return new links found."""
    if link["href"] not in urls_dict:
        urls_dict[link["href"]] = link
        logger.info(f"Processing link: {link['href']}")

        try:
            result = await crawler.arun(url=link["href"], config=config)
            return result.links.get("internal", [])
        except Exception as e:
            logger.error(f"Error processing link {link['href']}: {e}")
            return []
    else:
        logger.debug(f"Link already processed: {link['href']}")
        return []


async def deep_link_analysis(
    base_url: str, 
    n_rounds: int = 5, 
    workers: int = 8,
    max_urls: int = -1,
    crawler_config: Optional[CrawlerRunConfig] = None
) -> Dict[str, Any]:
    """
    Perform deep link analysis starting from a base URL.
    
    Args:
        base_url: Starting URL for the analysis
        n_rounds: Number of rounds of link processing
        workers: Number of concurrent workers
        max_urls: Maximum number of URLs to process
        crawler_config: Optional crawler configuration
        
    Returns:
        Dictionary of processed URLs
    """
    urls_dict: Dict[str, Any] = {}
    base_url = clean_url(base_url)
    
    if not crawler_config:
        config = CrawlerRunConfig(
            cache_mode=CacheMode.ENABLED,
            exclude_external_links=False,
            exclude_social_media_links=True,
        )
    else:
        config = crawler_config
    
    # Initialize the crawler as a context manager for proper cleanup
    async with AsyncWebCrawler() as crawler:
        # Initial link analysis
        logger.info(f"Starting link analysis from {base_url}")
        try:
            result = await crawler.arun(url=base_url, config=config)
        except Exception as e:
            logger.error(f"Failed to analyze base URL {base_url}: {e}")
            return urls_dict
            
        # Extract and clean initial links
        links = []
        for link in result.links.get("internal", []):
            href = clean_url(link["href"])
            link["href"] = href
            links.append(link)
        
        # Process links in rounds
        for round_idx in range(n_rounds):
            logger.info(f"Starting round {round_idx+1} of {n_rounds}")
            logger.info(f"Number of links to process: {len(links)}")
            
            new_links = []
            
            # Process links in batches
            total_batches = (len(links) + workers - 1) // workers
            for i in tqdm(range(0, len(links), workers), desc=f"Round {round_idx+1}"):
                batch_num = i // workers + 1
                logger.info(f'Processing batch {batch_num}/{total_batches}')
                
                # Get batch of links
                batch = links[i:i + workers]
                
                # Create processing tasks
                tasks = [process_link(link, urls_dict, crawler, config) for link in batch]
                
                # Run tasks in parallel
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Handle exceptions in batch results
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Failed to process link {batch[j]['href']}: {result}")
                        batch_results[j] = []
                
                # Process new links from results
                for sublist in batch_results:
                    for link in sublist:
                        href = clean_url(link["href"])
                        link["href"] = href
                        new_links.append(link)
                
                # Rate limiting between batches
                await asyncio.sleep(1)
                
                if max_urls > 0 and len(urls_dict) >= max_urls:
                    logger.info(f"Maximum number of URLs reached ({max_urls}). Finishing early.")
                    break
            
            # Add new links to the processing queue
            links = new_links
            
            # Exit early if no new links found or max URLs reached
            if not links:
                logger.info(f"No new links found in round {round_idx+1}. Finishing early.")
                break
            
            elif max_urls > 0 and len(urls_dict) >= max_urls:
                logger.info(f"Maximum number of URLs reached ({max_urls}). Finishing early.")
                break

    logger.info("Link analysis completed")
    logger.info(f"Total links found: {len(urls_dict)}")
    return urls_dict


def recursive_link_analysis(
    base_url: str,
    n_rounds: int = 3, 
    n_workers: int = 8,
    max_urls: int = -1,
    output_file: str = "",
) -> Dict:
    """
    Run deep link analysis and save results to a file.
    
    Args:
        base_url (str): Starting URL for the analysis
        n_rounds (int, optional): Number of rounds of link processing. Defaults to 3.
        n_workers (int, optional): Number of concurrent workers. Defaults to 8.
        max_urls (int, optional): Maximum number of URLs to process. Defaults to -1 (no limit).
        output_file (str, optional): Output file to save results. Defaults to "".
    
    Returns:
        Dictionary of processed URLs
    """
    # Validate inputs
    if not base_url.startswith(("http://", "https://")):
        raise ValueError("Base URL must start with 'http://' or 'https://'")
        
    # Create output directory if needed
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    # Warn if output file exists
    if os.path.exists(output_file):
        logger.warning(f"Output file {output_file} already exists and will be overwritten")

    # Run analysis
    result = asyncio.run(deep_link_analysis(base_url, n_rounds, workers=n_workers, max_urls=max_urls))

    # Save results
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Analysis results saved to {output_file}")
    
    return result


if __name__ == "__main__":
    import fire
    fire.Fire(recursive_link_analysis)