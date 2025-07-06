import json
import pandas as pd
from typing import Dict, List, Any


def split_url(url: str) -> List[str]:
    """Split URL into components after removing common prefixes."""
    cleaned_url = url
    for prefix in ["https://", "http://", "www."]:
        cleaned_url = cleaned_url.replace(prefix, "")
    
    return cleaned_url.split("/")


def clean_url(url: str) -> str:
    """Remove query parameters, fragments, and trailing slashes from URL."""
    
    clean_url = url.split('?')[0].split('#')[0]
    return clean_url.rstrip('/')


def load_crawl4ai_links(file_paths: List[str] | str) -> Dict[str, Dict[str, Any]]:
    """Load and clean URLs from Craw4AI output files."""
    all_urls_dict = {}
    
    if isinstance(file_paths, str):
        file_paths = [file_paths]
        
    for file_path in file_paths:
        with open(file_path, "r") as f:
            data = json.load(f)
            
        for _, url_obj in data.items():
            url_obj["href"] = clean_url(url_obj["href"])
            all_urls_dict[url_obj["href"]] = url_obj
            
    return all_urls_dict


def create_url_dataframe(urls_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Create a DataFrame from URLs dictionary and add url domain information."""
    df_url = pd.DataFrame(list(urls_dict.values()))
    
    # Add domain information
    df_url["domains"] = df_url["href"].apply(split_url)
    df_url["base_domain"] = df_url["domains"].apply(lambda x: x[0])
    df_url["domain_idx_02"] = df_url["domains"].apply(
        lambda x: "/".join(x[:2]) if len(x) > 1 else x[0]
    )
    df_url["end_domain"] = df_url["domains"].apply(lambda x: x[-1])
    
    return df_url


def filter_urls(input_urls: pd.DataFrame | Dict[str, Any], prefix_filter: list[str] | str = "") -> pd.DataFrame:
    """
    Filter URLs based on various criteria.
    
    Args:
        input_urls: DataFrame or dictionary of craw4ai URLs 
        prefix_filter: Domain prefix to filter by
        
    Returns:
        Filtered DataFrame
    """
    
    if isinstance(input_urls, dict):
        df_links = create_url_dataframe(input_urls)
    else:
        df_links = input_urls
    
    filtered_df = df_links.copy()
    
    # Exclude specific base domains
    # exclude_base_domains = ["careers.singaporeair.com"]
    # filtered_df = df[~df["base_domain"].isin(exclude_base_domains)]
    
    # Filter by domain prefixes (include URLs matching any of the prefixes)
    if prefix_filter:
        if isinstance(prefix_filter, str):
            prefix_filter = [prefix_filter]

        prefix_mask = filtered_df["href"].apply(
            lambda x: any(x.startswith(prefix) for prefix in prefix_filter)
        )
        filtered_df = filtered_df[prefix_mask]
    
    # Remove URLs with specific file extensions
    filtered_df = filtered_df[~filtered_df["end_domain"].str.endswith((".pdf", ".form", ".jsp"))]
    
    # Exclude specific domain indexes
    # excluded_domains_idx_02 = ["telusdigital.com/insights"]
    # filtered_df = filtered_df[~filtered_df["domain_idx_02"].isin(excluded_domains_idx_02)]
    
    # Remove duplicates
    filtered_df = filtered_df.drop_duplicates(subset='href')
    filtered_df = filtered_df.drop_duplicates(subset='end_domain')
    
    return filtered_df


def main(
    input_json_paths: List[str] | str,
    output_csv_path: str,
    prefix_filter: str,
):
    """Main function to process and filter URLs.
    
    Args:
        input_json_paths (List[str] | str): List of JSON file paths containing URLs or a single file path
        output_csv_path (str): Output CSV file path
        prefix_filter (str): Domain prefix to filter by (e.g., "https://www.example.com/en")
    """

    # Load URLs
    all_urls_dict = load_crawl4ai_links(input_json_paths)
    print(f"Total URLs loaded: {len(all_urls_dict)}")
    
    # Create URLs DataFrame
    df_url = create_url_dataframe(all_urls_dict)
    
    ## filter URLs
    filtered_df = filter_urls(df_url, prefix_filter)
    print(f"URLs after filtering: {len(filtered_df)}")
    
    # Save to CSV
    filtered_df.to_csv(output_csv_path, index=False)
    print(f"Filtered URLs saved to {output_csv_path}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)