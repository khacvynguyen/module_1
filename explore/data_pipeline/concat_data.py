from typing import List
import pandas as pd


def concat_dataframes(
    public_data_path: str,
    internal_data_path: str,
    output_csv_path: str = "",
    concat_columns: List[str] = ["doc_id", "url", "metadata", "cleaned_markdown"]
) -> pd.DataFrame | None:
    """
    Concatenates two CSV files into a single DataFrame, keeping only specified columns, and optionally saves the result to a CSV file.

        public_data_path (str): Path to the first (public) CSV file.
        internal_data_path (str): Path to the second (internal) CSV file.
        output_csv_path (str, optional): Path to save the concatenated DataFrame as a CSV file. If not provided, the DataFrame is returned.
        concat_columns (List[str], optional): List of columns to include in the concatenated DataFrame. Defaults to ["doc_id", "url", "metadata", "cleaned_markdown"].

        pd.DataFrame | None: The concatenated DataFrame if `output_csv_path` is not provided; otherwise, None.
    """
    # Read DataFrames from file paths
    data_paths = [public_data_path, internal_data_path]
    dataframes = [pd.read_csv(path) for path in data_paths]

    # Concatenate DataFrames
    concatenated_df = pd.concat(dataframes, ignore_index=True)

    # Keep only specified columns
    concatenated_df = concatenated_df[concat_columns]

    # Save to CSV if output path is provided
    if output_csv_path:
        concatenated_df.to_csv(output_csv_path, index=False)
        print(f"Output DataFrame saved to {output_csv_path}")
        return None
    else:
        return concatenated_df

if __name__ == "__main__":
    import fire
    fire.Fire(concat_dataframes)