from typing import Dict, Any
import pandas as pd
import yaml
import json
import re

def read_json(file_path: str) -> Dict[Any, Any]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise Exception(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        raise Exception(f"Error decoding JSON: {e}")
    
    
def read_yaml_file(file_path: str) -> Dict[Any, Any]:
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as exc:

            print(f"Error parsing YAML file: {exc}")
            return None


def clean_text(text: str) -> str:
    # remove special character
    return re.sub(r"[^a-zA-Z0-9 ]", "", text).strip()


def get_balanced_sample(df: pd.DataFrame, group_col:str, sample_size=20, random_state=42):
    
    # Group by the specified column and sample from each group
    df_balanced = (df.groupby(group_col)
                     .apply(lambda x: x.sample(n=sample_size, random_state=random_state)
                     if len(x) >= sample_size else x)
                     .reset_index(drop=True))
    
    return df_balanced


def get_diverse_sample(df: pd.DataFrame, group_col1: str, group_col2: str, sample_size=20, random_state=42):
    # First get balanced samples for both columns
    df_temp = df.copy()

    # Group by both columns and sample from each combination
    df_balanced = (df_temp.groupby([group_col1, group_col2])
                         .apply(lambda x: x.sample(n=sample_size, random_state=random_state)
                         if len(x) >= sample_size else x)
                         .reset_index(drop=True))

    return df_balanced
