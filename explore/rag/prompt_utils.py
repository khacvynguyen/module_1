from jinja2 import Template, Environment, meta
from typing import Dict, List
from string import Formatter
import json
import yaml
import textwrap


def extract_placeholders(template: str):
    """
    Extracts placeholders from a template string using string.Formatter.
    """
    return list({field_name for field_name, *_ in Formatter().parse(template) if field_name})


def to_print(text):
    print(textwrap.fill(text, width=80))
    print("")


def get_jinja_placeholder(jinja_template: str):
    """Get placeholer from jinja template"""
    
    env = Environment()
    parsed_content = env.parse(jinja_template)
    placeholders = meta.find_undeclared_variables(parsed_content)

    return placeholders


def fill_metadata(prompt_template: str, **kwargs):
    """Format prompt with kwargs"""
    template = Template(prompt_template)

    # jinja template replace "\n" by "<br>""
    for k, v in kwargs.items():
        if isinstance(v, str):
            kwargs[k] = v.replace("\n", "<br>")

    return template.render(**kwargs)


def read_json_file(file_path):
    with open(file_path, 'r') as file:
        try:
            data = json.load(file)
            return data
        except json.JSONDecodeError as exc:
            print(f"Error parsing JSON file: {exc}")
        return None


def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as exc:
            print(f"Error parsing YAML file: {exc}")
        return None


def load_jsonl(file_path) -> list[dict[str, any]]:
    """Load a JSONL file into a list of dictionaries."""
    data = []
    with open(file_path, "r") as jsonl_file:
        for line in jsonl_file:
            data.append(json.loads(line))
    return data


def msg_to_conv(messages: List[Dict[str, str]], role_map: Dict[str, str]) -> str:
    """Converts message role keys to role info keys."""
    for msg in messages:
        if msg["role"] in role_map:
            msg["role"] = role_map[msg["role"]]

    conv_str = "\n\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

    return conv_str