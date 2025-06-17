from typing import Dict, Union

def parse_llm_response(raw_response: Union[Dict[str, str], str], delimiter: str = "<JSON>", left_find: bool = True) -> str:
    """Parse the LLM response

    Args:
        raw_response: response from the LLM [dict | str], if dict, then it should have a key "content"
        delimiter: str, default is "<JSON>"
        left_find: bool, default is True

    if delimiter.startswith("<"):
        end_delimiter = "</" + delimiter[1:]
    else:
        end_delimiter = delimiter
        str: parsed response
    """

    response = raw_response["content"] if isinstance(raw_response, dict) else raw_response
    end_delimiter = "</" + delimiter[1:]

    if left_find:
        start_index = response.find(delimiter)
        end_index = response.find(end_delimiter, start_index + len(delimiter))
    else:
        start_index = response.rfind(delimiter)
        end_index = response.find(end_delimiter, start_index + len(delimiter))

    return response[start_index + len(delimiter) : end_index].strip()


