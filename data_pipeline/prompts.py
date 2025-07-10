PROMPT_TO_CONVERT_DOC_TO_MARKDOWN = """
Given the company document, your task is to extract the main content and response in structured markdown

1. Maintain the exact language and terminology
2. Remove only clearly irrelevant elements
3. If document have multiple sections, separate them by markdown headers

Return your response in markdown format, do not explain anything else
""".strip()