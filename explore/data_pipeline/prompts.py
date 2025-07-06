# Prompt to clean raw markdown document
PROMPT_TO_CLEAN_DOC = """
Givent crawled data, extract the main content while preserving its original wording and substance completely.
1. Maintain the exact language and terminology
3. Preserve the original flow and structure
4. Remove only clearly irrelevant elements like navigation menus and ads

Format the output as clean markdown with proper code blocks and headers.
""".strip()


# Prompt to summarize the document
PROMPT_TO_SUMMARIZE_DOC = """
Given the cleaned markdown document, provide a brief summary of the main content.

## Document:
{company_context}

## Your task
1. Summarize the key points and main topics of the document.
2. Focus on the most important information and ideas.
3. Write in a clear and concise manner.

Provide the result inside the <output> and </output> tags.
""".strip()

# Prompt to extract key information about the company
PROMPT_TO_EXTRACT_KEY_INFO = """
You are an advanced chatbot specialized in analyzing content and extracting key information.

## Company context:
{company_context}

## Task:
Analyze the company context  to identify and extract the key information about the company if available:
   - Company Name
   - Company Description (a concise summary of what the company does)
   - Key Products or Services (list them)
   - Target Audience or Customer Base
   - Key Personnel (founders, CEO, etc., if available)
   - Core Values or Mission Statement (if explicitly stated)
   - Any notable achievements or awards

Provide the result inside the <output> and </output> tags.
""".strip()

# Prompt to summarize the company information from cleaned file
PROMPT_TO_SUMMARIZE_KEY_INFO = """
Write a brief summary of the company {company_name} based on the provided company context. The summary should be concise and informative, highlighting the key aspects of the company.

### Company Context:
```
{company_context}
```

### Task:
Summarize the company information in a few pages, focusing on the company's industry, products or services, founder, target market, achievement and any other relevant details.

### Output Format:
Return your response in plain text format wrapped between `<output>` and `</output>` tags.

#### Example Output:

<output>
Summary of the company information...
</output>
""".strip()


PROMPT_TO_CONVERT_DOC_TO_MARKDOWN = """
Given the company document, your task is to extract the main content and response in structured markdown

1. Maintain the exact language and terminology
2. Analyze the company context  to identify and extract the key information about the company if available
3. Remove only clearly irrelevant elements

Return your response in markdown format, do not explain anything else
""".strip()