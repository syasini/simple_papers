from loguru import logger
from langchain_aws import ChatBedrock
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import ListOutputParser, CommaSeparatedListOutputParser, PydanticOutputParser
from typing import List
from pydantic import BaseModel
from simple_papers.path_handler import PathHandler
import json
import os
from functools import wraps


def save_keywords_to_json(func):
    """
    Decorator to save extracted keywords to a JSON file.
    
    Takes the list output from extract_keywords and saves each keyword as a key
    in a JSON file with an empty string as value. If the key already exists,
    its value is preserved.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Call the original function to get the keywords
        keywords = func(self, *args, **kwargs)
        
        try:
            # Create directory for the file if it doesn't exist
            os.makedirs(os.path.dirname(self.keywords_path), exist_ok=True)
            
            # Load existing data if file exists, or create new dict
            if os.path.exists(self.keywords_path):
                try:
                    with open(self.keywords_path, 'r') as f:
                        data = json.load(f)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON in {self.keywords_path}. Creating new file.")
                    data = {}
            else:
                data = {}
            
            # Add new keywords (only if they don't exist)
            for keyword in keywords:
                if keyword not in data:
                    data[keyword] = ""
            
            # Write back to file
            with open(self.keywords_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved {len(keywords)} keywords to {self.keywords_path}")
        except Exception as e:
            logger.error(f"Error saving keywords to JSON: {e}")
        
        # Return original result
        return keywords
    
    return wrapper


class KeywordList(BaseModel):
    keywords: List[str]


class WikiUrlExtractor:
    def __init__(self, paper_path: str, text: str):

        self.paper_path = paper_path
        self.path_handler = PathHandler(paper_path)
        self.keywords_path = self.path_handler.keywords_path
        
        self.text = text
        # self.context = context
        self.urls = {}

    @save_keywords_to_json
    def extract_keywords(self):
        
        # parser = CommaSeparatedListOutputParser()
        parser = PydanticOutputParser(pydantic_object=KeywordList)

        format_instructions = parser.get_format_instructions()

        logger.info(format_instructions)
        prompt_template = PromptTemplate.from_template(r"""
        

        Technical Keyword Extraction

        You are an expert technical indexer.
        Your task is to read simplified paragraph[s] (`<text>`) from a research paper in the AI and Machine Learning domain, and extract technical **keywords** a reader would likely want to look up on Wikipedia.

        ## What to Extract
        The technical keywords are the concepts that are central to understanding the text. These are words that a student would need to look up on Wikipedia to understand the text.
        Here are some examples categories along with examples of technical keywords:

        - **Models/architectures** (e.g., Transformer, ResNet, BERT, GPT, EfficientNet, U-Net, LSTM, GAN, VAE, AlexNet).
        - **Components/operations** (e.g., encoder, decoder, attention mechanism, self-attention, convolution, recurrence, pooling, normalization, embedding, dropout).
        - **Tasks/domains** (e.g., machine translation, constituency parsing, image classification, object detection, semantic segmentation, speech recognition, text summarization).
        - **Datasets/benchmarks** (e.g., WMT 2014, ImageNet, COCO, GLUE, SQuAD, CIFAR-10, MNIST).
        - **Metrics** (e.g., BLEU, ROUGE, mAP, F1-score, accuracy, precision, recall, perplexity).
        - **Core ComputerScience ML/NLP concepts** central to understanding the text (e.g., backpropagation, gradient descent, transfer learning, fine-tuning, tokenization).

        ## What NOT to Extract

        - Author names, institutions, footnotes.
        - Numbers or time spans unless part of a named benchmark (e.g., extract "WMT 2014" ✅, but not "3.5 days" ❌).
        - Generic words (e.g., “team,” “code,” “faster,” “records”).
        - Everyday language or filler.

        ## Guidelines

        - extracted keywords must be EXACTLY as they are in the text without any formatting. 
        - cases of the keywords must be preserved (e.g., Latent Diffusion -> Latent Diffusion, not latent diffusion, MNIST -> MNIST, not Mnist or mnist)
        - extracted keywords must be in the same case as they are in the text (e.g., if the keyword is in lowercase in the text, it must be in lowercase in the output).
        - extracted abbreviations must remain in their original form (e.g., if the keyword is in uppercase in the text, it must be in uppercase in the output).
        - if both abbreviation and its full form are present in the text, extract both.
        - Only extract keywords from `<text>` section. 
        - Return only the minimal set of technical terms that someone would realistically want to look up on Wikipedia.
        - **Output must ONLY be a flat Python-style list of strings (List[str])** — no explanations, no extra formatting.

        ## Input 

        Here is the text to extract keywords from:
        <text>
        {text}
        </text>
 
        

        ## Output Format

        Return a comma-separated list of strings (List[str]).
        example: ["Transformer", "ResNet", "BLEU", "WMT 2014"]

        <format_instructions>
        {format_instructions}
        </format_instructions>
        
        Here is the list of keywords:
        """,
        partial_variables={"format_instructions": format_instructions}
        )
        
        # prompt_template = prompt_template.partial(format_instructions=format_instructions)

        prompt = prompt_template.format(text=self.text, format_instructions=format_instructions)
        
        llm = ChatBedrock(
            model_id="amazon.nova-micro-v1:0", 
            # model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            region="us-east-1")
        
        

        # llm = llm.with_output_parsers(parser)
        response = llm.invoke(prompt)
        parsed_response = parser.parse(response.content)
        return parsed_response.keywords

    def extract_url(self, keyword: str) -> str:
        
        pass

    def extract_urls(self):
        pass
        
        