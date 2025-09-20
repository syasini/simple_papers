import json
import time
from loguru import logger
from typing import Dict, List, Any, Optional
from pathlib import Path

# Import langchain components
from langchain_aws import ChatBedrock
from langchain_core.prompts import PromptTemplate

# Import local modules
from simple_papers.path_handler import PathHandler
from simple_papers.wiki import WikiUrlExtractor


class Summarizer:
    """Class to handle summarization of document sections.
    
    Uses AWS Bedrock's Claude 3.7 Sonnet model through LangChain to generate
    summaries of document sections based on their text content.
    """
    
    # Claude 3.7 Sonnet model ID for AWS Bedrock
    MODEL_ID = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
    
    # Default AWS region
    AWS_REGION = "us-east-1"
    
    # System prompt template for summarization
    SYSTEM_PROMPT_TEMPLATE = """
        <system>
        You are a friendly and slightly goofy scientist named "Doc Scribbles." You help high school students understand academic research papers by summarizing them clearly and lightly.

        Your job is to read the provided section of an academic paper and explain what it means in simple, fun language — but still accurately and in a way that fits with the bigger picture of the research.

        <persona>
        - Speak casually, like a quirky science teacher who makes things fun and approachable
        - Use light humor to enhance clarity, not to distract from it
        - Keep a warm, approachable tone with just enough nerdy charm to make complex ideas feel fun
        </persona>

        <example_phrases>
        <humorous>
            - "Umm so basically…"
            - "This part is kinda interesting!"
            - "They did math — lots of it!"
            - "This would make most calculators cry."
            - "Okay, time to nerd out a bit."
            - "Science goggles on — this one's fun."
            - "Fire up the neurons — we're diving in."
            - "This step is like sorting your socks… but with calculus."
            - "If this were a recipe, we'd be preheating the oven and measuring flour with a laser."
            - "Looks like a jigsaw puzzle made of equations."
            - "Okay, deep breath — here comes the tricky math."
            - "This bit is dense. Like, neutron star dense."
            - "Now the numbers are flexing their muscles."
            - "They're building this AI like LEGO — one clever block at a time."
            - "Here's where things get slightly more confusing (but also cooler)."
            - "This is where the math wizardry happens."
            - "Imagine explaining this to your dog…"
            - "Now they're turning ideas into formulas — hold onto your neurons."
            - "Imagine doing all this on a chalkboard… with just coffee and hope."
        </humorous>

        <section_openers>
            - "Let's set the stage — here's what makes this so powerful."
            - "Time to meet the star of the show."
            - "Okay, this is where they explain what all the fuss is about."
            - "Here's why this model became a machine learning celebrity."
            - "They're about to make their case — and it's a strong one."
            - "This section lays the groundwork — and it's surprisingly compelling."
            - "Here's where they show off why their idea matters."
            - "Before the math kicks in, let's see what all the hype is about."
            - "This part's all about the 'why' — and it's pretty convincing."
        </section_openers>
        </example_phrases>

        <usage_guidance>
        - These example phrases are just suggestions. You can use them as inspiration but don't use them verbatim.
        - Rotate and remix these phrases naturally. Use them rarely and don't repeat the same one across adjacent sections.
        - ONLY use humor when it helps the explanation land better — never let it distract from clarity.
        - Don't force a joke into every paragraph. Use your judgment and skip the jokes entirely if clarity is more important.
        - Vary your tone and phrase choices based on the section content: some sections will need more structure, others more fun.
        </usage_guidance>

        <format>
        For each summary:
        1. **For main sections (whole numbers like "2. Background")**: Start with a short, playful opener that highlights the main idea of the section
        2. **For subsections (like "3.1", "4.2.1")**: Skip the opener and go straight to bullet points
        3. **Follow with 2 to 3 bullet points** that:
        - Break down the core ideas in plain language
        - Use analogies or examples if helpful
        4. If summarizing a main section (whole number), include a brief "so what?" explanation if it helps clarify why something matters
        - for subsections skip the "so what" explanation
        5. **Close with a short connector sentence** (optional), like "So basically…" to wrap it up

        <markdown_formatting>
        - Always apply correct markdown formatting to section titles, even if the original input is missing it.
        - Use a single `#` for top-level sections (e.g., "# Abstract", "# 1. Introduction").
        - Use `##`, `###`, etc. for subsections based on numerical hierarchy (e.g., "## 3.1 Results", "### 4.2.1 Details").
        - DO NOT change the wording of the section title — preserve the original title text exactly.
        - Keep the properly formatted heading at the top of the summary before the explanation begins.
        - Start each summary with an italicized opening phrase using *italic text* (e.g., *Time to meet the star of the show*)
        - Separate each bullet point with a new line (\n) for proper formatting and readability.
        </markdown_formatting>
        </format>


        <context>
        Here is the abstract of the paper to help you understand the overall topic:

        <abstract>
        {abstract}
        </abstract>

        These are the opening of existing summaries to avoid repetition:
        <existing_summaries_opening>
        {existing_summaries_opening}
        </existing_summaries_opening>
        </context>

        <instructions>
        - Use the abstract to stay grounded in the paper's main goals
        - Do NOT copy original text — always paraphrase in your own voice
        - Be concise — no more than 3 bullet points
        - Focus on clarity and lightness — aim to teach and delight, not overwhelm
        - Avoid too much repetition in phrasing or jokes
        - Use the existing opening of summaries from other sections to avoid repetition
        - Always start with an italicized opening phrase using *italic text* to grab attention and set the tone (e.g., *Time to meet the star of the show*, *Let's dive into the nitty-gritty*)
        </instructions>
        </system>


        Please summarize the following text:

        {text}

        Here is the summary:
    """
    
    def __init__(self, paper_path: str, annotations_list: List[Dict[str, Any]]):
        """Initialize with a list of annotations.
        
        Parameters
        ----------
        paper_path : str
            Path to the paper file
        annotations_list : List[Dict[str, Any]]
            List of annotation dictionaries from Annotation.get_annotations()
        """
        self.paper_path = paper_path
        self.path_handler = PathHandler(paper_path)
        self.summaries_path = self.path_handler.summaries_path
        self.keywords_path = self.path_handler.keywords_path
        self.annotations_list = annotations_list
        self.abstract = self.get_abstract()
        self._llm = None
        
        # Load summaries from file if it exists
        self._summaries = self._load_summaries()
    

    def _load_summaries(self) -> Dict[str, str]:
        """Load summaries from JSON file if it exists.
        
        Returns
        -------
        Dict[str, str]
            Dictionary of section summaries
        """
        if self.summaries_path.exists():
            try:
                logger.info(f"Loading summaries from {self.summaries_path}")
                with open(self.summaries_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading summaries: {str(e)}")
                return {}
        return {}
        
    def _save_summaries(self) -> None:
        """Save summaries to JSON file."""
        try:
            logger.info(f"Saving summaries to {self.summaries_path}")
            with open(self.summaries_path, "w") as f:
                json.dump(self._summaries, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving summaries: {str(e)}")
    
    @property
    def summaries(self) -> Dict[str, str]:
        """Get summaries for all sections."""
        return self._load_summaries()
    
    def get_abstract(self) -> str:
        """Extract the abstract from annotations list.
        
        Returns
        -------
        str
            The text of the abstract, or an empty string if not found
        """
        for annot in self.annotations_list:
            if "abstract" in annot["group"]:
                return annot["group_text"]
        return ""
    
    @property
    def llm(self):
        """Lazy-load the LangChain model."""
        if self._llm is None:
            self._llm = self._initialize_llm()
        return self._llm
    
    def _initialize_llm(self):
        """Initialize the AWS Bedrock Claude 3.7 model.
        
        Returns
        -------
        ChatBedrock
            Configured LangChain Claude 3.7 Sonnet model
        """
        try:
            # Initialize the model with reasoning capability
            model = ChatBedrock(
                model=self.MODEL_ID,
                region=self.AWS_REGION,
                max_tokens=2000,
                model_kwargs={
                    "temperature": 0.1,  # Low temperature for factual summaries
                    # "thinking": {"type": "enabled", "budget_tokens": 1024}
                }
            )
            logger.info(f"Successfully initialized Claude 3.7 Sonnet model")
            return model
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            raise
    
    def summarize_section(self, group_id: str, override_summary: bool = False) -> str:
        """Summarize a section based on its group ID.
        
        Parameters
        ----------
        group_id : str
            The group ID to summarize (e.g., "1-abstract", "2-introduction")
        override_summary : bool, default=False
            If True, generate a new summary even if one exists
            
        Returns
        -------
        str
            The generated summary, or error message if summarization failed
        """
        logger.info(f"Summarizing section {group_id}")

        if not override_summary:
            if self._summaries.get(group_id):
                return self._summaries[group_id]
        
        # Find all annotations with the given group
        section_text = ""
        for annot in self.annotations_list:
            if annot["group"] == group_id:
                section_text = annot["group_text"]
                break
        
        if not section_text:
            return f"No content found for section {group_id}"
        
        if group_id in ["0-title", "1-title"]:
            section_summary = self._summarize_title(section_text)
        
        elif "reference" in group_id:
            section_summary = self._summarize_reference_group()
        
        else:
            section_summary = self.summarize_text(section_text)
        
        self._summaries[group_id] = section_summary
        self._save_summaries()  # Save to file after updating
        
        return section_summary

    def _summarize_title(self, title: str) -> str:
        """Summarize the title to be short, and fun using a sports announcer style.
        
        Parameters
        ----------
        title : str
            The paper title to summarize
            
        Returns
        -------
        str
            The announced title in a fun, energetic sports announcer style
        """
        # Define the system prompt template
        system_prompt_template = """
        <system>
        You are an energetic and charismatic **sports announcer**, and your job is to introduce academic papers the same way you'd hype up an epic championship match.

        <persona>
        - Sound excited and upbeat — like you're calling a big game or unveiling a new star player
        - Use fun, punchy phrases like:
        "Coming in hot!", "Buckle up, folks!", "Let's give it up for…"
        - DO NOT say "crowd roars", or "airhorn sounds", or "cheering", or "applause", or "applause and cheers", or similar phrases to indicate audience reaction
        - Emphasize dramatic delivery and rhythm — your goal is to get people *excited* about reading a paper!
        </persona>

        <format>
        Here is the title and list of authors:
        <title>
        {title}
        </title>
        

        Now generate a short one-paragraph hype-style **announcement** that:
        - Introduces the title with flair (like it's a big reveal)
        - Starts with LAAAAAAADIES AND GENTLEMEN! 
        - Calls out the authors like an all-star lineup (use first and last names)
        - Is fun, bold, and a little over-the-top — just like a sports commentator at their peak
        </format>

        <instructions>
        - Don't just read the title and author list — perform it!
        - Keep it short (1 paragraph), but packed with energy
        - Add dramatic pauses, exclamations, or alliteration for flair
        - No need to explain what the paper is about — just make the *title* sound amazing and the authors sound legendary
        </instructions>
        </system>
        """

        prompt_template = PromptTemplate.from_template(system_prompt_template).format(title=title)

        try:
            return self._invoke_with_retry(prompt_template)
        except Exception as e:
            logger.error(f"Error in title summarization: {str(e)}")
            return f"Failed to generate title summary: {str(e)}"
    
    def _get_title(self) -> str:
        """Extract the paper title from annotations list.
        
        Returns
        -------
        str
            The paper title, or an empty string if not found
        """
        for annot in self.annotations_list:
            if annot["group"] == "0-title" or annot["group"] == "1-title":
                return annot["group_text"]
        return ""
    
    def _summarize_reference_group(self) -> str:
        """Summarize the reference group to blah blah."""
        return "References omitted from summary"
    
    
    def _invoke_with_retry(self, formatted_prompt: str, max_retries: int = 3, base_delay: float = 1.0) -> str:
        """Invoke LLM with exponential backoff retry logic.

        Parameters
        ----------
        formatted_prompt : str
            The formatted prompt to send
        max_retries : int
            Maximum number of retry attempts
        base_delay : float
            Base delay in seconds (doubles each retry)

        Returns
        -------
        str
            Model response content
        """
        for attempt in range(max_retries + 1):
            try:
                response = self.llm.invoke(formatted_prompt)

                # Log if thinking is available in the response
                if hasattr(response, 'additional_kwargs') and 'thinking' in response.additional_kwargs:
                    logger.debug(f"Model thinking: {response.additional_kwargs['thinking']}")

                return response.content if hasattr(response, 'content') else str(response)

            except Exception as e:
                if attempt == max_retries:
                    # Final attempt failed
                    raise e

                if "ThrottlingException" in str(e) or "Too many requests" in str(e):
                    # Calculate exponential backoff delay
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Rate limit hit, retrying in {delay} seconds... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                else:
                    # Non-throttling error, don't retry
                    raise e

    def summarize_text(self, text: str, system_prompt_template: str = None) -> str:
        """Summarize text using AWS Bedrock's Claude model.

        Parameters
        ----------
        text : str
            Text to summarize
        system_prompt_template : str, optional
            Custom system prompt template to use, by default None

        Returns
        -------
        str
            Generated summary
        """
        # Use default system prompt template if not specified
        template = system_prompt_template or self.SYSTEM_PROMPT_TEMPLATE

        # get the opening of existing summaries to avoid repetition
        existing_summaries = {group_id: summary[:150]
            for group_id, summary in self.summaries.items()
            if group_id != "0-title" and group_id != "1-title"}
        existing_summaries_opening = "\n\n".join(existing_summaries.values())
        logger.debug(f"Existing summaries openning: {existing_summaries_opening}")

        # Create a prompt template from the system prompt template
        prompt_template = PromptTemplate.from_template(template)

        try:
            # Format the prompt with abstract and text to summarize
            formatted_prompt = prompt_template.format(
                abstract=self.abstract,
                text=text,
                existing_summaries_opening=existing_summaries_opening
            )

            # Log the prompt for debugging (optional)
            logger.debug(f"Sending prompt to model: {formatted_prompt[:100]}...")

            # Use retry logic for the API call
            return self._invoke_with_retry(formatted_prompt)

        except Exception as e:
            logger.error(f"Error in summarization: {str(e)}")
            return f"Failed to generate summary: {str(e)}"
    
    def summarize_all_sections(self, delay_between_requests: float = 0.5, progress_callback=None) -> Dict[str, str]:
        """Summarize all sections in the document with rate limiting protection.

        Parameters
        ----------
        delay_between_requests : float
            Delay in seconds between API requests to avoid rate limits
        progress_callback : callable, optional
            Function to call with progress updates (progress_value, progress_text)

        Returns
        -------
        Dict[str, str]
            Dictionary mapping section group IDs to summaries
        """
        results = {}

        # Create a list of groups from annotations
        groups = set()
        for annot in self.annotations_list:
            groups.add(annot["group"])

        # Sort groups numerically by the number before the dash
        def sort_key(group_id):
            try:
                # Extract number before the dash (e.g., "2-intro" -> 2)
                return int(group_id.split('-')[0])
            except (ValueError, IndexError):
                # If no number or invalid format, sort to end
                return float('inf')

        sorted_groups = sorted(groups, key=sort_key)
        total_groups = len(sorted_groups)

        for i, group_id in enumerate(sorted_groups):
            try:
                logger.info(f"Summarizing section {group_id} ({i+1}/{total_groups})")

                # Update progress before processing
                if progress_callback:
                    progress_value = i / total_groups
                    progress_callback(progress_value, f"Summarizing {group_id}... ({i+1}/{total_groups})")

                summary = self.summarize_section(group_id, override_summary=True)
                results[group_id] = summary

                # Update progress after completing this section
                if progress_callback:
                    progress_value = (i + 1) / total_groups
                    progress_callback(progress_value, f"{((i+1)/total_groups)*100:.0f}% Complete")

                # Add delay between requests (except for the last one)
                if i < total_groups - 1:
                    logger.info(f"Waiting {delay_between_requests} seconds before next request...")
                    time.sleep(delay_between_requests)

            except Exception as e:
                logger.error(f"Failed to summarize section {group_id}: {str(e)}")
                results[group_id] = f"Failed to generate summary: {str(e)}"

        return results

    def extract_all_keywords(self):
        """Extract keywords from all sections."""


        # load the summaries from the summaries file
        if self.summaries_path.exists():
            with open(self.summaries_path, "r") as f:
                summaries = json.load(f)

            for group_id, summary in summaries.items():
                if group_id in ["0-title", "1-title"]:
                    continue
                elif "reference" in group_id:
                    continue
                else:
                    wiki_extractor = WikiUrlExtractor(self.paper_path, summary)
                    wiki_extractor.extract_keywords()

    # def highlight_summary_keywords(self, summary: str):
    #     logger.info(self.keywords_path)
    #     if self.keywords_path.exists():
    #         with open(self.keywords_path, "r") as f:
    #             keywords = json.load(f)
    #         for keyword in keywords:
    #             logger.info(keyword)
    #             keyword_url = f"https://www.google.com/search?q=what+is"
    #             logger.info(keyword_url)
    #             logger.info(summary)
    #             # Check if keyword is already in markdown link format or is part of existing link text
    #             import re
    #             # Pattern to match keyword that's not inside square brackets of markdown links
    #             pattern = r'(?<!\[)(?<!\[[^\]]*)\b' + re.escape(keyword) + r'\b(?![^\[]*\])'
    #             if re.search(pattern, summary):
    #                 summary = re.sub(pattern, f"[{keyword}]({keyword_url})", summary)
    #     return summary

    def percentage_summarized(self):
        """Calculate percentage of sections summarized"""

        n_groups = len(set([annot["group"] for annot in self.annotations_list]))
        return len(self.summaries) / n_groups
        
    def highlight_summary_keywords(self, summary):
        """Highlight keywords in the summary with links"""
        if not summary:
            return summary

        if self.keywords_path.exists():
            with open(self.keywords_path, "r") as f:
                keywords = json.load(f)
        else:
            keywords = []
            
        for keyword in keywords:
            if keyword.lower() in summary.lower():
                import re
                
                # Find all markdown links first
                link_positions = []
                for link_match in re.finditer(r'\[([^\]]*)\]\([^\)]*\)', summary):
                    link_positions.append((link_match.start(), link_match.end()))
                
                # Find all instances of the keyword
                keyword_pattern = r'\b' + re.escape(keyword) + r'\b'
                result = summary
                
                # Process matches from end to beginning to avoid index shifts
                for match in reversed(list(re.finditer(keyword_pattern, summary))):
                    start, end = match.span()
                    
                    # Check if this keyword is inside any markdown link
                    in_link = False
                    for link_start, link_end in link_positions:
                        if link_start <= start and end <= link_end:
                            in_link = True
                            break
                    
                    # If not in a link, replace it
                    if not in_link:
                        keyword_url = f"https://www.google.com/search?q=what+is+{keyword.replace(' ', '+')}+in+machine+learning"
                        replacement = f"[{keyword}]({keyword_url})"
                        result = result[:start] + replacement + result[end:]
                        
                summary = result
                
        return summary       


