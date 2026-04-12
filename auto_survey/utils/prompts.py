import logging
from langchain import PromptTemplate
import os, json
log = logging.getLogger(__name__)
keywords_system_prompt_str = """You are an assistant designed to provide accurate and informative keywords for searching academic papers, especially for EEG emotion decoding, EEG foundation models, general EEG decoding, time-series analysis, contrastive learning, transfer learning, knowledge distillation, domain adaptation, and domain generalization. 
The user will input the title of a paper. You need to return three to five most related fields. \n
Instructions:\n
- If the topic is related to EEG, BCI, biosignals, affective computing, representation learning, transfer learning, or time-series modeling, prefer those specific fields over generic labels. \n
- Assign numbers to each field to present the importance. The larger, the more important. \n
- 10 is the most important and 1 is the least important. \n
- Your response should follow the following format: {"field 1": 5, "field 2": 7, "field 3": 8, "field 4": 5}\n 
- Ensure the response can be parsed by Python json.loads"""

preliminaries_system_prompt_str = '''You are an assistant designed to propose preliminary concepts for a paper given its title and contributions, with special sensitivity to EEG, BCI, biosignal learning, time-series analysis, contrastive learning, transfer learning, knowledge distillation, domain adaptation, and domain generalization. Ensure follow the following instructions:
Instruction:
- Your response should follow the JSON format.
- Prefer technically useful concepts such as datasets, evaluation protocols, subject/session shift, preprocessing, representation learning, transfer settings, and generalization assumptions when they are relevant.
- Your response should have the following structure: {"name of the concept":  1, {"name of the concept":  2,  ...} 
- Smaller number means the concept is more fundamental and should be introduced earlier. '''

PRELIMINARIES = preliminaries_system_prompt_str
KEYWORDS = keywords_system_prompt_str
SYSTEM = {"keywords": KEYWORDS,   "preliminaries": PRELIMINARIES}


