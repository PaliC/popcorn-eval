
from datetime import datetime
from typing import Dict

import os

import tomli
from dotenv import load_dotenv
import anthropic
import json

def get_anthropic_response(prompt: Dict[str, str]) -> str:
    """Get response from Anthropic API using Claude model"""
    # Load environment variables from .env file
    load_dotenv()
    
    # Get API key from environment
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        
    # Initialize Anthropic client
    client = anthropic.Anthropic(api_key=api_key)
    user_prompt = compose_prompt_for_completion(prompt)
    system_prompt = prompt["system_prompt"]
    user_prompt = prompt["user_prompt"]
    
    # Get completion from Claude
    message = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}]
    )
    
    return message.content[0].text




# get todays date in the format Day Month Year
TODAY_DATE = datetime.now().strftime("%d %B %Y")
CUTOFF_KNOWLEDGE_DATE = "December 2023"
SYSTEM_PROMPT_TOKEN = "[[SYSTEM_PROMPT]]"
USER_PROMPT_TOKEN = "[[USER_PROMPT]]"

COMPLETION_PROMPT_TEMPLATE = f"""

  <|begin_of_text|><|start_header_id|>system<|end_header_id|>

  Cutting Knowledge Date: {CUTOFF_KNOWLEDGE_DATE}
  Today Date: {TODAY_DATE}

  {SYSTEM_PROMPT_TOKEN}
  <|eot_id|><|start_header_id|>user<|end_header_id|>

  {USER_PROMPT_TOKEN}
  <|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

def compose_prompt_for_completion(prompt_dict: Dict[str, str]) -> str:
    system_prompt = prompt_dict["system_prompt"]
    user_prompt = prompt_dict["user_prompt"]

    prompt = COMPLETION_PROMPT_TEMPLATE.replace(SYSTEM_PROMPT_TOKEN, system_prompt).replace(USER_PROMPT_TOKEN, user_prompt)
    return prompt



if __name__ == "__main__":
    # grab first prompt in eval_prompts.toml
    with open("prompts/eval_prompts.toml", "rb") as f:
        all_prompts = tomli.load(f)["prompts"]
        prompt_dict = all_prompts[0]
    
    response = get_anthropic_response(prompt_dict)
    print(response)
    
    

