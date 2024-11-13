import os
from typing import Dict
import anthropic
from dotenv import load_dotenv

def get_anthropic_response(
    prompt: Dict[str, str], model_name="claude-3-5-haiku-20241022"
) -> str:
    """Get response from Anthropic API using Claude model"""
    # Load environment variables from .env file
    load_dotenv()

    # Get API key from environment
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

    # Initialize Anthropic client
    client = anthropic.Anthropic(api_key=api_key)
    system_prompt = prompt["system_prompt"]
    user_prompt = prompt["user_prompt"]

    # Get completion from Claude
    message = client.messages.create(
        model=model_name,
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    return message.content[0].text