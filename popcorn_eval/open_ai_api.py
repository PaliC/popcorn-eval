import os
from typing import Dict

from dotenv import load_dotenv
from openai import OpenAI


def get_openai_response(prompt: Dict[str, str], model_name="gpt-4o-mini") -> str:
    """Get response from Anthropic API using Claude model"""
    # Load environment variables from .env file
    load_dotenv()

    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    openai_organization = os.getenv("OPENAI_ORGANIZATION")
    openai_project_id = os.getenv("OPENAI_PROJECT_ID")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    if not openai_organization:
        raise ValueError("OPENAI_ORGANIZATION not found in environment variables")
    if not openai_project_id:
        raise ValueError("OPENAI_PROJECT_ID not found in environment variables")

    # Initialize Anthropic client
    client = OpenAI(
        api_key=api_key,
        organization=openai_organization,
        project=openai_project_id,
    )
    system_prompt = prompt["system_prompt"]
    user_prompt = prompt["user_prompt"]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Get completion from Claude
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=4096,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    prompt = {
        "system_prompt": "You are a helpful assistant.",
        "user_prompt": "What is the capital of France?",
    }
    response = get_openai_response(prompt)
