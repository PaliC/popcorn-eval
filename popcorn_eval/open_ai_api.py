import os
from typing import Dict

from dotenv import load_dotenv
from openai import OpenAI


def get_openai_gpt_response(
    prompt: Dict[str, str], model_name="gpt-4o-mini-2024-07-18"
) -> str:
    """Get response from OpenAI API using GPT model"""
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

    # Initialize OpenAI client
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

    # Get completion from GPT model
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=4096,
    )
    return response.choices[0].message.content


def get_openai_o1_response(
    prompt: Dict[str, str], model_name="o1-preview-2024-09-12"
) -> str:
    """Get response from OpenAI API using O1 model"""
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

    # Initialize OpenAI client
    client = OpenAI(
        api_key=api_key,
        organization=openai_organization,
        project=openai_project_id,
    )

    # o1 does not support system prompt so we need to combine system and user prompt

    system_prompt = prompt["system_prompt"]
    user_prompt = prompt["user_prompt"]

    user_prompt = f"{system_prompt}\n{user_prompt}"

    messages = [
        {"role": "user", "content": user_prompt},
    ]

    # Get completion from o1 model
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=4096,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    prompt = {
        "system_prompt": "You are a helpful assistant.",
        "user_prompt": "What is the capital of France?",
    }
    response = get_openai_gpt_response(prompt)
    print(response)
    response = get_openai_o1_response(prompt)
    print(response)
