import openai
import anthropic
from app.config import (
    OPENAI_API_KEY, ANTHROPIC_API_KEY,
    OPENAI_MODEL, ANTHROPIC_MODEL, DEFAULT_LLM_PROVIDER
)


def call_openai(system_prompt: str, user_message: str) -> str:
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0.1
    )
    return response.choices[0].message.content # type: ignore


def call_anthropic(system_prompt: str, user_message: str) -> str:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=2048,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_message}
        ],
        temperature=0.1
    )
    return response.content[0].text # type: ignore


def call_llm(system_prompt: str, user_message: str, provider: str = DEFAULT_LLM_PROVIDER) -> str:
    """
    Unified LLM interface. Swap providers via config or per-call.
    
    This is the adapter pattern — same interface, different implementations.
    Same approach used at Airtap for multi-model orchestration.
    """
    if provider == "openai":
        return call_openai(system_prompt, user_message)
    elif provider == "anthropic":
        return call_anthropic(system_prompt, user_message)
    else:
        raise ValueError(f"Unsupported provider: {provider}. Use 'openai' or 'anthropic'.")