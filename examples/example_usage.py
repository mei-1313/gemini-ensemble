import asyncio
import os
from google import genai
from pydantic import BaseModel
from gemini_ensemble import (
    EnsembleClient,
    CriticReducer,
    VotingReducer,
    generate_ensemble,
    reduce_critic,
    reduce_voting,
)

# Setup a Pydantic model for structured output demonstration
class SentimentReport(BaseModel):
    sentiment: str  # positive, negative, or neutral
    confidence: float
    reason: str

async def main():
    """
    Example demonstrating how to use the gemini-ensemble library
    for both text synthesis and structured consensus validation, with Japanese outputs.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Warning: GEMINI_API_KEY environment variable is not set.")
        print("Please set it to run actual API calls, or run tests with mock clients.")
        return

    print("Initializing genai.Client...")
    base_client = genai.Client()
    ensemble = EnsembleClient(client=base_client)
    
    prompt = (
        "Analyze the sentiment of the following customer review and provide a rationale:\n"
        "\"The product build quality is premium and sturdy, but the battery life is slightly below average. "
        "Also, customer service was helpful but took three days to respond.\""
    )
    
    print("\n1. Running ensemble with n=3, CriticReducer, and output_language='Japanese' (Text Synthesis)...")
    try:
        response = await ensemble.generate(
            prompt=prompt,
            model="gemini-3.1-flash-lite", 
            n=3,
            strategy=CriticReducer(reducer_model="gemini-3.1-flash-lite"),
            output_language="Japanese"
        )
        print("--- Critic Reducer Response ---")
        print(response.text)
    except Exception as e:
        print(f"Failed to execute: {e}")
    
    print("\n2. Running ensemble with n=3, VotingReducer, response_schema, and output_language='Japanese'...")
    try:
        response_structured = await ensemble.generate(
            prompt=prompt,
            model="gemini-3.1-flash-lite",
            n=3,
            strategy=VotingReducer(reducer_model="gemini-3.1-flash-lite"),
            response_schema=SentimentReport,
            output_language="Japanese"
        )
        print("--- Voting Reducer Response (Structured) ---")
        print(response_structured.text)
    except Exception as e:
        print(f"Failed to execute: {e}")

    print("\n3. Running pure functional ensemble with generate_ensemble, reduce_voting, and output_language='Japanese'...")
    try:
        response_functional = await generate_ensemble(
            client=base_client,
            prompt=prompt,
            model="gemini-3.1-flash-lite",
            n=3,
            strategy=reduce_voting,
            response_schema=SentimentReport,
            output_language="Japanese",
            reducer_model="gemini-3.1-flash-lite"
        )
        print("--- Functional Voting Reducer Response (Structured) ---")
        print(response_functional.text)
    except Exception as e:
        print(f"Failed to execute: {e}")

    print("\n4. Running pure functional ensemble with output_language='Japanese'...")
    try:
        response_japanese = await generate_ensemble(
            client=base_client,
            prompt=prompt,
            model="gemini-3.1-flash-lite",
            n=3,
            strategy=reduce_critic,
            output_language="Japanese",
            reducer_model="gemini-3.1-flash-lite"
        )
        print("--- Critic Reducer Response (in Japanese) ---")
        print(response_japanese.text)
    except Exception as e:
        print(f"Failed to execute: {e}")

if __name__ == "__main__":
    asyncio.run(main())
