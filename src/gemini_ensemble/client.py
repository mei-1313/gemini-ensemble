import asyncio
from typing import Any, Optional, Union, Callable
from google import genai
from google.genai import types

from .reducer import BaseReducer, reduce_critic

async def generate_ensemble(
    client: genai.Client,
    prompt: str,
    model: str = "gemini-3.5-flash",
    n: int = 3,
    strategy: Optional[Union[BaseReducer, Callable]] = None,
    response_schema: Optional[Any] = None,
    **kwargs
) -> Any:
    """
    Pure function to execute n parallel requests to the specified base model with temperature scattering
    ranging from 0.0 to 1.0, and reduce the results using the specified strategy.

    Args:
        client: An instance of google.genai.Client.
        prompt: The input prompt string.
        model: The name of the base model (e.g., 'gemini-3.5-flash').
        n: The number of parallel base model calls. Must be at least 1.
        strategy: A reduction strategy. Can be a BaseReducer instance, a callable function, or None (defaults to reduce_critic).
        response_schema: Optional Pydantic model class for structured JSON output.
        **kwargs: Additional parameters passed to both base and reducer generation calls.
    """
    if n < 1:
        raise ValueError("Number of parallel instances (n) must be at least 1.")

    # 1. Determine the temperatures for the parallel layer
    if n == 1:
        temperatures = [0.0]
    else:
        temperatures = [0.0 + (1.0 / (n - 1)) * i for i in range(n)]

    # 2. Extract and prepare base configuration from kwargs
    base_config = kwargs.get("config")
    config_params = {}
    if base_config is not None:
        if isinstance(base_config, dict):
            config_params = base_config.copy()
        elif hasattr(base_config, "model_dump"):
            config_params = base_config.model_dump(exclude_none=True)
        else:
            config_params = {k: getattr(base_config, k) for k in dir(base_config) if not k.startswith("_") and getattr(base_config, k) is not None}

    # If schema is provided, enforce json output structure in the parallel layer
    if response_schema is not None:
        config_params["response_schema"] = response_schema
        config_params["response_mime_type"] = "application/json"

    # 3. Spawn parallel async generation tasks
    tasks = []
    for temp in temperatures:
        loop_config_params = config_params.copy()
        loop_config_params["temperature"] = temp
        config = types.GenerateContentConfig(**loop_config_params)
        
        task = client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=config
        )
        tasks.append(task)

    # Gather parallel results
    responses = await asyncio.gather(*tasks)
    
    # Extract the raw text from responses for aggregation
    outputs = [resp.text for resp in responses]

    # 4. Resolve reduction strategy
    if strategy is None:
        final_response = await reduce_critic(
            client=client,
            fallback_model=model,
            prompt=prompt,
            outputs=outputs,
            response_schema=response_schema,
            **kwargs
        )
    elif hasattr(strategy, "reduce"):
        # If strategy is a BaseReducer class instance
        final_response = await strategy.reduce(
            client=client,
            fallback_model=model,
            prompt=prompt,
            outputs=outputs,
            response_schema=response_schema,
            **kwargs
        )
    elif callable(strategy):
        # If strategy is a pure function
        final_response = await strategy(
            client=client,
            fallback_model=model,
            prompt=prompt,
            outputs=outputs,
            response_schema=response_schema,
            **kwargs
        )
    else:
        raise TypeError("strategy must be a BaseReducer instance, a callable function, or None")

    return final_response


# --- Compatibility Wrapper ---

class EnsembleClient:
    """
    EnsembleClient compatibility wrapper class.
    """
    def __init__(self, client: genai.Client):
        self.client = client

    async def generate(
        self,
        prompt: str,
        model: str = "gemini-3.5-flash",
        n: int = 3,
        strategy: Optional[Union[BaseReducer, Callable]] = None,
        response_schema: Optional[Any] = None,
        **kwargs
    ) -> Any:
        return await generate_ensemble(
            client=self.client,
            prompt=prompt,
            model=model,
            n=n,
            strategy=strategy,
            response_schema=response_schema,
            **kwargs
        )
