import asyncio
from typing import Any, Optional, Union, Callable, List
from google import genai
from google.genai import types

from .reducer import BaseReducer, reduce_critic

def _get_temperatures(n: int) -> List[float]:
    """
    Calculate the distributed temperatures from 0.0 to 1.0.
    
    Args:
        n: Number of instances.
    """
    if n == 1:
        return [0.0]
    return [0.0 + (1.0 / (n - 1)) * i for i in range(n)]


def _prepare_generation_config(
    base_config: Optional[Any],
    response_schema: Optional[Any],
    temperature: float
) -> types.GenerateContentConfig:
    """
    Prepare and construct a GenerateContentConfig object with custom temperature and schema.
    
    Args:
        base_config: User provided configuration dict or object.
        response_schema: Optional structured schema (Pydantic).
        temperature: Scattered temperature to assign.
    """
    config_params = {}
    if base_config is not None:
        if isinstance(base_config, dict):
            config_params = base_config.copy()
        elif hasattr(base_config, "model_dump"):
            config_params = base_config.model_dump(exclude_none=True)
        else:
            config_params = {
                k: getattr(base_config, k)
                for k in dir(base_config)
                if not k.startswith("_") and getattr(base_config, k) is not None
            }

    if response_schema is not None:
        config_params["response_schema"] = response_schema
        config_params["response_mime_type"] = "application/json"

    config_params["temperature"] = temperature
    return types.GenerateContentConfig(**config_params)


async def _execute_reduction(
    client: genai.Client,
    strategy: Optional[Union[BaseReducer, Callable]],
    model: str,
    prompt: str,
    outputs: List[str],
    response_schema: Optional[Any],
    output_language: Optional[str],
    **kwargs
) -> Any:
    """
    Execute the reduction step using the resolved strategy.
    
    Args:
        client: The google-genai Client.
        strategy: Reduction class instance or callable function.
        model: Base model name.
        prompt: Original prompt.
        outputs: Parallel generated outputs list.
        response_schema: Pydantic response schema.
        output_language: Target output language name.
    """
    if strategy is None:
        return await reduce_critic(
            client=client,
            fallback_model=model,
            prompt=prompt,
            outputs=outputs,
            response_schema=response_schema,
            output_language=output_language,
            **kwargs
        )
    
    if hasattr(strategy, "reduce"):
        return await strategy.reduce(
            client=client,
            fallback_model=model,
            prompt=prompt,
            outputs=outputs,
            response_schema=response_schema,
            output_language=output_language,
            **kwargs
        )
    
    if callable(strategy):
        return await strategy(
            client=client,
            fallback_model=model,
            prompt=prompt,
            outputs=outputs,
            response_schema=response_schema,
            output_language=output_language,
            **kwargs
        )
        
    raise TypeError("strategy must be a BaseReducer instance, a callable function, or None")


async def generate_ensemble(
    client: genai.Client,
    prompt: str,
    model: str = "gemini-3.5-flash",
    n: int = 3,
    strategy: Optional[Union[BaseReducer, Callable]] = None,
    response_schema: Optional[Any] = None,
    output_language: Optional[str] = None,
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
        output_language: Optional target language for the final output (e.g., 'Japanese', 'English').
        **kwargs: Additional parameters passed to both base and reducer generation calls.
    """
    if n < 1:
        raise ValueError("Number of parallel instances (n) must be at least 1.")

    temperatures = _get_temperatures(n)
    base_config = kwargs.get("config")

    # Spawn parallel async generation tasks
    tasks = []
    for temp in temperatures:
        config = _prepare_generation_config(base_config, response_schema, temp)
        task = client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=config
        )
        tasks.append(task)

    # Gather parallel results
    responses = await asyncio.gather(*tasks)
    outputs = [resp.text for resp in responses]

    # Run reduction
    return await _execute_reduction(
        client=client,
        strategy=strategy,
        model=model,
        prompt=prompt,
        outputs=outputs,
        response_schema=response_schema,
        output_language=output_language,
        **kwargs
    )


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
        output_language: Optional[str] = None,
        **kwargs
    ) -> Any:
        return await generate_ensemble(
            client=self.client,
            prompt=prompt,
            model=model,
            n=n,
            strategy=strategy,
            response_schema=response_schema,
            output_language=output_language,
            **kwargs
        )
