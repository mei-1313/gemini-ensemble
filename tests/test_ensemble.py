import pytest
from unittest.mock import MagicMock, AsyncMock
from google import genai
from google.genai import types
from pydantic import BaseModel

from gemini_ensemble import EnsembleClient, CriticReducer, VotingReducer

class DummySchema(BaseModel):
    score: float

@pytest.mark.asyncio
async def test_ensemble_client_temperature_scattering():
    """
    Test that the EnsembleClient correctly distributes temperatures from 0.0 to 1.0
    and executes the default CriticReducer reduction strategy.
    """
    mock_client = MagicMock(spec=genai.Client)
    mock_aio = MagicMock()
    mock_models = MagicMock()
    mock_generate = AsyncMock()
    
    mock_client.aio = mock_aio
    mock_aio.models = mock_models
    mock_models.generate_content = mock_generate
    
    mock_response = MagicMock()
    mock_response.text = "Result text"
    mock_generate.return_value = mock_response

    ensemble = EnsembleClient(client=mock_client)
    
    response = await ensemble.generate(
        prompt="Explain quantum physics.",
        model="gemini-3.5-flash",
        n=3
    )
    
    # Total calls: 3 (parallel layer) + 1 (reducer layer) = 4
    assert mock_generate.call_count == 4
    assert response == mock_response

    # Verify the temperature distribution in the parallel layer
    temperatures = []
    for call in mock_generate.call_args_list[:3]:
        config = call.kwargs.get("config")
        temperatures.append(config.temperature)
    
    assert temperatures == [0.0, 0.5, 1.0]

    # Verify that the reducer was called with the default fallback model and temperature 0.0
    reducer_call = mock_generate.call_args_list[3]
    assert reducer_call.kwargs.get("model") == "gemini-3.5-flash"
    reducer_config = reducer_call.kwargs.get("config")
    assert reducer_config.temperature == 0.0


@pytest.mark.asyncio
async def test_ensemble_client_custom_reducer_and_schema():
    """
    Test EnsembleClient with a custom reducer model, a custom strategy (VotingReducer),
    and a specific Pydantic response_schema.
    """
    mock_client = MagicMock(spec=genai.Client)
    mock_aio = MagicMock()
    mock_models = MagicMock()
    mock_generate = AsyncMock()
    
    mock_client.aio = mock_aio
    mock_aio.models = mock_models
    mock_models.generate_content = mock_generate
    
    mock_response = MagicMock()
    mock_response.text = '{"score": 0.95}'
    mock_generate.return_value = mock_response

    ensemble = EnsembleClient(client=mock_client)
    
    response = await ensemble.generate(
        prompt="Analyze the security logs.",
        model="gemini-3.1-flash-lite",
        n=2,
        strategy=VotingReducer(reducer_model="gemini-3.5-flash"),
        response_schema=DummySchema
    )
    
    # Total calls: 2 (parallel layer) + 1 (reducer layer) = 3
    assert mock_generate.call_count == 3
    assert response == mock_response

    # Verify parallel layer configuration
    for call in mock_generate.call_args_list[:2]:
        config = call.kwargs.get("config")
        assert config.response_schema == DummySchema
        assert config.response_mime_type == "application/json"
    
    # Verify reducer layer configuration
    reducer_call = mock_generate.call_args_list[2]
    assert reducer_call.kwargs.get("model") == "gemini-3.5-flash"
    reducer_config = reducer_call.kwargs.get("config")
    assert reducer_config.response_schema == DummySchema
    assert reducer_config.response_mime_type == "application/json"
    assert reducer_config.temperature == 0.0
