"""
Pytest fixtures for agentic tests.
"""

import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def mock_litellm():
    """
    Mock litellm.completion for testing without API calls.
    
    This fixture patches litellm.completion to return a mock response,
    allowing tests to run without actual LLM API calls.
    
    Usage:
        def test_agent_run(mock_litellm):
            agent = Agent(system_prompt="You are helpful")
            output = agent.run("Hello")
            assert output.content == "Test response"
    """
    with patch("agentic.agent.agent.litellm") as mock:
        # Create a mock response object
        mock_message = MagicMock()
        mock_message.content = "Test response"
        
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20
        mock_usage.total_tokens = 30
        
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage
        
        # Set up both sync and async returns
        mock.completion.return_value = mock_response
        mock.acompletion.return_value = mock_response
        
        yield mock


@pytest.fixture
def mock_litellm_error():
    """
    Mock litellm.completion to raise an error.
    
    Useful for testing error handling in agent execution.
    
    Usage:
        def test_agent_error_handling(mock_litellm_error):
            agent = Agent(system_prompt="You are helpful")
            output = agent.run("Hello")
            assert output.status == ExecutionStatus.FAILED
    """
    with patch("agentic.agent.agent.litellm") as mock:
        mock.completion.side_effect = Exception("LLM API Error")
        mock.acompletion.side_effect = Exception("LLM API Error")
        yield mock


@pytest.fixture
def sample_agent():
    """
    Create a sample agent for testing.
    
    Usage:
        def test_something(sample_agent, mock_litellm):
            output = sample_agent.run("Hello")
    """
    from agentic import Agent
    
    return Agent(
        model="gpt-4o-mini",
        system_prompt="You are a helpful assistant.",
        name="test-agent",
    )

