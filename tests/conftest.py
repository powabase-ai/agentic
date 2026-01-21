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
def mock_litellm_streaming():
    """
    Mock litellm.completion for streaming tests.
    
    This fixture patches litellm to return a mock streaming response,
    simulating chunks arriving one at a time.
    
    Usage:
        def test_agent_stream(mock_litellm_streaming):
            agent = Agent(system_prompt="You are helpful")
            for chunk in agent.stream("Hello"):
                print(chunk)  # Will print "Hello", " ", "World", "!"
    """
    with patch("agentic.agent.agent.litellm") as mock:
        # Create mock chunks for streaming
        def create_chunk(content):
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta = MagicMock()
            chunk.choices[0].delta.content = content
            return chunk
        
        # Create chunks that simulate streaming "Hello World!"
        def get_chunks():
            return [
                create_chunk("Hello"),
                create_chunk(" "),
                create_chunk("World"),
                create_chunk("!"),
            ]
        
        # Mock sync streaming - returns an iterable (new iterator each call)
        mock.completion.side_effect = lambda **kwargs: iter(get_chunks())
        
        # Mock async streaming - returns an awaitable that returns async iterable
        async def async_completion(**kwargs):
            async def async_chunks():
                for chunk in get_chunks():
                    yield chunk
            return async_chunks()
        
        mock.acompletion.side_effect = async_completion
        
        yield mock


@pytest.fixture
def mock_litellm_streaming_error():
    """
    Mock litellm.completion to raise an error during streaming.
    
    Usage:
        def test_agent_stream_error(mock_litellm_streaming_error):
            agent = Agent(system_prompt="You are helpful")
            gen = agent.stream("Hello")
            # Error occurs during iteration
    """
    with patch("agentic.agent.agent.litellm") as mock:
        mock.completion.side_effect = Exception("LLM Streaming Error")
        mock.acompletion.side_effect = Exception("LLM Streaming Error")
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

