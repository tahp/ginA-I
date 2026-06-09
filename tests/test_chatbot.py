import unittest
import asyncio
from unittest.mock import AsyncMock, patch
import sys
import os

# Add src to path so we can import chatbot
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from chatbot import ChatClient, ChatConfig

class TestChatClientAsync(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.config = ChatConfig(model_name="test-model", history_limit=2)
        self.config.history_file = "test_history.json"
        if os.path.exists(self.config.history_file):
            os.remove(self.config.history_file)

    def tearDown(self):
        if os.path.exists(self.config.history_file):
            os.remove(self.config.history_file)

    @patch('ollama.AsyncClient.chat')
    async def test_get_response_updates_history_async(self, mock_chat):
        # Mock response for non-streaming
        mock_chat.return_value = {'message': {'content': 'Hello!'}}
        
        client = ChatClient(self.config)
        client.config.stream = False
        
        # Test async generator
        responses = []
        async for chunk in client.get_response("Hi"):
            responses.append(chunk)
        
        self.assertEqual("".join(responses), 'Hello!')
        self.assertEqual(len(client.history), 2)
        self.assertEqual(client.history[0]['role'], 'user')
        self.assertEqual(client.history[1]['role'], 'assistant')

    async def test_trim_history(self):
        client = ChatClient(self.config)
        client.history = [
            {'role': 'user', 'content': '1'},
            {'role': 'assistant', 'content': '1a'},
            {'role': 'user', 'content': '2'},
            {'role': 'assistant', 'content': '2a'},
            {'role': 'user', 'content': '3'}
        ]
        client._trim_history()
        self.assertEqual(len(client.history), 2)
        self.assertEqual(client.history[0]['content'], '2a')
        self.assertEqual(client.history[1]['content'], '3')

if __name__ == '__main__':
    unittest.main()
