import unittest
import asyncio
from unittest.mock import AsyncMock, patch
import sys
import os

# Add src to path so we can import chatbot
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from chatbot import ChatClient, ChatConfig

class TestChatClientAdvanced(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # history_limit is 10, summarize_at is 15. 
        # For testing, let's make them smaller to trigger easily.
        self.config = ChatConfig(model_name="test-model", history_limit=5, summarize_at=8)
        self.config.history_file = "test_history_unittest.json"
        if os.path.exists(self.config.history_file):
            os.remove(self.config.history_file)

    def tearDown(self):
        if os.path.exists(self.config.history_file):
            os.remove(self.config.history_file)

    @patch('ollama.AsyncClient.chat')
    async def test_summarization_trigger(self, mock_chat):
        # Mock for summarization response
        mock_chat.return_value = {'message': {'content': 'This is a summary.'}}
        
        client = ChatClient(self.config)
        # Manually set history to avoid initialization logic if needed
        # Initial history might have system prompt if configured in env
        # Let's ensure it has exactly what we expect.
        client.history = [
            {'role': 'system', 'content': 'sys'},
            {'role': 'user', 'content': '1'},
            {'role': 'assistant', 'content': '1a'},
            {'role': 'user', 'content': '2'},
            {'role': 'assistant', 'content': '2a'},
            {'role': 'user', 'content': '3'},
            {'role': 'assistant', 'content': '3a'},
            {'role': 'user', 'content': '4'} # Total 8
        ]
        
        # history_limit is 5. If we run _manage_history now, it will TRIM down to 6 (system + 5)
        # because summarize_at is 8 (8 is not > 8).
        
        await client._manage_history() 
        self.assertEqual(len(client.history), 6) # sys + 5 most recent
        
        # Reset and test actual summarization
        client.history = [
            {'role': 'system', 'content': 'sys'},
            {'role': 'user', 'content': '1'},
            {'role': 'assistant', 'content': '1a'},
            {'role': 'user', 'content': '2'},
            {'role': 'assistant', 'content': '2a'},
            {'role': 'user', 'content': '3'},
            {'role': 'assistant', 'content': '3a'},
            {'role': 'user', 'content': '4'},
            {'role': 'assistant', 'content': '4a'} # Total 9
        ]
        
        await client._manage_history() # Should trigger (9 > 8)
        
        # It takes index 1 to 7 (6 messages). 
        # History was: [sys, 1, 1a, 2, 2a, 3, 3a, 4, 4a]
        # Summarized: [1, 1a, 2, 2a, 3, 3a]
        # New history: [sys, summary, 4, 4a] -> Length 4
        self.assertEqual(client.history[1]['content'], 'Summary of previous conversation: This is a summary.')
        self.assertEqual(len(client.history), 4)

    async def test_trim_fallback(self):
        # Test that it still trims if summarization is somehow bypassed or not enough
        self.config.summarize_at = 100 # effectively disable for this test
        client = ChatClient(self.config)
        client.history = [{'role': 'user', 'content': str(i)} for i in range(20)]
        await client._manage_history()
        self.assertEqual(len(client.history), 5) # history_limit is 5

if __name__ == '__main__':
    unittest.main()
