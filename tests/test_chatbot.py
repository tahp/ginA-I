import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add src to path so we can import chatbot
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from chatbot import ChatClient, ChatConfig

class TestChatClient(unittest.TestCase):
    def setUp(self):
        self.config = ChatConfig(model_name="test-model", history_limit=2)
        # Mock history file to avoid side effects
        self.config.history_file = "test_history.json"
        if os.path.exists(self.config.history_file):
            os.remove(self.config.history_file)

    def tearDown(self):
        if os.path.exists(self.config.history_file):
            os.remove(self.config.history_file)

    @patch('ollama.chat')
    def test_get_response_updates_history(self, mock_chat):
        # Mock response
        mock_chat.return_value = {'message': {'content': 'Hello!'}}
        
        client = ChatClient(self.config)
        client.config.stream = False
        
        response = next(client.get_response("Hi"))
        
        self.assertEqual(response, 'Hello!')
        self.assertEqual(len(client.history), 2)
        self.assertEqual(client.history[0]['role'], 'user')
        self.assertEqual(client.history[1]['role'], 'assistant')

    def test_trim_history(self):
        client = ChatClient(self.config)
        client.history = [
            {'role': 'user', 'content': '1'},
            {'role': 'assistant', 'content': '1a'},
            {'role': 'user', 'content': '2'},
            {'role': 'assistant', 'content': '2a'},
            {'role': 'user', 'content': '3'}
        ]
        client._trim_history()
        # limit is 2, so it should keep last 2 messages? 
        # Actually my logic was: while len(history) > history_limit + start_index
        # history_limit=2, start_index=0. So it should keep last 2.
        self.assertEqual(len(client.history), 2)
        self.assertEqual(client.history[0]['content'], '2a')
        self.assertEqual(client.history[1]['content'], '3')

if __name__ == '__main__':
    unittest.main()
