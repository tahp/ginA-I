import os
from typing import List, Dict, Optional
import ollama
from dotenv import load_dotenv
from dataclasses import dataclass, field

# Load environment variables from .env file
load_dotenv()

@dataclass
class ChatConfig:
    """Configuration for the Chatbot."""
    model_name: str = field(default_factory=lambda: os.getenv("MODEL_NAME", "llama3"))
    history_limit: int = 10
    system_prompt: Optional[str] = None

class ChatClient:
    """Client for interacting with the Ollama API."""
    def __init__(self, config: ChatConfig):
        self.config = config
        self.history: List[Dict[str, str]] = []
        if self.config.system_prompt:
            self.history.append({'role': 'system', 'content': self.config.system_prompt})

    def get_response(self, user_input: str) -> str:
        """Sends user input to Ollama and returns the AI response."""
        self.history.append({'role': 'user', 'content': user_input})
        
        try:
            response = ollama.chat(model=self.config.model_name, messages=self.history)
            ai_response: str = response['message']['content']
            self.history.append({'role': 'assistant', 'content': ai_response})
            self._trim_history()
            return ai_response
        except Exception as e:
            self.history.pop()  # Remove the user input if it failed
            raise e

    def _trim_history(self) -> None:
        """Keeps history within the configured limit."""
        # System prompt should always stay if present
        start_index = 1 if self.config.system_prompt else 0
        while len(self.history) > self.config.history_limit + start_index:
            self.history.pop(start_index)

def main() -> None:
    """
    Main entry point of the chatbot program.
    Runs an infinite loop to take user input and provide responses via Ollama API.
    """
    config = ChatConfig()
    client = ChatClient(config)

    print(f"Chatbot initialized with model: {config.model_name}")
    print("Type 'quit' to exit.")

    while True:
        try:
            user_input: str = input("You: ").strip()
            
            if not user_input:
                continue

            if user_input.lower() == 'quit':
                print("Chatbot: Goodbye!")
                break
            
            ai_response = client.get_response(user_input)
            print(f"Chatbot: {ai_response}")

        except KeyboardInterrupt:
            print("\nChatbot: Goodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
