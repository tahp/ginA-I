import os
import sys
from typing import List, Dict, Optional, Generator
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
    stream: bool = True

class ChatClient:
    """Client for interacting with the Ollama API."""
    def __init__(self, config: ChatConfig):
        self.config = config
        self.history: List[Dict[str, str]] = []
        if self.config.system_prompt:
            self.history.append({'role': 'system', 'content': self.config.system_prompt})

    def get_response(self, user_input: str) -> Generator[str, None, str]:
        """Sends user input to Ollama and yields/returns the AI response."""
        self.history.append({'role': 'user', 'content': user_input})
        
        try:
            full_response = ""
            if self.config.stream:
                response = ollama.chat(model=self.config.model_name, messages=self.history, stream=True)
                for chunk in response:
                    content = chunk['message']['content']
                    full_response += content
                    yield content
                self.history.append({'role': 'assistant', 'content': full_response})
                self._trim_history()
                return full_response
            else:
                response = ollama.chat(model=self.config.model_name, messages=self.history, stream=False)
                full_response = response['message']['content']
                self.history.append({'role': 'assistant', 'content': full_response})
                self._trim_history()
                return full_response
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
            user_input: str = input("\nYou: ").strip()
            
            if not user_input:
                continue

            if user_input.lower() == 'quit':
                print("Chatbot: Goodbye!")
                break
            
            print("Chatbot: ", end="", flush=True)
            if config.stream:
                for chunk in client.get_response(user_input):
                    print(chunk, end="", flush=True)
                print()
            else:
                ai_response = next(client.get_response(user_input))
                print(ai_response)

        except KeyboardInterrupt:
            print("\nChatbot: Goodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
