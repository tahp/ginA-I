import os
import sys
import asyncio
import json
from typing import List, Dict, Optional, AsyncGenerator
from ollama import AsyncClient
from dotenv import load_dotenv
from dataclasses import dataclass, field

# Load environment variables from .env file
load_dotenv()

@dataclass
class ChatConfig:
    """Configuration for the Chatbot."""
    model_name: str = field(default_factory=lambda: os.getenv("MODEL_NAME", "llama3"))
    history_limit: int = 10
    system_prompt: Optional[str] = field(default_factory=lambda: os.getenv("SYSTEM_PROMPT"))
    stream: bool = True
    history_file: str = "chat_history.json"
    summarize_at: int = 15  # Trigger summarization when history exceeds this

class ChatClient:
    """Asynchronous client for interacting with the Ollama API."""
    def __init__(self, config: ChatConfig):
        self.config = config
        self.history: List[Dict[str, str]] = []
        self.client = AsyncClient()
        self._load_history()
        
        # If history is empty and we have a system prompt, add it
        if not self.history and self.config.system_prompt:
            self.history.insert(0, {'role': 'system', 'content': self.config.system_prompt})

    def _load_history(self) -> None:
        """Loads history from a JSON file if it exists."""
        if os.path.exists(self.config.history_file):
            try:
                with open(self.config.history_file, 'r') as f:
                    self.history = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load history from {self.config.history_file}: {e}")

    def _save_history(self) -> None:
        """Saves history to a JSON file."""
        try:
            with open(self.config.history_file, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save history to {self.config.history_file}: {e}")

    def set_model(self, model_name: str) -> None:
        """Updates the active model."""
        self.config.model_name = model_name
        print(f"Model switched to: {model_name}")

    async def list_models(self) -> List[str]:
        """Lists available local models."""
        try:
            response = await self.client.list()
            return [m['name'] for m in response['models']]
        except Exception:
            return []

    async def get_response(self, user_input: str) -> AsyncGenerator[str, None]:
        """Sends user input to Ollama and yields/returns the AI response asynchronously."""
        self.history.append({'role': 'user', 'content': user_input})
        
        try:
            full_response = ""
            if self.config.stream:
                response = await self.client.chat(model=self.config.model_name, messages=self.history, stream=True)
                async for chunk in response:
                    content = chunk['message']['content']
                    full_response += content
                    yield content
                self.history.append({'role': 'assistant', 'content': full_response})
                await self._manage_history()
                self._save_history()
            else:
                response = await self.client.chat(model=self.config.model_name, messages=self.history, stream=False)
                full_response = response['message']['content']
                self.history.append({'role': 'assistant', 'content': full_response})
                await self._manage_history()
                self._save_history()
                yield full_response
        except Exception as e:
            if self.history and self.history[-1]['role'] == 'user':
                self.history.pop()  # Remove the user input if it failed
            raise e

    async def _manage_history(self) -> None:
        """Manages history by trimming or summarising."""
        if len(self.history) > self.config.summarize_at:
            await self._summarize_history()
        
        # Fallback trimming if summarization didn't reduce it enough or is disabled
        start_index = 0
        if self.history and self.history[0]['role'] == 'system':
            start_index = 1
        
        while len(self.history) > self.config.history_limit + start_index:
            self.history.pop(start_index)

    async def _summarize_history(self) -> None:
        """Summarizes the middle part of the history to save context."""
        print("\n[System: Summarizing older conversation to save space...]")
        
        # We keep the system prompt (index 0) and summarize the next few messages
        start_idx = 1 if self.history[0]['role'] == 'system' else 0
        messages_to_summarize = self.history[start_idx : start_idx + 6]
        
        summary_prompt = "Summarize the following part of a conversation concisely, focusing on key facts and requests:\n\n"
        for msg in messages_to_summarize:
            summary_prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"
        
        try:
            response = await self.client.chat(
                model=self.config.model_name,
                messages=[{'role': 'user', 'content': summary_prompt}]
            )
            summary_content = response['message']['content']
            
            # Replace the summarized messages with a single summary message
            new_history = []
            if start_idx == 1:
                new_history.append(self.history[0]) # Keep system prompt
            
            new_history.append({'role': 'system', 'content': f"Summary of previous conversation: {summary_content}"})
            new_history.extend(self.history[start_idx + 6 :])
            self.history = new_history
            
        except Exception as e:
            print(f"Warning: Summarization failed: {e}")

async def main() -> None:
    """
    Main entry point of the chatbot program.
    Runs an asynchronous loop to take user input and provide responses via Ollama API.
    """
    config = ChatConfig()
    client = ChatClient(config)

    print(f"Chatbot initialized with model: {config.model_name}")
    print("Commands: /quit, /model <name>, /list")

    while True:
        try:
            user_input: str = input("\nYou: ").strip()
            
            if not user_input:
                continue

            # Command Routing
            if user_input.lower() in ['/quit', 'quit']:
                print("Chatbot: Goodbye!")
                break
            
            if user_input.lower() == '/list':
                models = await client.list_models()
                print("Available models:")
                for m in models:
                    print(f" - {m}")
                continue

            if user_input.lower().startswith('/model '):
                new_model = user_input[7:].strip()
                if new_model:
                    client.set_model(new_model)
                continue
            
            # Normal chat
            print("Chatbot: ", end="", flush=True)
            if config.stream:
                async for chunk in client.get_response(user_input):
                    print(chunk, end="", flush=True)
                print()
            else:
                async for chunk in client.get_response(user_input):
                    print(chunk)
                    break

        except KeyboardInterrupt:
            print("\nChatbot: Goodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
