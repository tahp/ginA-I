import os
import sys
import asyncio
import json
from datetime import datetime
from typing import List, Dict, Optional, AsyncGenerator, Any
from ollama import AsyncClient
from dotenv import load_dotenv
from dataclasses import dataclass, field

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich.spinner import Spinner
from rich.prompt import Prompt

# Load environment variables from .env file
load_dotenv()

# --- Tools Definition ---

async def get_current_time() -> str:
    """Returns the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

async def get_weather(city: str) -> str:
    """Returns a mock weather report for a city."""
    # In a real app, this would call a weather API
    return f"The weather in {city} is currently 22°C and sunny."

TOOLS = {
    "get_current_time": get_current_time,
    "get_weather": get_weather
}

TOOL_SCHEMAS = [
    {
        'type': 'function',
        'function': {
            'name': 'get_current_time',
            'description': 'Get the current date and time',
            'parameters': {
                'type': 'object',
                'properties': {},
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'get_weather',
            'description': 'Get the current weather for a specific city',
            'parameters': {
                'type': 'object',
                'properties': {
                    'city': {
                        'type': 'string',
                        'description': 'The name of the city',
                    },
                },
                'required': ['city'],
            },
        },
    },
]

# --- Core Logic ---

@dataclass
class ChatConfig:
    """Configuration for the Chatbot."""
    model_name: str = field(default_factory=lambda: os.getenv("MODEL_NAME", "llama3"))
    history_limit: int = 10
    system_prompt: Optional[str] = field(default_factory=lambda: os.getenv("SYSTEM_PROMPT"))
    stream: bool = True
    history_file: str = "chat_history.json"
    summarize_at: int = 15

class ChatClient:
    """Asynchronous client for interacting with the Ollama API with Tool Support."""
    def __init__(self, config: ChatConfig):
        self.config = config
        self.history: List[Dict[str, Any]] = []
        self.client = AsyncClient()
        self._load_history()
        
        if not self.history and self.config.system_prompt:
            self.history.insert(0, {'role': 'system', 'content': self.config.system_prompt})

    def _load_history(self) -> None:
        """Loads history from a JSON file if it exists."""
        if os.path.exists(self.config.history_file):
            try:
                with open(self.config.history_file, 'r') as f:
                    self.history = json.load(f)
            except Exception:
                pass

    def _save_history(self) -> None:
        """Saves history to a JSON file."""
        try:
            with open(self.config.history_file, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception:
            pass

    def set_model(self, model_name: str) -> None:
        self.config.model_name = model_name

    async def list_models(self) -> List[str]:
        try:
            response = await self.client.list()
            return [m['name'] for m in response['models']]
        except Exception:
            return []

    async def get_response(self, user_input: str) -> AsyncGenerator[str, None]:
        """Sends user input to Ollama and handles tool calls if necessary."""
        self.history.append({'role': 'user', 'content': user_input})
        
        try:
            # First pass: Check for tool calls
            # Note: We disable streaming for the tool call check to simplify logic
            response = await self.client.chat(
                model=self.config.model_name,
                messages=self.history,
                tools=TOOL_SCHEMAS
            )
            
            # Process tool calls if any
            if response.get('message', {}).get('tool_calls'):
                self.history.append(response['message'])
                
                for tool_call in response['message']['tool_calls']:
                    function_name = tool_call['function']['name']
                    arguments = tool_call['function']['arguments']
                    
                    if function_name in TOOLS:
                        yield f"\n[bold cyan][System: Calling tool '{function_name}' with {arguments}...][/bold cyan]\n"
                        result = await TOOLS[function_name](**arguments)
                        self.history.append({
                            'role': 'tool',
                            'content': str(result),
                        })

                # Second pass: Get final response with tool results
                # We re-enable streaming here if configured
                if self.config.stream:
                    final_response = await self.client.chat(
                        model=self.config.model_name,
                        messages=self.history,
                        stream=True
                    )
                    full_content = ""
                    async for chunk in final_response:
                        content = chunk['message']['content']
                        full_content += content
                        yield content
                    self.history.append({'role': 'assistant', 'content': full_content})
                else:
                    final_response = await self.client.chat(
                        model=self.config.model_name,
                        messages=self.history,
                        stream=False
                    )
                    content = final_response['message']['content']
                    self.history.append({'role': 'assistant', 'content': content})
                    yield content
            else:
                # No tool calls, just stream normal response
                if self.config.stream:
                    stream_response = await self.client.chat(
                        model=self.config.model_name,
                        messages=self.history,
                        stream=True
                    )
                    full_content = ""
                    async for chunk in stream_response:
                        content = chunk['message']['content']
                        full_content += content
                        yield content
                    self.history.append({'role': 'assistant', 'content': full_content})
                else:
                    normal_response = await self.client.chat(
                        model=self.config.model_name,
                        messages=self.history,
                        stream=False
                    )
                    content = normal_response['message']['content']
                    self.history.append({'role': 'assistant', 'content': content})
                    yield content

            await self._manage_history()
            self._save_history()

        except Exception as e:
            if self.history and self.history[-1]['role'] == 'user':
                self.history.pop()
            raise e

    async def _manage_history(self) -> None:
        if len(self.history) > self.config.summarize_at:
            await self._summarize_history()
        
        start_index = 1 if self.history and self.history[0]['role'] == 'system' else 0
        while len(self.history) > self.config.history_limit + start_index:
            self.history.pop(start_index)

    async def _summarize_history(self) -> None:
        start_idx = 1 if self.history[0]['role'] == 'system' else 0
        messages_to_summarize = self.history[start_idx : start_idx + 6]
        
        summary_prompt = "Summarize the following conversation concisely:\n\n"
        for msg in messages_to_summarize:
            summary_prompt += f"{msg['role'].capitalize()}: {msg.get('content', '[Tool Call]')}\n"
        
        try:
            response = await self.client.chat(
                model=self.config.model_name,
                messages=[{'role': 'user', 'content': summary_prompt}]
            )
            summary_content = response['message']['content']
            new_history = []
            if start_idx == 1: new_history.append(self.history[0])
            new_history.append({'role': 'system', 'content': f"Summary: {summary_content}"})
            new_history.extend(self.history[start_idx + 6 :])
            self.history = new_history
        except Exception:
            pass

async def main() -> None:
    console = Console()
    config = ChatConfig()
    client = ChatClient(config)

    console.print(Panel.fit(
        f"[bold blue]Ollama Chatbot with Tools[/bold blue]\n"
        f"Model: [green]{config.model_name}[/green]\n"
        f"Tools: [cyan]get_current_time, get_weather[/cyan]",
        title="Welcome",
        border_style="blue"
    ))

    while True:
        try:
            user_input: str = Prompt.ask("\n[bold yellow]You[/bold yellow]")
            if not user_input.strip(): continue

            if user_input.lower() in ['/quit', 'quit']:
                console.print("[bold red]Goodbye![/bold red]")
                break
            
            if user_input.lower() == '/list':
                with console.status("[bold green]Fetching models..."):
                    models = await client.list_models()
                console.print("\n[bold cyan]Available models:[/bold cyan]")
                for m in models: console.print(f" • {m}")
                continue

            if user_input.lower().startswith('/model '):
                new_model = user_input[7:].strip()
                if new_model:
                    client.set_model(new_model)
                    console.print(f"[bold green]Switched to {new_model}[/bold green]")
                continue
            
            console.print("\n[bold blue]Chatbot:[/bold blue]")
            full_response = ""
            
            with Live(Text("Thinking..."), refresh_per_second=10, console=console) as live:
                async for chunk in client.get_response(user_input):
                    full_response += chunk
                    live.update(Markdown(full_content=full_response))

        except KeyboardInterrupt:
            console.print("\n[bold red]Goodbye![/bold red]")
            break
        except Exception as e:
            console.print(f"\n[bold red]Error:[/bold red] {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
