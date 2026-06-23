import asyncio
import ollama
import datetime
import platform

# --- CONFIGURATION ---
MODEL = 'hf.co/Jiunscled/supergemma4-26b-uncensored-gguf-v2:Q4_K_M'
SYSTEM_PROMPT = (
    "You are a helpful, witty assistant named Gemma-Bot. "
    "Your personality is sophisticated but charming. Use emojis sparingly "
    "but effectively to show emotion. You are clever and slightly playful."
)

class GemmaBot:
    def __init__(self, model=MODEL, system_prompt=SYSTEM_PROMPT):
        """Initializes the bot with its identity and environment context."""
        self.model = model
        self.system_prompt = system_prompt
        # Initialize messages with the system prompt
        self.messages = [{'role': 'system', 'content': self.system_prompt}]
        self.env_info = self._get_environment_context()
        try:
            self.client = ollama.AsyncClient()
            async def _test_connection():
                try:
                    await self.client.chat(model='llama3', messages=[{'role': 'user', 'content': 'ping'}])
                except Exception as e:
                    print(f"[Setup]: OLLAMA service is running. Model check skipped.")
                    return True
            import asyncio as aio
            aio.run(_test_connection())
        except Exception as e:
            err_msg = str(e).replace('\n', ': ') or "Connection failed"
            print(f"[Error]: {err_msg}")

    def _get_model_info(self):
        """Check if model exists on local server."""
        try:
            response = await self.client.list()
            return next((m['name'] for m in response.get('models', []) 
                         if 'llama3' not in str(m) and 'gemma' not in str(m)), None)
        except Exception as e:
            print(f"[Info]: Cannot verify models (Error: {e})")
            return None

    def _get_environment_context(self):
        """Generates a string containing the current time and OS for the bot."""
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        os_name = platform.system()
        return f"(Current Time: {now}, Operating System: {os_name})"

    async def _summarize(self):
        """Takes the oldest messages and turns them into one summary."""
        # We take everything from index 1 up to (but not including) the last 10.
        to_summarize = self.messages[1:-10]

        if not to_summarize:
            return "The conversation is just beginning."

        context_str = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in to_summarize])
        prompt = f"Summarize the key context from these conversation logs into one concise sentence: {context_str}"

        try:
            # Use await with the async client
            summary_response = await self.client.chat(model=self.model, messages=[{'role': 'user', 'content': prompt}]) 
            if not summary_response or 'message' not in summary_response:
                return "The conversation is still fresh."
            content = summary_response['message']['content']
        except Exception as e:
            err_msg = str(e).replace('\n', ': ') or "Failed to summarize"
            print(f"[Info]: Summarization error (Error: {err_msg})")
            return "The conversation is still fresh."

    async def _condense_history(self):
        """Triggers the summarization and rebuilds the message history."""
        summary = await self._summarize()
        print(f"\n[System]: Condensing memories... (Summary: {summary})")

        # New structure: [System] + [The Summary as a context-setting system message] + [Last 10 messages]
        new_history = [
            self.messages[0],
            {'role': 'system', 'content': f"Context from previous conversation: {summary}"}
        ]

        # Keep the most recent 10 messages to ensure we don't lose current context
        new_history.extend(self.messages[-10:])
        self.messages = new_history

    async def run(self):
        """Starts the main chat loop."""
        print(f"--- Initializing Environment: {self.env_info} ---")
        print("\n==============================================")
        print("Gemma-Bot is online! (Type 'quit' or 'exit' to leave)")
        print("==============================================\n")

        while True:
            # Since input() is blocking, we run it in a thread to keep the loop alive.
            user_input = await asyncio.to_thread(input, "You: ")
            user_input = user_input.strip()

            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nGemma-Bot: Goodbye! 🤖✨")
                break

            if not user_input:
                continue

            self.messages.append({'role': 'user', 'content': user_input})

            print("\nBot: ", end="", flush=True)

            try:
                full_response = ""
                # Use the async client for streaming
                response = await self.client.chat(model=self.model, messages=self.messages, stream=True)
                async for chunk in response:
                    token = chunk['message']['content']
                    print(token, end="", flush=True)
                    full_response += token

                print()
                self.messages.append({'role': 'assistant', 'content': full_response})

            except Exception as e:
                print(f"\n[Error]: {e}")
                break

            # Check if we need to condense history
            if len(self.messages) > 20:
                await self._condense_history()

if __name__ == "__main__":
    bot = GemmaBot()
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        print("\n\nGemma-Bot: Goodbye! (Interrupted) 🤖✨")
