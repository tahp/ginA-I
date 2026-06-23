import asyncio
import datetime
import platform
from typing import Never

import ollama

MODEL = "hf.co/Jiunscled/supergemma4-26b-uncensored-gguf-v2:Q4_K_M"
SYSTEM_PROMPT = (
    "You are a helpful, witty assistant named Gemma-Bot. "
    "Your personality is sophisticated but charming. Use emojis sparingly "
    "but effectively to show emotion. You are clever and slightly playful."
)
HISTORY_THRESHOLD = 20
RETAIN_MESSAGES = 10


class GemmaBot:
    def __init__(self, model: str = MODEL, system_prompt: str = SYSTEM_PROMPT) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        self.client = ollama.AsyncClient()
        self.env_info = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    async def _summarize(self) -> str:
        to_summarize = self.messages[1:-RETAIN_MESSAGES]
        if not to_summarize:
            return "The conversation is just beginning."

        context_str = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in to_summarize
        )
        prompt = (
            "Summarize the key context from these conversation logs "
            f"into one concise sentence:\n{context_str}"
        )

        try:
            response = await self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response["message"]["content"]
        except Exception:
            return "The conversation is still fresh."

    async def _condense_history(self) -> None:
        summary = await self._summarize()
        print(f"\n[System] Condensing memories... ({summary})")

        self.messages = [
            self.messages[0],
            {"role": "system", "content": f"Context from previous conversation: {summary}"},
            *self.messages[-RETAIN_MESSAGES:],
        ]

    async def _stream_response(self) -> str:
        print("\nBot: ", end="", flush=True)
        full: list[str] = []

        try:
            response = await self.client.chat(
                model=self.model, messages=self.messages, stream=True
            )
            async for chunk in response:
                token: str = chunk["message"]["content"]
                print(token, end="", flush=True)
                full.append(token)
        except Exception as e:
            print(f"\n[Error] {e}", flush=True)

        print()
        return "".join(full)

    async def run(self) -> Never:
        print(f"--- Gemma-Bot initialized at {self.env_info} on {platform.system()} ---")
        print("=" * 46)
        print("Gemma-Bot is online!  (type 'quit' or 'exit' to leave)")
        print("=" * 46)

        while True:
            user_input = (await asyncio.to_thread(input, "You: ")).strip()

            if user_input.lower() in {"quit", "exit", "bye"}:
                print("\nGemma-Bot: Goodbye!")
                break

            if not user_input:
                continue

            self.messages.append({"role": "user", "content": user_input})
            reply = await self._stream_response()

            if reply:
                self.messages.append({"role": "assistant", "content": reply})

            if len(self.messages) > HISTORY_THRESHOLD:
                await self._condense_history()


def main() -> None:
    bot = GemmaBot()
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        print("\n\nGemma-Bot: Goodbye! (Interrupted)")


if __name__ == "__main__":
    main()