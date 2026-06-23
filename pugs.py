import asyncio
import datetime
import platform

import ollama

MODEL = "hf.co/Jiunsong/supergemma4-26b-uncensored-gguf-v2:Q4_K_M"
SYSTEM_PROMPT = (
    "You are a helpful, witty assistant named Gemma-Bot. "
    "Your personality is sophisticated but charming. Use emojis sparingly "
    "but effectively to show emotion. You are clever and slightly playful."
)
HISTORY_THRESHOLD = 20
RETAIN_MESSAGES = 10
CONTEXT_LIMIT = 8192
RETRY_ATTEMPTS = 3
RETRY_BASE_DELAY = 1.0


def _estimate_tokens(text: str) -> int:
    return len(text) // 4 + 1


def _total_tokens(messages: list[dict[str, str]]) -> int:
    overhead = 4
    return sum(
        _estimate_tokens(m.get("content", "") + m.get("role", "")) + overhead
        for m in messages
    )


class GemmaBot:
    def __init__(self, model: str = MODEL, system_prompt: str = SYSTEM_PROMPT) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]
        self.client = ollama.AsyncClient()
        self.env_info = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    async def _check_connection(self) -> str | None:
        try:
            models = await self.client.list()
            available = [m["name"] for m in models.get("models", [])]
            if not any(self.model in name or name in self.model for name in available):
                closest = ", ".join(available[:5]) if available else "none"
                return (
                    f"Model '{self.model}' not found locally.\n"
                    f"Run: ollama pull {self.model}\n"
                    f"Available: {closest}"
                )
            return None
        except Exception as e:
            return f"Cannot connect to ollama: {e}\nIs ollama running? Try: ollama serve"

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
            {
                "role": "system",
                "content": f"Context from previous conversation: {summary}",
            },
            *self.messages[-RETAIN_MESSAGES:],
        ]

    async def _stream_response(self) -> str:
        print("\nBot: ", end="", flush=True)
        last_error: Exception | None = None

        for attempt in range(RETRY_ATTEMPTS):
            if attempt > 0:
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                print(
                    f"\n[Retry] Attempt {attempt + 1}/{RETRY_ATTEMPTS} in {delay}s...",
                    flush=True,
                )
                await asyncio.sleep(delay)

            try:
                full: list[str] = []
                response = await self.client.chat(
                    model=self.model, messages=self.messages, stream=True
                )
                async for chunk in response:
                    token: str = chunk["message"]["content"]
                    print(token, end="", flush=True)
                    full.append(token)
                print()
                return "".join(full)
            except Exception as e:
                last_error = e
                continue

        print(f"\n[Error] {last_error}", flush=True)
        return ""

    async def run(self) -> None:
        print(
            f"--- Gemma-Bot initialized at {self.env_info} on {platform.system()} ---"
        )
        print("=" * 46)
        print("Gemma-Bot is online!  (type 'quit' or 'exit' to leave)")
        print("=" * 46)

        print("\n[System] Checking connection...", end=" ", flush=True)
        err = await self._check_connection()
        if err:
            print(f"\n[Error] {err}")
            return
        print("OK")

        while True:
            user_input = (await asyncio.to_thread(input, "\nYou: ")).strip()

            if user_input.lower() in {"quit", "exit", "bye"}:
                print("\nGemma-Bot: Goodbye!")
                break

            if not user_input:
                continue

            if _total_tokens(self.messages) >= int(CONTEXT_LIMIT * 0.75):
                await self._condense_history()

            self.messages.append({"role": "user", "content": user_input})
            reply = await self._stream_response()

            if reply:
                self.messages.append({"role": "assistant", "content": reply})
            else:
                self.messages.pop()

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
