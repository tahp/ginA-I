import ollama
import datetime
import platform
import sys

# --- CONFIGURATION ---
# The model you are using
MODEL = 'hf.co/Jiunsced/supergemma4-26b-uncensored-gguf-v2:Q4_K_M'
MAX_MESSAGES = 30

# The core personality of your bot
SYSTEM_PROMPT = (
    "You are a helpful, witty assistant named Gemma-Bot. "
    "Your personality is sophisticated but charming. Use emojis sparingly "
    "but effectively to show emotion. You are clever and slightly playful."
)

def get_environment_context():
    """Generates a string containing the current time and OS for the bot."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os_name = platform.system()
    return f"(Current Time: {now}, Operating System: {os_name})"

def summarize_old_messages(messages):
    """Takes the oldest messages and turns them into one summary."""
    # We take everything from index  than the last 10, but excluding the system prompt (index 0)
    # This ensures we are summarizing actual conversation history.
    to_summarize = messages[1:-10]
    
    if not to_summarize:
        return "The conversation is just beginning."

    context_str = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in to_summarize])

    prompt = f"Summarize the key context from these conversation logs into one concise sentence: {context_str}"

    # Call ollama to get a summary
    summary_response = ollama.chat(model=MODEL, messages=[{'role': 'user', 'content': prompt}])
    return summary_response['message']['content']

def chat():
    # 1. Initialize messages with the system prompt
    # We keep the 'system' message at index 0 so it is never lost.
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]

    # Add an initial hidden "environment" message to give the bot awareness from the start
    env_info = get_environment_context()
    print(f"--- Initializing Environment: {env_info} ---")

    print("\n==============================================")
    print("Gemma-Bot is online! (Type 'quit' or 'exit' to leave)")
    print("==============================================\n")

    while True:
        # Get user input and strip whitespace
        user_input = input("You: ").strip()

        # Exit conditions
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("\nGemma-Bot: Goodbye! 🤖✨")
            break

        if not user_input:
            continue

        # Add the user's message to our permanent history list
        messages.append({'role': 'user', 'content': user_input})

        print("\nBot: ", end="", flush=True)

        try:
            # --- LEVEL 3: STREAMING ---
            # We use stream=True to get the "typing" effect.
            full_response = ""
            stream = ollama.chat(model=MODEL, messages=messages, stream=True)

            for chunk in stream:
                token = chunk['message']['content']
                print(token, end="", flush=True) # Print the bit of text as it arrives
                full_response += token

            print() # Print a newline when the response is done

            # Add the complete assistant response to history for future context
            messages.append({'role': 'assistant', 'content': full_response})

        except Exception as e:
            print(f"\n[Error]: {e}")
            break

        # --- NEW LOGIC CONCEPT: THE SUMMARY LOOP ---
        # If the message list gets too long, we "condense" it!
        if len(messages) > 15:
            print("\n[System]: Condensing memories...")
            summary = summarize_old_messages(messages)

            # The new structure becomes: [System] + [The Summary as a system/user context] + [Last 10 messages]
            # We keep the system prompt at index 0.
            # We then add the summary as a 'system' message so it acts as permanent context.
            new_history = [messages[0], {'role': 'system', 'content': f"Context from previous conversation: {summary}"}]
            
            # Add the most recent messages (the last 10) to the new history
            # We use [-10:] but ensure we don't duplicate if the list is short.
            new_history.extend(messages[-10:])
            messages = new_history

if __name__ == "__main__":
    chat()
