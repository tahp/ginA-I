import os
import ollama
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def main():
    """
    Main entry point of the chatbot program.
    Runs an infinite loop to take user input and provide responses via Ollama API.
    """
    model_name = os.getenv("MODEL_NAME", "llama3")
    # history will store the last 10 messages (alternating between user and bot)
    history = []

    while True:
        # Prompt the user for input and clean up whitespace/casing
        user_input = input("You: ").strip()
        
        # Check if the user wants to exit the program
        if user_input.lower() == 'quit':
            print("Chatbot: Goodbye!")
            break
        
        # Add user input to history for context
        history.append({'role': 'user', 'content': user_input})

        try:
            # Send the entire history to Ollama to generate a response using the specific model
            response = ollama.chat(model=model_name, messages=history)
            
            # Extract content from the response
            ai_response = response['message']['content']
            
            print(f"Chatbot: {ai_response}")
            
            # Add the AI's response to history for context
            history.append({'role': 'assistant', 'content': ai_response})

        except Exception as e:
            print(f"An error occurred: {e}")
            # If there is an error, we might want to remove the last user input 
            # so the history doesn't get desynced with successful turns.
            history.pop()
            continue

        # Keep only the last 10 messages in history (5 pairs of user/assistant)
        if len(history) > 10:
            history.pop(0)

if __name__ == "__main__":
    # Execute the main function when the script is run directly
    main()
