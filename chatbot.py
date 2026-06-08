def main():
    """
    Main entrys point of the chatbot program.
    Runs an infinite loop to take user input and provide responses.
    """
    # history will store the last 10 messages (alternating between user and bot)
    history = []

    while True:
        # Prompt the user for input and clean up whitespace/casing
        user_input = input("You: ").strip().lower()
        
        # Check if the user wants to exit the program
        if user_input == 'quit':
            print("Chatbot: Goodbye!")
            break
        
        # Add user input to history
        history.append(f"User: {user_input}")
        
        # Keep only the last 10 messages in history
        if len(history) > 10:
            history.pop(0)

        # Echo the user's input back to them
        print(f"Chatbot: You said '{user_input}'")
        print(f"(Memory size: {len(history)}/10)")

if __name__ == "__main__":
    # Execute the main function when the script is run directly
    main()
