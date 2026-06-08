def main():
    """
    Main entry point of the chatbot program.
    Runs an infinite loop to take user input and provide responses.
    """
    while True:
        # Prompt the user for input and clean up whitespace/casing
        user_input = input("You: ").strip().lower()
        
        # Check if the user wants to exit the program
        if user_input == 'quit':
            print("Chatbot: Goodbye!")
            break
        
        # Echo the user's input back to them
        print(f"Chatbot: You said '{user_input}'")

if __name__ == "__main__":
    # Execute the main function when the script is run directly
    main()
