def main():
    while True:
        user_input = input("You: ").strip().lower()
        
        if user_input == 'quit':
            print("Chatbot: Goodbye!")
            break
        
        print(f"Chatbot: You said '{user_input}'")

if __name__ == "__main__":
    main()
