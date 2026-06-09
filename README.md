# Ollama Chatbot

A simple, modular Python chatbot that interacts with Ollama models.

## Features

- **Streaming Responses**: Real-time token printing for a smoother chat experience.
- **Persistent History**: Chat history is saved to `chat_history.json` and loaded automatically.
- **Environment Configuration**: Easily configure the model name and system prompt via `.env`.
- **Modular Design**: Clean class-based structure for easy extension.
- **Unit Tested**: Includes tests for history management and core logic.

## Prerequisites

- [Ollama](https://ollama.com/) installed and running.
- Python 3.8+

## Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env to set your desired model name
   ```

5. **Pull the model**:
   ```bash
   ollama pull <your-model-name>
   ```

## Usage

### CLI Chatbot
Run the chatbot in your terminal:
```bash
python3 src/chatbot.py
```

### Web Interface
Run the web-based chat interface:
```bash
streamlit run src/web_app.py
```

Type `quit` (in CLI) or close the browser tab to exit.

## Running Tests

```bash
python3 tests/test_chatbot.py
```

## License

MIT
