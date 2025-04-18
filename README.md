# Brainstormer Agent 🧠💡

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Brainstormer is an AI-powered agent designed to listen actively to your conversations (capturing both microphone input and system audio output) and provide real-time, insightful contributions. It transcribes the ongoing dialogue and uses Large Language Models (LLMs) to generate relevant ideas, questions, alternative perspectives, or identify missed connections, logging them for review.

## ✨ Features

*   **Dual Audio Capture**: Captures audio from both your microphone and system speakers simultaneously using `sounddevice`.
*   **Real-time Transcription**: Uses OpenAI's Whisper API (model configurable, e.g., `gpt-4o-mini-transcribe`) to transcribe captured audio chunks.
*   **Intelligent Idea Generation**: Periodically sends recent transcript segments to a configurable LLM via OpenRouter (e.g., `anthropic/claude-3-haiku`, `google/gemini-2.5-flash-preview`) using customizable prompts (`config.yaml`) to generate contextually relevant insights.
*   **Asynchronous Architecture**: Built with Python's `asyncio` for efficient, non-blocking I/O operations (audio handling, API calls, file logging).
*   **Transcript Logging**: Saves the complete conversation transcript to timestamped `.txt` files in a configurable directory (default: `./transcripts/`).
*   **Idea Logging**: Records generated ideas, timestamps, and corresponding transcript snippets to a structured JSONL file (configurable, e.g., `ideas_log.jsonl`).
*   **Configurable**: Easily customize models, API endpoints, timing intervals, prompts, and logging paths via a central `config.yaml` file. A default configuration is generated if none exists.
*   **Audio Device Management**: Automatically detects and lists available audio devices. Uses system defaults but can be adapted (code currently supports setting specific devices). Includes a microphone test function.
*   **Graceful Shutdown**: Handles `KeyboardInterrupt` (Ctrl+C) to stop processes cleanly, ensuring final audio segments are transcribed and analyzed.

## ⚙️ Requirements

*   Python 3.8+
*   Dependencies listed in `requirements.txt` (`pip install -r requirements.txt`)
*   **OpenAI API Key**: For audio transcription.
*   **OpenRouter API Key**: For LLM-based idea generation and summarization. (OpenRouter provides access to various LLMs).

## 🚀 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/brainstormer.git # Replace with your repo URL
    cd brainstormer
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up API Keys:**
    Export your API keys as environment variables. You can add these lines to your `.bashrc`, `.zshrc`, or run them in your terminal session before starting the agent.
    ```bash
    export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
    export OPENROUTER_API_KEY="YOUR_OPENROUTER_API_KEY"
    ```

## ▶️ Usage

1.  **Run the main script:**
    ```bash
    python main.py
    ```
    The agent will start capturing audio, transcribing, and generating ideas based on the intervals defined in `config.yaml`. Transcripts and ideas will be saved to the configured log files.

2.  **Stop the agent:**
    Press `Ctrl+C` in the terminal where the script is running. The agent will perform a graceful shutdown, processing any remaining audio.

3.  **Configuration:**
    Modify the `config.yaml` file to change models, prompts, intervals, etc. If the file doesn't exist, it will be created with default values on the first run.

    *   `--config <path>`: Use a specific configuration file path (optional).

## 🔧 Configuration (`config.yaml`)

The `config.yaml` file allows you to tailor Brainstormer's behavior.

```yaml
transcription:
  # Model used for OpenAI transcription
  model: gpt-4o-mini-transcribe

main:
  # Interval (seconds) between sending transcript chunks for idea generation
  idea_interval: 30
  # Interval (seconds) for checking if transcript needs summarization (currently inactive logic)
  summarize_interval: 120

logging:
  # Directory to save full conversation transcripts
  transcript_log_dir: "transcripts"

llm:
  # LLM model accessed via OpenRouter for idea generation
  model: google/gemini-2.5-flash-preview
  # OpenRouter API endpoint
  api_url: https://openrouter.ai/api/v1/chat/completions
  # Max tokens for the LLM response (ideas)
  max_tokens: 1000
  # Sampling temperature for LLM response
  temperature: 0.8
  # File path to save generated ideas (JSONL format recommended)
  ideas_log_path: ideas_log.jsonl # Changed default to jsonl

  # --- Prompts ---
  # System prompt guiding the LLM's role for idea generation
  system_prompt: |
    You are a concise and insightful assistant analyzing conversation transcripts.
    Your goal is to identify truly novel ideas, missed connections, critical questions,
    alternative perspectives, or overlooked challenges. Focus on adding unique value.
    If the conversation segment doesn't spark any genuinely useful or non-obvious insight,
    respond with only the exact text: NO_IDEAS
  # User prompt template for idea generation (includes the transcript)
  user_prompt_template: |
    Analyze the following conversation transcript. Focus primarily on the **most recent exchanges** to generate 1-2 highly relevant and insightful contributions (ideas, connections, questions, perspectives, challenges) that participants might not have considered.

    **Only provide a response if you have a genuinely useful and non-obvious insight.** Otherwise, respond with only the exact text: NO_IDEAS

    Ensure your output is a simple list, one item per line, without numbers, bullets, or explanations.

    TRANSCRIPT:
    {transcript}

  # --- Summarization Settings (Currently used by TranscriptionService, loop logic in main.py disabled) ---
  summarization_system_prompt: "You are a helpful assistant that summarizes conversations accurately while preserving the key points and context."
  summarization_user_prompt_template: "Summarize the following conversation transcript concisely while preserving all key information, topics, and relevant context:\n\n{transcript}"
  summarization_max_tokens: 5000
  summarization_temperature: 0.7
```

## 🛠️ How It Works

1.  **Audio Capture (`AudioManager`)**: A background thread uses `sounddevice` to capture audio from the default input (mic) and output (speakers) devices into a shared queue.
2.  **Main Loop (`main.py`)**: An asynchronous loop runs periodically (defined by `idea_interval`).
3.  **Get Audio Chunk**: Retrieves all accumulated audio data from the queue.
4.  **Transcription (`TranscriptionService`)**: If audio data exists, it's saved to a temporary WAV file and sent asynchronously to the OpenAI Whisper API.
5.  **Transcript Logging**: The returned transcript text is appended to the current timestamped log file in `transcript_log_dir`.
6.  **Idea Generation (`IdeaGenerator`)**: The *entire* current transcript log is read, and (potentially with summarization - though currently inactive) sent asynchronously to the configured LLM via the OpenRouter API, using the defined prompts.
7.  **Idea Logging**: If the LLM returns ideas (not "NO_IDEAS"), they are parsed and appended as a JSON object (containing timestamp, ideas list, and transcript snippet) to the `ideas_log_path` file.
8.  **Summarization (`summarization_loop` in `main.py`)**: A separate async loop periodically checks the transcript length. *Note: The current implementation checks length but the summarization call itself is commented out.* If enabled, it would call `TranscriptionService.summarize`.
9.  **Shutdown**: On `Ctrl+C`, loops are cancelled, remaining audio is processed for a final transcription/idea generation cycle, and network clients are closed.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (if one exists, otherwise state MIT). 