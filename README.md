# Brainstormer Agent 🧠💡

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Brainstormer is an AI agent that actively listens to your conversations (mic + system audio) and provides real-time insights. It transcribes the dialogue and uses Large Language Models (LLMs) via OpenAI and OpenRouter to suggest ideas, questions, or alternative perspectives, logging them for later review.

## ✨ Features

*   **Dual Audio Capture**: Records microphone and system audio simultaneously using `sounddevice`.
*   **Real-time Transcription**: Uses OpenAI's API for fast and accurate transcription (model configurable).
*   **Intelligent Idea Generation**: Sends transcript segments to a configurable LLM via OpenRouter using customizable prompts (`config.yaml`) to generate relevant insights.
*   **Asynchronous**: Built with `asyncio` for efficient, non-blocking operation.
*   **Logging**: Saves full transcripts (`.txt`) and generated ideas (`.jsonl`) with timestamps.
*   **Configurable**: Customize models, API keys, timings, prompts, and log paths via `config.yaml`.
*   **Audio Device Management**: Auto-detects devices; includes a microphone test function.
*   **Graceful Shutdown**: Handles `Ctrl+C` cleanly, processing remaining audio.

## ⚙️ Requirements

*   Python 3.8+
*   `pip install -r requirements.txt`
*   **OpenAI API Key**: For transcription.
*   **OpenRouter API Key**: For idea generation (accesses various LLMs).

## 🚀 Installation

1.  **Clone:** `git clone https://github.com/yourusername/brainstormer.git && cd brainstormer` (Replace URL)
2.  **Install:** `pip install -r requirements.txt`
3.  **API Keys:** Set `OPENAI_API_KEY` and `OPENROUTER_API_KEY` environment variables (e.g., in `.zshrc`, `.bashrc`, or your session).
    ```bash
    export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
    export OPENROUTER_API_KEY="YOUR_OPENROUTER_API_KEY"
    ```

## ▶️ Usage

1.  **Run:** `python main.py`
    *   The agent starts capturing, transcribing, and generating ideas based on `config.yaml`.
    *   Logs are saved to configured directories (`transcripts/`, `ideas_log.jsonl`).
2.  **Stop:** Press `Ctrl+C`.
3.  **Configure:** Modify `config.yaml` to change models, prompts, intervals, etc. A default config is created if missing.
    *   Optional: Use a specific config file: `python main.py --config <path>`

## 🔧 Configuration (`config.yaml`)

Tailor Brainstormer's behavior:

```yaml
transcription:
  # Model for OpenAI transcription (e.g., whisper-1)
  model: "YOUR_TRANSCRIPTION_MODEL" # Replace with a valid OpenAI transcription model

main:
  # Interval (seconds) for idea generation
  idea_interval: 30
  # Interval (seconds) for summarization check (currently inactive logic)
  summarize_interval: 120

logging:
  # Directory for full transcripts
  transcript_log_dir: "transcripts"

llm:
  # LLM model via OpenRouter for ideas (e.g., anthropic/claude-3-haiku, google/gemini-flash)
  model: "YOUR_LLM_MODEL" # Replace with a valid OpenRouter model identifier
  api_url: https://openrouter.ai/api/v1/chat/completions
  max_tokens: 1000 # Max tokens for LLM idea response
  temperature: 0.8 # Sampling temperature
  ideas_log_path: ideas_log.jsonl # Log file for ideas

  # --- Prompts ---
  system_prompt: |
    You are a concise and insightful assistant analyzing conversation transcripts.
    Your goal is to identify truly novel ideas, missed connections, critical questions,
    alternative perspectives, or overlooked challenges. Focus on adding unique value.
    If the conversation segment doesn't spark any genuinely useful or non-obvious insight,
    respond with only the exact text: NO_IDEAS
  user_prompt_template: |
    Analyze the following conversation transcript. Focus primarily on the **most recent exchanges** to generate 1-2 highly relevant and insightful contributions (ideas, connections, questions, perspectives, challenges) that participants might not have considered.

    **Only provide a response if you have a genuinely useful and non-obvious insight.** Otherwise, respond with only the exact text: NO_IDEAS

    Ensure your output is a simple list, one item per line, without numbers, bullets, or explanations.

    TRANSCRIPT:
    {transcript}

  # --- Summarization Settings (Currently inactive) ---
  summarization_system_prompt: "You are a helpful assistant that summarizes conversations accurately..." # Truncated for brevity
  summarization_user_prompt_template: "Summarize the following conversation transcript concisely...\n\n{transcript}" # Truncated
  summarization_max_tokens: 5000
  summarization_temperature: 0.7
```

## 🛠️ How It Works (Simplified)

1.  **Audio Capture (`AudioManager`)**: Records mic/speaker audio to a queue.
2.  **Main Loop (`main.py`)**: Periodically processes audio from the queue.
3.  **Transcription (`TranscriptionService`)**: Sends audio chunks to OpenAI Whisper API.
4.  **Logging**: Appends transcript to `.txt` file.
5.  **Idea Generation (`IdeaGenerator`)**: Sends transcript (potentially summarized - currently inactive) to an LLM via OpenRouter.
6.  **Idea Logging**: If ideas are generated (not "NO_IDEAS"), logs them to `.jsonl`.
7.  **Shutdown**: `Ctrl+C` triggers cleanup and final processing.

## 📄 License

MIT License. 