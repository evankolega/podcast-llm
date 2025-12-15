# Project Brief: Podcast-LLM

## Overview
Podcast-LLM is an AI-powered system that automatically generates engaging podcast conversations using Large Language Models (LLMs) and text-to-speech technology.

## Core Features
- **Two operational modes**:
  - Research mode: Automated research and content gathering using Wikipedia and Tavily search
  - Context mode: Generate podcasts from provided source materials (URLs and files)
- Dynamic podcast outline generation
- Natural conversational script writing with multiple Q&A rounds
- High-quality text-to-speech synthesis (Google Cloud or ElevenLabs)
- Checkpoint system for saving progress and resuming generation
- Configurable voices and audio settings
- Gradio web UI

## Key Technical Stack
- Python with Pydantic for data models
- LangChain for LLM orchestration
- Multiple LLM provider support (OpenAI, Google, Anthropic)
- Vector stores for RAG-based answer generation
- Text-to-speech integration

## Project Goals
1. Generate natural-sounding podcast conversations
2. Automate research and content gathering
3. Produce high-quality audio output
4. Support both fully automated and user-guided content generation

## License
Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
