# Technical Context: Podcast-LLM

## Technology Stack
- **Language**: Python 3.x
- **Package Management**: Poetry (pyproject.toml)
- **Testing**: pytest
- **LLM Framework**: LangChain
- **Data Validation**: Pydantic

## LLM Providers Supported
- OpenAI (GPT-4o)
- Google (Gemini-1.5-pro)
- Anthropic (Claude-3.5-sonnet)

## Key Dependencies
- langchain, langchain-openai, langchain-google, langchain-anthropic
- pydantic
- InMemoryVectorStore for RAG
- RecursiveCharacterTextSplitter for document chunking
- Tavily for web search
- Wikipedia API for research

## Commands
- Run tests: `./run_tests.sh` or `pytest`
- Install: `pip install podcast-llm`
- CLI: `podcast-llm "Topic Name"`
- GUI: `podcast-llm-gui`

## Environment Variables Required
- OPENAI_API_KEY
- GOOGLE_API_KEY
- ELEVENLABS_API_KEY
- TAVILY_API_KEY
- ANTHROPIC_API_KEY

## Configuration
- Main config: `config/config.yaml`
- Prompts pulled from LangChain Hub (evandempsey/*)
