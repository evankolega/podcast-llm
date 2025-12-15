# System Patterns: Podcast-LLM

## Architecture Overview

### Core Pipeline
```
research_background_info() → outline_episode() → research_discussion_topics() → write_draft_script() → write_final_script() → TTS
```

### Key Modules
- `generate.py`: Main orchestration/entry point
- `writer.py`: Core conversation generation logic
- `models.py`: Pydantic models (Question, Answer, Script, etc.)
- `outline.py`: Episode outline generation
- `research.py`: Background research gathering
- `text_to_speech.py`: TTS synthesis
- `utils/llm.py`: LLM client wrapper
- `utils/rate_limits.py`: Retry logic with exponential backoff

## Conversation Generation Pattern (Current)

### The discuss() Function Loop
```python
for section in outline.sections:
    for subsection in section.subsections:
        for _ in range(qa_rounds):  # Default: 2 rounds
            draft_discussion.append(ask_question(...))   # 1 LLM call
            draft_discussion.append(answer_question(...)) # 1 LLM call
```

### Key Characteristics
1. **Single-line generation**: Each LLM call produces exactly ONE question or ONE answer
2. **Growing context**: Full conversation history sent with each call
3. **Repeated static context**: Full outline, topic, background info sent every call
4. **Separate prompts**: Interviewer and interviewee use different LangChain Hub prompts

### LLM Call Flow
- `ask_question()`: Generates ONE question per call
- `answer_question()`: Generates ONE answer per call
- `rewrite_script_section()`: Processes batch of 4 exchanges (more efficient)

## Data Models

### Conversation Models
- `Question`: Single question with `question` field
- `Answer`: Single answer with `answer` field
- `ScriptLine`: Speaker + text
- `Script`: List of ScriptLine objects

### Outline Models
- `PodcastOutline`: List of PodcastSections
- `PodcastSection`: Title + list of PodcastSubsections
- `PodcastSubsection`: Title only

## Known Inefficiency: Context Duplication

Each call to `ask_question()` and `answer_question()` sends:
- Full topic
- Full outline (`outline.as_str`)
- Full section/subsection context
- Full background info (all Wikipedia docs)
- **Growing conversation history** (larger with each call)
