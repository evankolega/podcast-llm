Configuration Guide
===================

Basic Usage
-----------

Here's how to configure the Podcast LLM system by providing a custom configuration file:

.. code-block:: bash 
   python -m podcast_llm.generate "Artificial Intelligence" --config path/to-config.yml

YAML configuration
------------------

The system can be configured using the ``config.yaml`` file:

The configuration file contains several sections:

LLM Configuration
~~~~~~~~~~~~~~~
- ``fast_llm_provider``: Provider for quick LLM operations (options: 'openai', 'google', 'anthropic')
- ``long_context_llm_provider``: Provider for operations requiring longer context (options: 'openai', 'google', 'anthropic', 'moonshot')

Note: The 'moonshot' provider uses Kimi K2 from Moonshot AI, which is optimized for long context tasks.

Text-to-Speech Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~ 
- ``tts_provider``: Text-to-speech service to use (options: 'google', 'elevenlabs')
- ``tts_settings``: Provider-specific settings including:
   - Voice mappings for interviewer and interviewee
   - Model settings
   - Language codes
   - Audio effect profiles

Audio Settings
~~~~~~~~~~~~
- ``output_format``: Format for generated audio (options: 'mp3', 'wav')
- ``temp_audio_dir``: Directory for temporary audio files
- ``output_dir``: Directory for final output files

Checkpointing Settings
~~~~~~~~~~~~~~~~~~~~~~

- ``checkpoint_dir``: Directory for saving generation checkpoints

Rate Limiting
~~~~~~~~~~~
Configure API rate limits per provider:

- ``requests_per_minute``: Maximum requests allowed per minute
- ``max_retries``: Number of retry attempts
- ``base_delay``: Base delay between retries

Content Settings
~~~~~~~~~~~~~
- ``podcast_name``: Name of the podcast
- ``intro``: Template for podcast introduction (variables: {podcast_name}, {topic})
- ``outro``: Template for podcast conclusion
- ``episode_structure``: List defining the structure of episodes

Example Configuration
~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # LLM Configuration
   fast_llm_provider: anthropic
   long_context_llm_provider: google

   # TTS Configuration  
   tts_provider: elevenlabs
   tts_settings:
     elevenlabs:
       voice_mapping:
         Interviewer: Chris
         Interviewee: Charlie
       model: eleven_multilingual_v2

   # Output settings
   output_format: mp3
   output_dir: ./output

   # Content settings
   podcast_name: My AI Podcast
   intro: "Welcome to {podcast_name}. Today we're exploring {topic}."
