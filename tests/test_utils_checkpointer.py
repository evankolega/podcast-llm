"""Tests for the checkpointer utility module."""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock

from langchain_core.documents import Document

from podcast_llm.utils.checkpointer import (
    Checkpointer,
    to_snake_case,
    _serialize_value,
    _deserialize_value,
)
from podcast_llm.models import (
    PodcastOutline,
    PodcastSection,
    PodcastSubsection,
    Question,
    Answer,
)


class TestToSnakeCase:
    """Tests for the to_snake_case function."""

    def test_spaces_converted_to_underscores(self):
        assert to_snake_case("hello world") == "hello_world"

    def test_hyphens_converted_to_underscores(self):
        assert to_snake_case("hello-world") == "hello_world"

    def test_converts_to_lowercase(self):
        assert to_snake_case("Hello World") == "hello_world"

    def test_removes_special_characters(self):
        assert to_snake_case("hello@world!") == "helloworld"

    def test_multiple_underscores_collapsed(self):
        assert to_snake_case("hello   world") == "hello_world"

    def test_leading_trailing_underscores_removed(self):
        assert to_snake_case("  hello  ") == "hello"

    def test_complex_string(self):
        assert to_snake_case("AI - Machine Learning!") == "ai_machine_learning"


class TestSerializeDeserialize:
    """Tests for serialization and deserialization functions."""

    def test_serialize_none(self):
        assert _serialize_value(None) is None

    def test_serialize_primitives(self):
        assert _serialize_value("hello") == "hello"
        assert _serialize_value(42) == 42
        assert _serialize_value(3.14) == 3.14
        assert _serialize_value(True) is True

    def test_serialize_simple_dict(self):
        data = {"speaker": "Interviewer", "text": "Hello"}
        result = _serialize_value(data)
        assert result == {"speaker": "Interviewer", "text": "Hello"}

    def test_serialize_langchain_document(self):
        doc = Document(
            page_content="Test content",
            metadata={"title": "Test Title", "source": "test.com"}
        )
        result = _serialize_value(doc)
        
        assert result["__type__"] == "langchain_document"
        assert result["__data__"]["page_content"] == "Test content"
        assert result["__data__"]["metadata"]["title"] == "Test Title"

    def test_serialize_question_model(self):
        question = Question(question="What is AI?")
        result = _serialize_value(question)
        
        assert result["__type__"] == "pydantic_model"
        assert result["model_name"] == "Question"
        assert result["__data__"]["question"] == "What is AI?"

    def test_serialize_answer_model(self):
        answer = Answer(answer="AI is artificial intelligence.")
        result = _serialize_value(answer)
        
        assert result["__type__"] == "pydantic_model"
        assert result["model_name"] == "Answer"
        assert result["__data__"]["answer"] == "AI is artificial intelligence."

    def test_serialize_podcast_outline(self):
        outline = PodcastOutline(
            sections=[
                PodcastSection(
                    title="Introduction",
                    subsections=[
                        PodcastSubsection(title="Overview"),
                        PodcastSubsection(title="Key Points"),
                    ]
                )
            ]
        )
        result = _serialize_value(outline)
        
        assert result["__type__"] == "pydantic_model"
        assert result["model_name"] == "PodcastOutline"
        assert len(result["__data__"]["sections"]) == 1
        assert result["__data__"]["sections"][0]["title"] == "Introduction"

    def test_serialize_list_of_documents(self):
        docs = [
            Document(page_content="Content 1", metadata={"title": "Doc 1"}),
            Document(page_content="Content 2", metadata={"title": "Doc 2"}),
        ]
        result = _serialize_value(docs)
        
        assert result["__type__"] == "typed_list"
        assert len(result["__data__"]) == 2
        assert result["__data__"][0]["__type__"] == "langchain_document"

    def test_serialize_list_of_qa(self):
        qa_list = [
            Question(question="Q1?"),
            Answer(answer="A1."),
            Question(question="Q2?"),
            Answer(answer="A2."),
        ]
        result = _serialize_value(qa_list)
        
        assert result["__type__"] == "typed_list"
        assert len(result["__data__"]) == 4
        assert result["__data__"][0]["model_name"] == "Question"
        assert result["__data__"][1]["model_name"] == "Answer"

    def test_serialize_final_script(self):
        """Test serializing final script format (list of dicts)."""
        final_script = [
            {"speaker": "Interviewer", "text": "Welcome!"},
            {"speaker": "Interviewee", "text": "Thank you."},
        ]
        result = _serialize_value(final_script)
        
        assert result["__type__"] == "typed_list"
        assert len(result["__data__"]) == 2
        assert result["__data__"][0]["speaker"] == "Interviewer"

    def test_deserialize_none(self):
        assert _deserialize_value(None) is None

    def test_deserialize_primitives(self):
        assert _deserialize_value("hello") == "hello"
        assert _deserialize_value(42) == 42
        assert _deserialize_value(3.14) == 3.14
        assert _deserialize_value(True) is True

    def test_deserialize_langchain_document(self):
        data = {
            "__type__": "langchain_document",
            "__data__": {
                "page_content": "Test content",
                "metadata": {"title": "Test Title"}
            }
        }
        result = _deserialize_value(data)
        
        assert isinstance(result, Document)
        assert result.page_content == "Test content"
        assert result.metadata["title"] == "Test Title"

    def test_deserialize_question_model(self):
        data = {
            "__type__": "pydantic_model",
            "model_name": "Question",
            "__data__": {"question": "What is AI?"}
        }
        result = _deserialize_value(data)
        
        assert isinstance(result, Question)
        assert result.question == "What is AI?"

    def test_deserialize_podcast_outline(self):
        data = {
            "__type__": "pydantic_model",
            "model_name": "PodcastOutline",
            "__data__": {
                "sections": [
                    {
                        "title": "Introduction",
                        "subsections": [{"title": "Overview"}]
                    }
                ]
            }
        }
        result = _deserialize_value(data)
        
        assert isinstance(result, PodcastOutline)
        assert len(result.sections) == 1
        assert result.sections[0].title == "Introduction"

    def test_deserialize_list_of_documents(self):
        data = {
            "__type__": "typed_list",
            "__data__": [
                {
                    "__type__": "langchain_document",
                    "__data__": {"page_content": "C1", "metadata": {}}
                },
                {
                    "__type__": "langchain_document",
                    "__data__": {"page_content": "C2", "metadata": {}}
                },
            ]
        }
        result = _deserialize_value(data)
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(doc, Document) for doc in result)

    def test_roundtrip_documents(self):
        """Test that serialization and deserialization are inverse operations."""
        original = [
            Document(page_content="Content 1", metadata={"title": "Doc 1"}),
            Document(page_content="Content 2", metadata={"title": "Doc 2"}),
        ]
        serialized = _serialize_value(original)
        deserialized = _deserialize_value(serialized)
        
        assert len(deserialized) == len(original)
        for orig, deser in zip(original, deserialized):
            assert orig.page_content == deser.page_content
            assert orig.metadata == deser.metadata

    def test_roundtrip_outline(self):
        """Test roundtrip for PodcastOutline."""
        original = PodcastOutline(
            sections=[
                PodcastSection(
                    title="Section 1",
                    subsections=[
                        PodcastSubsection(title="Sub 1.1"),
                        PodcastSubsection(title="Sub 1.2"),
                    ]
                ),
                PodcastSection(
                    title="Section 2",
                    subsections=[
                        PodcastSubsection(title="Sub 2.1"),
                    ]
                ),
            ]
        )
        serialized = _serialize_value(original)
        deserialized = _deserialize_value(serialized)
        
        assert isinstance(deserialized, PodcastOutline)
        assert deserialized.as_str == original.as_str

    def test_roundtrip_qa_list(self):
        """Test roundtrip for list of Question/Answer."""
        original = [
            Question(question="Q1?"),
            Answer(answer="A1."),
            Question(question="Q2?"),
            Answer(answer="A2."),
        ]
        serialized = _serialize_value(original)
        deserialized = _deserialize_value(serialized)
        
        assert len(deserialized) == 4
        assert isinstance(deserialized[0], Question)
        assert isinstance(deserialized[1], Answer)
        assert deserialized[0].question == "Q1?"
        assert deserialized[1].answer == "A1."

    def test_roundtrip_final_script(self):
        """Test roundtrip for final script format."""
        original = [
            {"speaker": "Interviewer", "text": "Welcome to the show!"},
            {"speaker": "Interviewee", "text": "Thanks for having me."},
        ]
        serialized = _serialize_value(original)
        deserialized = _deserialize_value(serialized)
        
        assert deserialized == original

    def test_json_serializable(self):
        """Test that serialized output is valid JSON."""
        outline = PodcastOutline(
            sections=[
                PodcastSection(
                    title="Test Section",
                    subsections=[PodcastSubsection(title="Test Sub")]
                )
            ]
        )
        serialized = _serialize_value(outline)
        
        # Should not raise
        json_str = json.dumps(serialized)
        parsed = json.loads(json_str)
        
        # Should deserialize correctly from JSON
        result = _deserialize_value(parsed)
        assert isinstance(result, PodcastOutline)

    def test_unknown_pydantic_model_raises(self):
        """Test that unregistered Pydantic models raise TypeError."""
        from pydantic import BaseModel
        
        class UnregisteredModel(BaseModel):
            value: str
        
        with pytest.raises(TypeError, match="not registered"):
            _serialize_value(UnregisteredModel(value="test"))

    def test_unknown_type_identifier_raises(self):
        """Test that unknown type identifiers raise ValueError."""
        data = {"__type__": "unknown_type", "__data__": {}}
        
        with pytest.raises(ValueError, match="Unknown type identifier"):
            _deserialize_value(data)


class TestCheckpointer:
    """Tests for the Checkpointer class."""

    def test_init_creates_directory(self, tmp_path: Path):
        checkpoint_dir = tmp_path / "checkpoints"
        checkpointer = Checkpointer(
            checkpoint_key="test",
            checkpoint_dir=str(checkpoint_dir),
            enabled=True
        )
        
        assert checkpoint_dir.exists()
        assert checkpointer.enabled is True

    def test_init_disabled_no_directory(self, tmp_path: Path):
        checkpoint_dir = tmp_path / "checkpoints"
        checkpointer = Checkpointer(
            checkpoint_key="test",
            checkpoint_dir=str(checkpoint_dir),
            enabled=False
        )
        
        assert not checkpoint_dir.exists()
        assert checkpointer.enabled is False

    def test_checkpoint_disabled_calls_function(self, tmp_path: Path):
        checkpointer = Checkpointer(
            checkpoint_key="test",
            checkpoint_dir=str(tmp_path),
            enabled=False
        )
        
        mock_fn = Mock(return_value="result")
        result = checkpointer.checkpoint(mock_fn, ["arg1", "arg2"], "stage")
        
        mock_fn.assert_called_once_with("arg1", "arg2")
        assert result == "result"

    def test_checkpoint_saves_to_file(self, tmp_path: Path):
        checkpointer = Checkpointer(
            checkpoint_key="test",
            checkpoint_dir=str(tmp_path),
            enabled=True
        )
        
        mock_fn = Mock(return_value=[{"speaker": "Test", "text": "Hello"}])
        result = checkpointer.checkpoint(mock_fn, [], "final_script")
        
        checkpoint_file = tmp_path / "test_final_script.json"
        assert checkpoint_file.exists()
        
        # Verify file content
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
        assert data["__type__"] == "typed_list"

    def test_checkpoint_loads_from_file(self, tmp_path: Path):
        checkpointer = Checkpointer(
            checkpoint_key="test",
            checkpoint_dir=str(tmp_path),
            enabled=True
        )
        
        # Create a checkpoint file
        checkpoint_file = tmp_path / "test_my_stage.json"
        data = {
            "__type__": "typed_list",
            "__data__": [
                {"speaker": "Test", "text": "Cached result"}
            ]
        }
        with open(checkpoint_file, 'w') as f:
            json.dump(data, f)
        
        # Should load from file, not call function
        mock_fn = Mock(return_value="should not be called")
        result = checkpointer.checkpoint(mock_fn, [], "my_stage")
        
        mock_fn.assert_not_called()
        assert result == [{"speaker": "Test", "text": "Cached result"}]

    def test_checkpoint_full_workflow(self, tmp_path: Path):
        """Test complete checkpoint workflow with real data types."""
        checkpointer = Checkpointer(
            checkpoint_key="podcast",
            checkpoint_dir=str(tmp_path),
            enabled=True
        )
        
        # First call - saves to checkpoint
        outline = PodcastOutline(
            sections=[
                PodcastSection(
                    title="Intro",
                    subsections=[PodcastSubsection(title="Welcome")]
                )
            ]
        )
        mock_fn = Mock(return_value=outline)
        result1 = checkpointer.checkpoint(mock_fn, [], "outline")
        
        assert mock_fn.call_count == 1
        assert isinstance(result1, PodcastOutline)
        
        # Second call - loads from checkpoint
        mock_fn2 = Mock(return_value="different")
        result2 = checkpointer.checkpoint(mock_fn2, [], "outline")
        
        mock_fn2.assert_not_called()
        assert isinstance(result2, PodcastOutline)
        assert result2.as_str == outline.as_str

    def test_checkpoint_documents(self, tmp_path: Path):
        """Test checkpointing LangChain Documents."""
        checkpointer = Checkpointer(
            checkpoint_key="test",
            checkpoint_dir=str(tmp_path),
            enabled=True
        )
        
        docs = [
            Document(page_content="Article 1", metadata={"title": "Title 1"}),
            Document(page_content="Article 2", metadata={"title": "Title 2"}),
        ]
        
        mock_fn = Mock(return_value=docs)
        result1 = checkpointer.checkpoint(mock_fn, [], "background_info")
        
        # Verify saved correctly
        mock_fn2 = Mock()
        result2 = checkpointer.checkpoint(mock_fn2, [], "background_info")
        
        mock_fn2.assert_not_called()
        assert len(result2) == 2
        assert all(isinstance(d, Document) for d in result2)
        assert result2[0].page_content == "Article 1"

    def test_checkpoint_qa_list(self, tmp_path: Path):
        """Test checkpointing Question/Answer list."""
        checkpointer = Checkpointer(
            checkpoint_key="test",
            checkpoint_dir=str(tmp_path),
            enabled=True
        )
        
        qa = [
            Question(question="What is this?"),
            Answer(answer="This is a test."),
        ]
        
        mock_fn = Mock(return_value=qa)
        checkpointer.checkpoint(mock_fn, [], "draft_script")
        
        # Load and verify
        mock_fn2 = Mock()
        result = checkpointer.checkpoint(mock_fn2, [], "draft_script")
        
        assert len(result) == 2
        assert isinstance(result[0], Question)
        assert isinstance(result[1], Answer)
        assert result[0].question == "What is this?"
