"""Tests for the GUI module path validation."""

import os
import pytest
from pathlib import Path
from unittest.mock import patch

from podcast_llm.gui import validate_output_path, PathTraversalError


class TestValidateOutputPath:
    """Tests for the validate_output_path function."""

    def test_valid_simple_filename(self, tmp_path: Path) -> None:
        """Test that a simple filename is accepted."""
        result = validate_output_path('podcast.mp3', allowed_dir=tmp_path)
        assert result == tmp_path / 'podcast.mp3'

    def test_valid_filename_with_subdirectory(self, tmp_path: Path) -> None:
        """Test that a filename with subdirectory is accepted."""
        result = validate_output_path('episodes/podcast.mp3', allowed_dir=tmp_path)
        assert result == tmp_path / 'episodes' / 'podcast.mp3'

    def test_creates_output_directory(self, tmp_path: Path) -> None:
        """Test that the output directory is created if it doesn't exist."""
        new_dir = tmp_path / 'new_output'
        assert not new_dir.exists()
        validate_output_path('test.mp3', allowed_dir=new_dir)
        assert new_dir.exists()

    def test_rejects_empty_path(self, tmp_path: Path) -> None:
        """Test that empty paths are rejected."""
        with pytest.raises(PathTraversalError, match="cannot be empty"):
            validate_output_path('', allowed_dir=tmp_path)

    def test_rejects_whitespace_only_path(self, tmp_path: Path) -> None:
        """Test that whitespace-only paths are rejected."""
        with pytest.raises(PathTraversalError, match="cannot be empty"):
            validate_output_path('   ', allowed_dir=tmp_path)

    def test_rejects_absolute_path_unix(self, tmp_path: Path) -> None:
        """Test that absolute Unix paths are rejected."""
        with pytest.raises(PathTraversalError, match="Absolute paths are not allowed"):
            validate_output_path('/etc/passwd', allowed_dir=tmp_path)

    def test_rejects_absolute_path_with_traversal(self, tmp_path: Path) -> None:
        """Test that absolute paths with traversal are rejected."""
        with pytest.raises(PathTraversalError, match="Absolute paths are not allowed"):
            validate_output_path('/tmp/../etc/passwd', allowed_dir=tmp_path)

    def test_rejects_path_traversal_simple(self, tmp_path: Path) -> None:
        """Test that simple path traversal is rejected."""
        with pytest.raises(PathTraversalError, match="Path traversal sequences"):
            validate_output_path('../secret.txt', allowed_dir=tmp_path)

    def test_rejects_path_traversal_nested(self, tmp_path: Path) -> None:
        """Test that nested path traversal is rejected."""
        with pytest.raises(PathTraversalError, match="Path traversal sequences"):
            validate_output_path('subdir/../../secret.txt', allowed_dir=tmp_path)

    def test_rejects_path_traversal_multiple(self, tmp_path: Path) -> None:
        """Test that multiple path traversal sequences are rejected."""
        with pytest.raises(PathTraversalError, match="Path traversal sequences"):
            validate_output_path('../../../etc/passwd', allowed_dir=tmp_path)

    def test_rejects_path_traversal_at_end(self, tmp_path: Path) -> None:
        """Test that path traversal at end is rejected."""
        with pytest.raises(PathTraversalError, match="Path traversal sequences"):
            validate_output_path('subdir/..', allowed_dir=tmp_path)

    def test_strips_whitespace(self, tmp_path: Path) -> None:
        """Test that leading/trailing whitespace is stripped."""
        result = validate_output_path('  podcast.mp3  ', allowed_dir=tmp_path)
        assert result == tmp_path / 'podcast.mp3'

    def test_handles_valid_dots_in_filename(self, tmp_path: Path) -> None:
        """Test that valid dots in filenames are accepted."""
        result = validate_output_path('my.podcast.episode.mp3', allowed_dir=tmp_path)
        assert result == tmp_path / 'my.podcast.episode.mp3'

    def test_handles_hidden_files(self, tmp_path: Path) -> None:
        """Test that hidden files (starting with .) are accepted."""
        result = validate_output_path('.hidden_podcast.mp3', allowed_dir=tmp_path)
        assert result == tmp_path / '.hidden_podcast.mp3'

    def test_result_is_within_allowed_dir(self, tmp_path: Path) -> None:
        """Test that the result path is always within the allowed directory."""
        result = validate_output_path('test.mp3', allowed_dir=tmp_path)
        # Verify the result is within tmp_path
        result.relative_to(tmp_path)  # This will raise if not within

    def test_rejects_symlink_escape(self, tmp_path: Path) -> None:
        """Test that symlinks that escape the allowed directory are rejected."""
        # Create a symlink that points outside the allowed directory
        escape_dir = tmp_path / 'escape'
        escape_dir.mkdir()
        symlink = tmp_path / 'allowed' / 'link'
        (tmp_path / 'allowed').mkdir()
        
        # Create a symlink pointing to parent directory
        try:
            symlink.symlink_to(tmp_path.parent)
        except OSError:
            pytest.skip("Cannot create symlinks on this system")
        
        # Attempting to use the symlink should be rejected
        with pytest.raises(PathTraversalError, match="Invalid output path"):
            validate_output_path('link/outside.txt', allowed_dir=tmp_path / 'allowed')

    def test_different_file_extensions(self, tmp_path: Path) -> None:
        """Test various valid file extensions."""
        extensions = ['mp3', 'wav', 'txt', 'md', 'json']
        for ext in extensions:
            result = validate_output_path(f'output.{ext}', allowed_dir=tmp_path)
            assert result == tmp_path / f'output.{ext}'


class TestPathTraversalError:
    """Tests for the PathTraversalError exception."""

    def test_inherits_from_value_error(self) -> None:
        """Test that PathTraversalError inherits from ValueError."""
        assert issubclass(PathTraversalError, ValueError)

    def test_can_be_raised_with_message(self) -> None:
        """Test that PathTraversalError can be raised with a custom message."""
        with pytest.raises(PathTraversalError, match="custom message"):
            raise PathTraversalError("custom message")
