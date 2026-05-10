"""Tests for the TTS engine + route validation pieces.

The synthesis happy-path needs voice files and ffmpeg, so we don't run it
in unit tests — that's covered by the make download-voice + smoke test
flow. These tests cover the validation surface that's easy to break.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from routes.tts import _OUTPUT_FILENAME
from tts.piper_engine import VoiceNotFound, synthesize_to_mp3

# --- output filename regex --------------------------------------------------


def test_uuid_filename_accepted() -> None:
    assert _OUTPUT_FILENAME.match(f"{uuid4()}.mp3")


def test_filename_rejects_path_traversal() -> None:
    assert not _OUTPUT_FILENAME.match("../etc/passwd")
    assert not _OUTPUT_FILENAME.match("../../output/file.mp3")
    assert not _OUTPUT_FILENAME.match("/etc/shadow")


def test_filename_rejects_non_uuid() -> None:
    assert not _OUTPUT_FILENAME.match("audiobook.mp3")
    assert not _OUTPUT_FILENAME.match("chapter1.mp3")


def test_filename_rejects_non_mp3() -> None:
    uid = uuid4()
    assert not _OUTPUT_FILENAME.match(f"{uid}.wav")
    assert not _OUTPUT_FILENAME.match(f"{uid}.exe")
    assert not _OUTPUT_FILENAME.match(f"{uid}")


def test_filename_rejects_nested_paths() -> None:
    uid = uuid4()
    assert not _OUTPUT_FILENAME.match(f"sub/{uid}.mp3")
    assert not _OUTPUT_FILENAME.match(f"./{uid}.mp3")


def test_filename_rejects_uppercase_uuid() -> None:
    """Python uuid4 always emits lowercase; reject anything uppercase to keep
    the input space tight."""
    upper = str(uuid4()).upper()
    assert not _OUTPUT_FILENAME.match(f"{upper}.mp3")


# --- engine: missing voice file -------------------------------------------


def test_missing_voice_raises(tmp_path: Path) -> None:
    """Engine should fail clearly when the voice files don't exist."""
    with pytest.raises(VoiceNotFound, match="not found"):
        synthesize_to_mp3(
            text="hello",
            voice_name="nonexistent-voice",
            voices_dir=tmp_path,
            output_path=tmp_path / "out.mp3",
        )


def test_missing_config_file_raises(tmp_path: Path) -> None:
    """Onnx without the json config — also a VoiceNotFound."""
    (tmp_path / "fake.onnx").write_bytes(b"not a real model")
    # No fake.onnx.json
    with pytest.raises(VoiceNotFound):
        synthesize_to_mp3(
            text="hello",
            voice_name="fake",
            voices_dir=tmp_path,
            output_path=tmp_path / "out.mp3",
        )
