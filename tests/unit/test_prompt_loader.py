from pathlib import Path

import pytest

from prompt_loader import PromptNotFoundError, PromptResolver


@pytest.fixture
def prompts_dir(tmp_path: Path) -> Path:
    base = tmp_path / "prompts"
    (base / "shared").mkdir(parents=True)
    (base / "clients" / "alpha").mkdir(parents=True)
    (base / "shared" / "bio_generation.md").write_text("shared bio prompt")
    (base / "shared" / "email_classification.md").write_text("shared classify prompt")
    (base / "clients" / "alpha" / "bio_generation.md").write_text("alpha override bio prompt")
    return base


def test_client_override_wins(prompts_dir: Path) -> None:
    r = PromptResolver(prompts_dir)
    out = r.resolve("bio_generation", client_name="alpha")
    assert out.content == "alpha override bio prompt"
    assert out.path.endswith("clients/alpha/bio_generation.md")


def test_falls_back_to_shared_when_no_client_override(prompts_dir: Path) -> None:
    r = PromptResolver(prompts_dir)
    # alpha has no email_classification.md → must fall back to shared
    out = r.resolve("email_classification", client_name="alpha")
    assert out.content == "shared classify prompt"
    assert out.path.endswith("shared/email_classification.md")


def test_no_client_returns_shared(prompts_dir: Path) -> None:
    r = PromptResolver(prompts_dir)
    out = r.resolve("bio_generation")
    assert out.content == "shared bio prompt"


def test_missing_prompt_raises(prompts_dir: Path) -> None:
    r = PromptResolver(prompts_dir)
    with pytest.raises(PromptNotFoundError):
        r.resolve("nonexistent_task")


def test_git_hash_is_none_outside_repo(prompts_dir: Path) -> None:
    r = PromptResolver(prompts_dir)
    out = r.resolve("bio_generation")
    # tmp_path isn't a git repo → hash should be None
    assert out.git_hash is None
