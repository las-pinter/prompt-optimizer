"""
tests/test_core.py
Basic tests for analyzer, rules engine, and transformer.
"""

import pytest
from prompt_optimizer.analyzer import PromptAnalyzer
from prompt_optimizer.rules_engine import RulesEngine
from prompt_optimizer.transformer import PromptTransformer


LLAMA2_PROMPT = "<s>[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\nHello [/INST]"
PLAIN_PROMPT = "You are a helpful assistant. Think step by step. Answer the user's question."


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

def test_analyzer_detects_llama2_tokens():
    features = PromptAnalyzer().analyze(LLAMA2_PROMPT)
    assert features.has_llama2_tokens is True


def test_analyzer_detects_cot():
    features = PromptAnalyzer().analyze(PLAIN_PROMPT)
    assert features.has_cot_instruction is True


def test_analyzer_no_length_instruction():
    features = PromptAnalyzer().analyze(PLAIN_PROMPT)
    assert features.has_length_instruction is False


def test_analyzer_word_count():
    features = PromptAnalyzer().analyze("hello world")
    assert features.word_count == 2


# ---------------------------------------------------------------------------
# Rules engine
# ---------------------------------------------------------------------------

def test_engine_lists_models():
    engine = RulesEngine()
    models = engine.list_models()
    assert "claude" in models
    assert "llama" in models
    assert "gpt" in models


def test_engine_unknown_model_raises():
    engine = RulesEngine()
    with pytest.raises(ValueError, match="Unknown model"):
        engine.get_profile("totally_fake_model_xyz")


def test_engine_profile_has_transformations():
    engine = RulesEngine()
    profile = engine.get_profile("llama")
    assert len(profile.transformations) > 0


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------

def test_transformer_strips_llama_tokens_for_claude():
    engine = RulesEngine()
    profile = engine.get_profile("claude")
    features = PromptAnalyzer().analyze(LLAMA2_PROMPT)
    result = PromptTransformer().transform(LLAMA2_PROMPT, profile, features)
    assert "[INST]" not in result.adapted
    assert "<<SYS>>" not in result.adapted


def test_transformer_removes_cot_for_claude():
    engine = RulesEngine()
    profile = engine.get_profile("claude")
    features = PromptAnalyzer().analyze(PLAIN_PROMPT)
    result = PromptTransformer().transform(PLAIN_PROMPT, profile, features, model_variant="3")
    assert "think step by step" not in result.adapted.lower()


def test_transformer_no_changes_clean_prompt():
    engine = RulesEngine()
    profile = engine.get_profile("gpt")
    clean = "You are a helpful assistant. Answer concisely."
    features = PromptAnalyzer().analyze(clean)
    result = PromptTransformer().transform(clean, profile, features)
    # No structural tokens to strip, no CoT, has no length issue
    assert result.adapted.strip() == clean.strip()


def test_transformer_records_changes():
    engine = RulesEngine()
    profile = engine.get_profile("claude")
    features = PromptAnalyzer().analyze(LLAMA2_PROMPT)
    result = PromptTransformer().transform(LLAMA2_PROMPT, profile, features)
    assert len(result.changes) > 0
