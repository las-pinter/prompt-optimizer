"""
tests/test_core.py
Basic tests for analyzer, rules engine, and transformer.
"""

import pytest
from prompt_optimizer.analyzer import PromptAnalyzer
from prompt_optimizer.rules_engine import RulesEngine, ConditionEvaluator
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


def test_engine_list_shows_new_models():
    """Test that newly added model families are discoverable."""
    engine = RulesEngine()
    models = engine.list_models()
    assert "deepseek" in models
    assert "kimi" in models
    assert "glm" in models
    assert "minimax" in models
    assert "grok" in models
    assert "phi" in models
    assert "cohere" in models


def test_engine_sub_families_loaded():
    """Test that sub_families are loaded from YAML."""
    engine = RulesEngine()
    claude = engine.get_profile("claude")
    assert len(claude.sub_families) > 0
    assert "claude-3" in claude.sub_families
    assert "claude-4" in claude.sub_families


def test_engine_get_sub_family():
    """Test sub-family resolution."""
    engine = RulesEngine()
    sf = engine.get_sub_family("claude", "claude-3.5-sonnet")
    assert sf is not None
    assert sf.key == "claude-3.5"
    assert sf.cot_policy == "strip"
    assert sf.examples == "required"


def test_engine_get_sub_family_claude4():
    """Test claude-4 sub-family resolution."""
    engine = RulesEngine()
    sf = engine.get_sub_family("claude", "claude-4-opus")
    assert sf is not None
    assert sf.key == "claude-4"
    assert sf.agent_mode is True


def test_engine_get_sub_family_no_match():
    """Test that non-matching variant returns None."""
    engine = RulesEngine()
    sf = engine.get_sub_family("claude", "unknown-model")
    assert sf is None


# ---------------------------------------------------------------------------
# Condition evaluator
# ---------------------------------------------------------------------------


def test_condition_starts_with():
    ev = ConditionEvaluator()
    assert ev.evaluate('model_variant starts_with "claude-3"', "claude-3.5-sonnet", None, None, 0, None) is True
    assert ev.evaluate('model_variant starts_with "claude-3"', "claude-4-opus", None, None, 0, None) is False


def test_condition_in_list():
    ev = ConditionEvaluator()
    assert ev.evaluate('model_variant in ["o1", "o3"]', "o1", None, None, 0, None) is True
    assert ev.evaluate('model_variant in ["o1", "o3"]', "o4", None, None, 0, None) is False


def test_condition_version_gte():
    ev = ConditionEvaluator()
    assert ev.evaluate("model_version >= 3", None, 3.5, None, 0, None) is True
    assert ev.evaluate("model_version >= 3", None, 2.1, None, 0, None) is False


def test_condition_and():
    ev = ConditionEvaluator()
    assert ev.evaluate('model_variant starts_with "claude-3" and model_version >= 3.5',
                       "claude-3.5-sonnet", 3.5, None, 0, None) is True
    assert ev.evaluate('model_variant starts_with "claude-3" and model_version >= 3.5',
                       "claude-3-haiku", 3.0, None, 0, None) is False


def test_condition_or():
    ev = ConditionEvaluator()
    assert ev.evaluate('model_variant starts_with "claude-3" or model_variant starts_with "claude-4"',
                       "claude-4-opus", None, None, 0, None) is True
    assert ev.evaluate('model_variant starts_with "claude-3" or model_variant starts_with "claude-4"',
                       "gpt-4", None, None, 0, None) is False


def test_condition_no_condition():
    ev = ConditionEvaluator()
    assert ev.evaluate(None, "anything", None, None, 0, None) is True


def test_condition_boolean_flag():
    ev = ConditionEvaluator()
    assert ev.evaluate("agent_mode == true", None, None, None, 0, {"agent_mode": True}) is True
    assert ev.evaluate("agent_mode == true", None, None, None, 0, {"agent_mode": False}) is False


def test_condition_prompt_length():
    ev = ConditionEvaluator()
    assert ev.evaluate("prompt_length > 300", None, None, None, 500, None) is True
    assert ev.evaluate("prompt_length > 300", None, None, None, 100, None) is False


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
    result = PromptTransformer().transform(PLAIN_PROMPT, profile, features, model_variant="claude-3")
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


def test_transformer_sub_family_resolved():
    """Test that sub-family info appears in result."""
    engine = RulesEngine()
    profile = engine.get_profile("claude")
    features = PromptAnalyzer().analyze(PLAIN_PROMPT)
    result = PromptTransformer().transform(PLAIN_PROMPT, profile, features, model_variant="claude-4-opus")
    assert result.model_variant == "claude-4-opus"
    assert result.sub_family == "claude-4"


def test_transformer_gpt_o_series_strips_cot():
    """Test that CoT is removed for o-series reasoning models."""
    engine = RulesEngine()
    profile = engine.get_profile("gpt")
    prompt_with_cot = "You are a math assistant. Think step by step. Solve: 2+2"
    features = PromptAnalyzer().analyze(prompt_with_cot)
    result = PromptTransformer().transform(prompt_with_cot, profile, features, model_variant="o3")
    assert "think step by step" not in result.adapted.lower()


def test_transformer_llama3_template():
    """Test Llama 3 template wrapping."""
    engine = RulesEngine()
    profile = engine.get_profile("llama")
    prompt = "You are a helpful assistant."
    features = PromptAnalyzer().analyze(prompt)
    result = PromptTransformer().transform(prompt, profile, features, model_variant="llama-3.1")
    assert "<|begin_of_text|>" in result.adapted or "<|start_header_id|>" in result.adapted


def test_transformer_deepseek_rag():
    """Test that deepseek-v4 sub-family is loaded."""
    engine = RulesEngine()
    profile = engine.get_profile("deepseek")
    sf = engine.get_sub_family("deepseek", "deepseek-v4-pro")
    assert sf is not None
    assert sf.raw.get("rag_focused") is True


def test_transformer_kimi_indexing():
    """Test that kimi sub-family has document indexing config."""
    engine = RulesEngine()
    profile = engine.get_profile("kimi")
    sf = engine.get_sub_family("kimi", "kimi-k2.5")
    assert sf is not None
    assert sf.raw.get("document_indexing") is True
    assert sf.raw.get("context_window_tokens") == 2500000


def test_transformer_cohere_rag_base():
    """Test that cohere has RAG base prompt."""
    engine = RulesEngine()
    profile = engine.get_profile("cohere")
    sf = engine.get_sub_family("cohere", "command-r-plus")
    assert sf is not None
    assert "sources" in sf.raw.get("base_prompt", "").lower()


def test_transformer_result_diff_summary():
    """Test that diff_summary includes sub_family info."""
    engine = RulesEngine()
    profile = engine.get_profile("claude")
    features = PromptAnalyzer().analyze(PLAIN_PROMPT)
    result = PromptTransformer().transform(PLAIN_PROMPT, profile, features, model_variant="claude-4-opus")
    summary = result.diff_summary()
    assert "claude-4-opus" in summary
    assert "claude-4" in summary
