"""
transformer.py
Applies model-specific transformations to a prompt based on a ModelProfile
and the detected PromptFeatures. Returns the adapted prompt and a change log.
No LLMs involved — all transformations are deterministic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .analyzer import PromptFeatures
from .rules_engine import ModelProfile, Transformation


@dataclass
class TransformResult:
    original: str
    adapted: str
    model_id: str
    changes: list[str] = field(default_factory=list)
    lint_warnings: list[str] = field(default_factory=list)

    def diff_summary(self) -> str:
        lines = [f"Target model: {self.model_id}"]
        if self.changes:
            lines.append("\nTransformations applied:")
            lines.extend(f"  • {c}" for c in self.changes)
        else:
            lines.append("\nNo structural transformations needed.")
        if self.lint_warnings:
            lines.append("\nLint warnings:")
            lines.extend(f"  ⚠ {w}" for w in self.lint_warnings)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Regex patterns used by transformations
# ---------------------------------------------------------------------------
_LLAMA_TOKENS = re.compile(r"\[/?INST\]|<</?SYS>>")
_LLAMA3_TOKENS = re.compile(
    r"<\|begin_of_text\|>|<\|start_header_id\|>.*?<\|end_header_id\|>|<\|eot_id\|>",
    re.DOTALL,
)
_CHATML_TOKENS = re.compile(r"<\|im_(start|end)\|>(system|user|assistant)?\n?")
_ALPACA_HEADERS = re.compile(r"###\s*(Instruction|Input|Response)\s*:\s*", re.IGNORECASE)
_SYS_TAGS = re.compile(r"<<SYS>>|<</SYS>>")

# CoT phrase variants to strip
_COT_PHRASES = [
    r"think\s+step\s+by\s+step\.?",
    r"let['']s\s+think\s+through\s+this\s+step\s+by\s+step\.?",
    r"think\s+carefully\s+step\s+by\s+step\.?",
    r"reason\s+step\s+by\s+step\.?",
]
_COT_PATTERN = re.compile("|".join(_COT_PHRASES), re.IGNORECASE)

# XML tag pattern for lint detection
_XML_TAGS = re.compile(r"<[a-zA-Z][a-zA-Z0-9_\-]*>.*?</[a-zA-Z][a-zA-Z0-9_\-]*>", re.DOTALL)
_MARKDOWN_HEADERS = re.compile(r"^#{1,4}\s+\w", re.MULTILINE)
_LENGTH_INSTRUCTIONS = re.compile(
    r"\b(brief|concise|short|verbose|detailed|in \d+ (words|sentences)|no more than|under \d+)\b",
    re.IGNORECASE,
)


class PromptTransformer:
    """
    Applies transformations from a ModelProfile to a prompt string,
    guided by the detected PromptFeatures.
    """

    def transform(
        self,
        prompt: str,
        profile: ModelProfile,
        features: PromptFeatures,
        model_variant: str | None = None,
    ) -> TransformResult:
        result = TransformResult(
            original=prompt,
            adapted=prompt,
            model_id=profile.model_id,
        )

        for t in profile.transformations:
            self._apply(t, result, features, profile, model_variant)

        # Always emit the model's static warnings as lint notes
        result.lint_warnings.extend(profile.warnings)

        return result

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _apply(
        self,
        t: Transformation,
        result: TransformResult,
        features: PromptFeatures,
        profile: ModelProfile,
        model_variant: str | None,
    ) -> None:
        handler = getattr(self, f"_handle_{t.type}", None)
        if handler is None:
            result.lint_warnings.append(
                f"[{t.id}] Unknown transformation type '{t.type}' — skipped."
            )
            return
        handler(t, result, features, profile, model_variant)

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _handle_regex_replace(
        self, t: Transformation, result: TransformResult, features: PromptFeatures, *_
    ) -> None:
        pattern = re.compile(t.params.get("pattern", ""), re.IGNORECASE)
        replacement = t.params.get("replacement", "")
        new_text, n = pattern.subn(replacement, result.adapted)
        if n > 0:
            result.adapted = new_text
            result.changes.append(f"[{t.id}] {t.description} ({n} occurrence(s) replaced)")

    def _handle_phrase_remove(
        self, t: Transformation, result: TransformResult, features: PromptFeatures, *_
    ) -> None:
        phrases = t.params.get("phrases", [])
        combined = re.compile(
            "|".join(re.escape(p) for p in phrases), re.IGNORECASE
        )
        new_text, n = combined.subn("", result.adapted)
        if n > 0:
            result.adapted = new_text.strip()
            result.changes.append(f"[{t.id}] {t.description} ({n} phrase(s) removed)")

    def _handle_lint_warning(
        self, t: Transformation, result: TransformResult, features: PromptFeatures, *_
    ) -> None:
        check = t.params.get("check", "")
        message = t.params.get("message", t.description)
        triggered = False

        if check == "no_length_instruction":
            triggered = not features.has_length_instruction
        elif check == "instruction_nesting_depth > 2":
            triggered = features.instruction_nesting_depth > 2
        elif check == "prompt_length > 300 and no_markdown_headers":
            triggered = features.char_count > 300 and not features.has_markdown_headers

        if triggered:
            result.lint_warnings.append(f"[{t.id}] {message}")

    def _handle_phrase_inject(
        self, t: Transformation, result: TransformResult, features: PromptFeatures, *_
    ) -> None:
        # Only inject if CoT not already present
        if features.has_cot_instruction:
            return
        phrase = t.params.get("phrase", "")
        position = t.params.get("position", "end_of_system")
        if position == "end_of_system":
            result.adapted = result.adapted.rstrip() + "\n" + phrase
            result.changes.append(f"[{t.id}] {t.description}")

    def _handle_chat_template_wrap(
        self, t: Transformation, result: TransformResult, features: PromptFeatures,
        profile: ModelProfile, model_variant: str | None
    ) -> None:
        # Only wrap if no template tokens already present
        if features.has_llama2_tokens or features.has_llama3_tokens or features.has_chatml_tokens:
            result.lint_warnings.append(
                f"[{t.id}] Template tokens already detected — skipping wrap to avoid double-wrapping."
            )
            return

        template_key = t.params.get("template")
        templates = profile.chat_template
        template = templates.get(template_key)
        if not template or not isinstance(template, dict):
            result.lint_warnings.append(
                f"[{t.id}] Template '{template_key}' not found in profile — skipped."
            )
            return

        fmt = template.get("format", "")
        if fmt and "{system}" in fmt:
            result.adapted = fmt.replace("{system}", result.adapted).replace("{user}", "<USER_MESSAGE>")
            result.changes.append(f"[{t.id}] Wrapped in {template_key} chat template")

    def _handle_xml_to_headers(
        self, t: Transformation, result: TransformResult, features: PromptFeatures, *_
    ) -> None:
        if not features.has_xml_tags:
            return
        # Convert <tag>content</tag> to "TAG:\ncontent" style plain headers
        def replacer(m: re.Match) -> str:
            full = m.group(0)
            tag_match = re.match(r"<([a-zA-Z][a-zA-Z0-9_\-]*)>(.*?)</\1>", full, re.DOTALL)
            if tag_match:
                tag, content = tag_match.group(1), tag_match.group(2).strip()
                return f"{tag.upper()}:\n{content}"
            return full

        new_text = _XML_TAGS.sub(replacer, result.adapted)
        if new_text != result.adapted:
            result.adapted = new_text
            result.changes.append(f"[{t.id}] {t.description}")
