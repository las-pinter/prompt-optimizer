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
from .rules_engine import ConditionEvaluator, ModelProfile, SubFamilyProfile, Transformation


@dataclass
class TransformResult:
    original: str
    adapted: str
    model_id: str
    model_variant: str | None = None
    sub_family: str | None = None
    changes: list[str] = field(default_factory=list)
    lint_warnings: list[str] = field(default_factory=list)

    def diff_summary(self) -> str:
        lines = [f"Target model: {self.model_id}"]
        if self.model_variant:
            lines.append(f"  Variant: {self.model_variant}")
        if self.sub_family:
            lines.append(f"  Sub-family: {self.sub_family}")
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
_ALPACA_HEADERS = re.compile(r"###\s*(Instruction|Input|Response)\s*:", re.IGNORECASE)
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

    def __init__(self) -> None:
        self._condition_evaluator = ConditionEvaluator()

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
            model_variant=model_variant,
        )

        # Resolve sub-family if available
        sub_family = None
        if model_variant:
            sub_family = self._resolve_sub_family(profile, model_variant)
            if sub_family:
                result.sub_family = sub_family.key

        # Build extra flags from sub_family config
        extra_flags: dict[str, bool] = {}
        if sub_family:
            extra_flags["agent_mode"] = sub_family.agent_mode
            if sub_family.example_injection:
                trigger = sub_family.example_injection.get("trigger", "")
                extra_flags[trigger] = True

        for t in profile.transformations:
            self._apply(t, result, features, profile, model_variant, sub_family, extra_flags)

        # Always emit the model's static warnings as lint notes
        result.lint_warnings.extend(profile.warnings)

        return result

    # ------------------------------------------------------------------
    # Sub-family resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_sub_family(
        profile: ModelProfile, variant: str,
    ) -> SubFamilyProfile | None:
        """Match a variant string to a sub_family key."""
        sub_families = profile.sub_families
        if not sub_families:
            return None

        # Try longest prefix match first
        for key in sorted(sub_families.keys(), key=len, reverse=True):
            if variant.startswith(key):
                return sub_families[key]

        # Try exact match
        if variant in sub_families:
            return sub_families[variant]

        # Try partial match (variant contains key)
        for key in sub_families:
            if key in variant:
                return sub_families[key]

        return None

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
        sub_family: SubFamilyProfile | None,
        extra_flags: dict[str, bool] | None,
    ) -> None:
        # Evaluate condition
        if not self._condition_evaluator.evaluate(
            t.condition,
            model_variant=model_variant,
            model_version=self._extract_version(model_variant),
            features=features,
            prompt_char_count=len(result.adapted),
            extra_flags=extra_flags,
        ):
            return  # Condition not met — skip this transformation

        handler = getattr(self, f"_handle_{t.type}", None)
        if handler is None:
            result.lint_warnings.append(
                f"[{t.id}] Unknown transformation type '{t.type}' — skipped."
            )
            return
        handler(t, result, features, profile, model_variant, sub_family, extra_flags or {})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_version(variant: str | None) -> float | None:
        if not variant:
            return None
        m = re.search(r"(\d+(?:\.\d+)?)", variant)
        return float(m.group(1)) if m else None

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _handle_regex_replace(
        self, t: Transformation, result: TransformResult, features: PromptFeatures,
        profile: ModelProfile, *_
    ) -> None:
        pattern = re.compile(t.params.get("pattern", ""), re.IGNORECASE)
        replacement = t.params.get("replacement", "")
        new_text, n = pattern.subn(replacement, result.adapted)
        if n > 0:
            result.adapted = new_text
            result.changes.append(f"[{t.id}] {t.description} ({n} occurrence(s) replaced)")

    def _handle_phrase_remove(
        self, t: Transformation, result: TransformResult, features: PromptFeatures,
        profile: ModelProfile, *_
    ) -> None:
        phrases = t.params.get("phrases", [])
        combined = re.compile("|".join(re.escape(p) for p in phrases), re.IGNORECASE)
        new_text, n = combined.subn("", result.adapted)
        if n > 0:
            result.adapted = new_text.strip()
            result.changes.append(f"[{t.id}] {t.description} ({n} phrase(s) removed)")

    def _handle_lint_warning(
        self, t: Transformation, result: TransformResult, features: PromptFeatures,
        profile: ModelProfile, *_
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
        self, t: Transformation, result: TransformResult, features: PromptFeatures,
        profile: ModelProfile, *_
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
        profile: ModelProfile, model_variant: str | None,
        sub_family: SubFamilyProfile | None, _extra_flags: dict[str, bool],
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
        self, t: Transformation, result: TransformResult, features: PromptFeatures,
        profile: ModelProfile, *_
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

    # ------------------------------------------------------------------
    # New handlers for sub_family-aware transformations
    # ------------------------------------------------------------------

    def _handle_example_inject(
        self, t: Transformation, result: TransformResult, features: PromptFeatures,
        profile: ModelProfile, model_variant: str | None,
        sub_family: SubFamilyProfile | None, extra_flags: dict[str, bool],
    ) -> None:
        """Insert <example> blocks when structured output is expected."""
        if not sub_family or not sub_family.example_injection:
            return

        trigger = sub_family.example_injection.get("trigger", "")
        if not extra_flags.get(trigger):
            return

        # Heuristic: user prompt contains JSON schema, array, table reference
        if not (re.search(r"\{.*\}", result.adapted) or
                re.search(r"\b(json|schema|array|table|list of|output format)\b", result.adapted, re.IGNORECASE)):
            return

        template = sub_family.example_injection.get("template", "")
        position = sub_family.example_injection.get("position", "before_instructions")

        if template:
            injected = template
        else:
            injected = (
                "\n<example>\n"
                "Input:  {{ user_example_input }}\n"
                "Output: {{ user_example_output }}\n"
                "</example>"
            )

        if position == "before_instructions":
            # Insert before the last block of text (assumed to be instructions)
            lines = result.adapted.rstrip().split("\n")
            if len(lines) >= 2:
                # Insert before the last non-empty line
                insert_idx = len(lines)
                for i in range(len(lines) - 1, -1, -1):
                    if lines[i].strip():
                        insert_idx = i
                        break
                lines.insert(insert_idx, injected)
                result.adapted = "\n".join(lines)
            else:
                result.adapted += "\n" + injected
        else:
            result.adapted += "\n" + injected

        result.changes.append(f"[{t.id}] Injected <example> block for structured output")

    def _handle_xml_rewrite(
        self, t: Transformation, result: TransformResult, features: PromptFeatures,
        profile: ModelProfile, model_variant: str | None,
        sub_family: SubFamilyProfile | None, extra_flags: dict[str, bool],
    ) -> None:
        """Rewrite a block of text into a specific XML tag structure."""
        if not sub_family:
            return

        target_block = t.params.get("target_block", "")
        xml_tag = sub_family.xml_tags.get(target_block, f"<{target_block}>")
        xml_close = f"</{target_block}>"

        # Detect unstructured directive text (lines that don't already have XML tags)
        # Strategy: find the main instruction block and wrap it
        lines = result.adapted.strip().split("\n")
        wrapped_lines = []
        in_block = False

        for line in lines:
            stripped = line.strip()
            if not in_block and stripped and not stripped.startswith("<"):
                wrapped_lines.append(f"{xml_tag}{stripped}")
                in_block = True
            elif in_block and stripped:
                wrapped_lines.append(stripped)
            else:
                wrapped_lines.append(line)

        if in_block:
            wrapped_lines.append(xml_close)
            result.adapted = "\n".join(wrapped_lines)
            result.changes.append(f"[{t.id}] Wrapped '{target_block}' in XML tags")

    def _handle_doc_wrap(
        self, t: Transformation, result: TransformResult, features: PromptFeatures,
        profile: ModelProfile, model_variant: str | None,
        sub_family: SubFamilyProfile | None, extra_flags: dict[str, bool],
    ) -> None:
        """Wrap source documents in <doc id="N"> tags with metadata."""
        if not sub_family or not sub_family.agent_mode:
            return

        doc_tag_template = sub_family.xml_tags.get("doc", '<doc id="{{doc_id}}">')
        documents_tag = sub_family.xml_tags.get("documents", "<documents>")
        docs = t.params.get("documents", [])

        if not docs:
            return

        doc_parts = []
        for i, doc in enumerate(docs, 1):
            doc_id = doc.get("id", str(i))
            content = doc.get("content", "")
            source = doc.get("source", "")
            open_tag = doc_tag_template.replace("{{doc_id}}", str(doc_id))
            part = f"{open_tag}\n{content}"
            if source:
                part += f"\n_source: {source}"
            part += f"\n</doc>"
            doc_parts.append(part)

        if doc_parts:
            result.adapted += "\n" + documents_tag + "\n" + "\n\n".join(doc_parts) + "\n</documents>"
            result.changes.append(f"[{t.id}] Wrapped {len(doc_parts)} document(s) in <doc> tags")

    def _handle_cot_policy(
        self, t: Transformation, result: TransformResult, features: PromptFeatures,
        profile: ModelProfile, model_variant: str | None,
        sub_family: SubFamilyProfile | None, extra_flags: dict[str, bool],
    ) -> None:
        """Apply CoT policy from sub_family (strip / keep / inject)."""
        if not sub_family:
            return

        policy = sub_family.cot_policy
        if policy == "strip":
            # Remove CoT phrases
            new_text, n = _COT_PATTERN.subn("", result.adapted)
            if n > 0:
                result.adapted = new_text.strip()
                result.changes.append(f"[{t.id}] Stripped {n} CoT phrase(s) per sub_family policy")
        elif policy == "inject" and not features.has_cot_instruction:
            phrase = t.params.get("phrase", "Think step by step.")
            result.adapted = result.adapted.rstrip() + "\n" + phrase
            result.changes.append(f"[{t.id}] Injected CoT instruction per sub_family policy")

    def _handle_prompt_style(
        self, t: Transformation, result: TransformResult, features: PromptFeatures,
        profile: ModelProfile, model_variant: str | None,
        sub_family: SubFamilyProfile | None, extra_flags: dict[str, bool],
    ) -> None:
        """Apply prompt style from sub_family (xml_block, xml_with_examples, agent_directives, etc.)."""
        if not sub_family:
            return

        style = sub_family.prompt_style
        if style == "xml_block":
            # Wrap context in <context> and instructions in <instructions>
            context_tag = sub_family.xml_tags.get("context", "<context>")
            instructions_tag = sub_family.xml_tags.get("instructions", "<instructions>")

            # Split prompt into context (first part) and instructions (last part)
            lines = result.adapted.strip().split("\n")
            # Heuristic: first block = context, last block = instructions
            mid = max(1, len(lines) // 2)
            context_lines = lines[:mid]
            instruction_lines = lines[mid:]

            context_text = "\n".join(l for l in context_lines if l.strip())
            instruction_text = "\n".join(l for l in instruction_lines if l.strip())

            result.adapted = f"{context_tag}\n{context_text}\n{context_tag.replace('>', '/>')}\n{instructions_tag}\n{instruction_text}\n{instructions_tag.replace('>', '/>')}"
            result.changes.append(f"[{t.id}] Rewrote prompt in '{style}' XML block format")

        elif style == "agent_directives" and sub_family.agent_mode:
            # Wrap behavior rules in <system_directives> and sources in <documents>
            directives_tag = sub_family.xml_tags.get("system_directives", "<system_directives>")
            directives_close = "</system_directives>"

            behavior = sub_family.agent_directives.get("behavior", "")
            sources = sub_family.agent_directives.get("sources", "")

            # Detect behavior rules (lines starting with "Act as", "You are", "Your role", etc.)
            behavior_lines = []
            other_lines = []
            for line in result.adapted.strip().split("\n"):
                stripped = line.strip()
                if re.match(r"^(act\s+as|you\s+are|your\s+role|your\s+task|you\s+will)", stripped, re.IGNORECASE):
                    behavior_lines.append(stripped)
                else:
                    other_lines.append(line)

            if behavior_lines:
                behavior_text = "\n".join(behavior_lines)
                other_text = "\n".join(other_lines)
                result.adapted = (
                    f"{directives_tag}\n{behavior_text}\n{directives_close}\n\n{other_text}"
                )
                result.changes.append(f"[{t.id}] Wrapped behavior rules in <system_directives>")

            if behavior:
                result.adapted = f"{directives_tag}\n{behavior}\n{directives_close}\n\n{result.adapted}"
                result.changes.append(f"[{t.id}] Injected agent behavior directives")
