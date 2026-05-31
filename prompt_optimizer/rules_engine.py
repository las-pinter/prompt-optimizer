"""
rules_engine.py
Loads model profiles from data/models/*.yaml and exposes them for the transformer.
No LLMs involved — pure data-driven rule lookup.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


DATA_DIR = Path(__file__).parent.parent / "data" / "models"


@dataclass
class LintWarning:
    rule_id: str
    message: str
    severity: str = "warning"  # "warning" | "error" | "info"


@dataclass
class Transformation:
    id: str
    type: str
    description: str
    params: dict[str, Any] = field(default_factory=dict)
    condition: str | None = None


@dataclass
class SubFamilyProfile:
    """Profile for a sub-family (version group) within a model family."""
    key: str
    prompt_style: str = "default"
    xml_tags: dict[str, str] = field(default_factory=dict)
    cot_policy: str = "default"  # "default" | "strip" | "keep" | "inject"
    examples: str = "optional"   # "optional" | "required" | "never"
    agent_mode: bool = False
    agent_directives: dict[str, str] = field(default_factory=dict)
    example_injection: dict[str, Any] = field(default_factory=dict)
    prompt_style_config: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelProfile:
    model_id: str
    display_name: str
    vendor: str
    api_type: str
    chat_template: dict[str, Any]
    preferences: dict[str, Any]
    warnings: list[str]
    transformations: list[Transformation]
    sources: list[str]
    raw: dict[str, Any]
    sub_families: dict[str, SubFamilyProfile] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelProfile":
        transformations = [
            Transformation(
                id=t["id"],
                type=t["type"],
                description=t.get("description", ""),
                params={k: v for k, v in t.items() if k not in ("id", "type", "description", "condition")},
                condition=t.get("condition"),
            )
            for t in data.get("transformations", [])
        ]
        sub_families: dict[str, SubFamilyProfile] = {}
        for key, sf_data in data.get("sub_families", {}).items():
            sf_raw = dict(sf_data) if isinstance(sf_data, dict) else {}
            sf = SubFamilyProfile(
                key=key,
                prompt_style=sf_raw.get("prompt_style", "default"),
                xml_tags=sf_raw.get("xml_tags", {}),
                cot_policy=sf_raw.get("cot_policy", "default"),
                examples=sf_raw.get("examples", "optional"),
                agent_mode=sf_raw.get("agent_mode", False),
                agent_directives=sf_raw.get("agent_directives", {}),
                example_injection=sf_raw.get("example_injection", {}),
                prompt_style_config={k: v for k, v in sf_raw.items()
                                     if k not in ("prompt_style", "xml_tags", "cot_policy",
                                                  "examples", "agent_mode", "agent_directives",
                                                  "example_injection", "raw")},
                raw=sf_raw.get("raw", sf_raw),
            )
            sub_families[key] = sf
        return cls(
            model_id=data["model_id"],
            display_name=data["display_name"],
            vendor=data["vendor"],
            api_type=data["api_type"],
            chat_template=data.get("chat_template", {}),
            preferences=data.get("preferences", {}),
            warnings=data.get("warnings", []),
            transformations=transformations,
            sources=data.get("sources", []),
            raw=data,
            sub_families=sub_families,
        )


class ConditionEvaluator:
    """
    Evaluates transformation condition strings against model metadata.

    Supported syntax (all one-liner, no multi-line):
      - model_variant == "x"
      - model_variant != "x"
      - model_variant in ["a", "b", "c"]
      - model_variant starts_with "prefix"
      - model_version >= 3
      - model_version > 2
      - model_version <= 4
      - model_version < 5
      - task_type == "reasoning"
      - has_cot_instruction
      - not has_cot_instruction
      - expects_structured_output
      - prompt_length > 300
      - prompt_length < 1000
      - no_length_instruction
      - no_markdown_headers
      - has_xml_tags
      - not has_xml_tags
      - agent_mode == true
      - compound: "A and B", "A or B", "(A) and B"
    """

    # Pattern: "starts_with <value>"
    _RE_STARTS_WITH = re.compile(r"^model_variant\s+starts_with\s+\"(.+?)\"$")
    # Pattern: "in [...]"
    _RE_IN = re.compile(r"^model_variant\s+in\s+\[([^\]]+)\]$")
    # Pattern: numeric comparison "model_version OP N"
    _RE_VERSION_CMP = re.compile(
        r"^model_version\s+(>=|<=|==|!=|>|<)\s+(\d+(?:\.\d+)?)$"
    )
    # Pattern: boolean flag "has_xxx" or "not has_xxx"
    _RE_BOOL_FLAG = re.compile(r"^(not\s+)?(has_\w+|expects_\w+|no_\w+|agent_mode)\s*(==\s*(true|false))?$")
    # Pattern: "prompt_length OP N"
    _RE_PROMPT_LEN = re.compile(r"^prompt_length\s+(>=|<=|==|!=|>|<)\s+(\d+)$")
    # Pattern: "task_type == \"x\""
    _RE_TASK_TYPE = re.compile(r"^task_type\s+==\s+\"(.+?)\"$")

    def evaluate(
        self,
        condition: str | None,
        model_variant: str | None,
        model_version: float | None,
        features: Any | None = None,
        prompt_char_count: int = 0,
        extra_flags: dict[str, bool] | None = None,
    ) -> bool:
        """
        Evaluate a condition string.

        Args:
            condition: The condition string (e.g. 'model_variant in ["o1", "o3"]').
            model_variant: The model variant (e.g. 'claude-3.5-sonnet').
            model_version: Numeric version (e.g. 3.5).
            features: PromptFeatures object (for has_xxx flags).
            prompt_char_count: Character count of the prompt.
            extra_flags: Additional boolean flags (e.g. {'expects_structured_output': True}).

        Returns:
            True if the condition is satisfied.
        """
        if condition is None:
            return True  # No condition = always apply

        # Handle compound conditions (and / or)
        for op in (" and ", " or "):
            # Find the operator outside of parentheses
            depth = 0
            for i in range(len(condition) - len(op)):
                if condition[i] == "(":
                    depth += 1
                elif condition[i] == ")":
                    depth -= 1
                if depth == 0 and condition[i:i + len(op)] == op:
                    left = condition[:i].strip()
                    right = condition[i + len(op):].strip()
                    left_result = self.evaluate(
                        left, model_variant, model_version, features, prompt_char_count, extra_flags,
                    )
                    right_result = self.evaluate(
                        right, model_variant, model_version, features, prompt_char_count, extra_flags,
                    )
                    return left_result or right_result if op == " or " else left_result and right_result

        # Simple conditions (no compound operators)
        result = self._evaluate_simple(
            condition, model_variant, model_version, features, prompt_char_count, extra_flags,
        )
        return result

    def _evaluate_simple(
        self,
        condition: str,
        model_variant: str | None,
        model_version: float | None,
        features: Any | None,
        prompt_char_count: int,
        extra_flags: dict[str, bool] | None,
    ) -> bool:
        condition = condition.strip()

        # starts_with
        m = self._RE_STARTS_WITH.match(condition)
        if m:
            prefix = m.group(1)
            return model_variant is not None and model_variant.startswith(prefix)

        # in [...]
        m = self._RE_IN.match(condition)
        if m:
            items = [item.strip().strip('"').strip("'") for item in m.group(1).split(",")]
            return model_variant is not None and model_variant in items

        # model_version comparisons
        m = self._RE_VERSION_CMP.match(condition)
        if m:
            op = m.group(1)
            target = float(m.group(2))
            if model_version is None:
                return False
            return self._compare_version(model_version, op, target)

        # task_type
        m = self._RE_TASK_TYPE.match(condition)
        if m:
            target = m.group(1)
            return extra_flags and extra_flags.get("task_type") == target

        # prompt_length comparisons
        m = self._RE_PROMPT_LEN.match(condition)
        if m:
            op = m.group(1)
            target = int(m.group(2))
            return self._compare_value(prompt_char_count, op, target)

        # boolean flags
        m = self._RE_BOOL_FLAG.match(condition)
        if m:
            is_negated = m.group(1) is not None  # "not" prefix
            flag_name = m.group(2)
            value_check = m.group(3)  # optional "== true" or "== false"

            flag_value = self._get_flag_value(flag_name, features, extra_flags)

            if value_check:
                expected = value_check.split()[-1] == "true"
                result = flag_value == expected
            else:
                result = bool(flag_value)

            return not result if is_negated else result

        # Fallback: treat as always-true if we can't parse it
        return True

    def _get_flag_value(
        self, flag_name: str, features: Any | None, extra_flags: dict[str, bool] | None,
    ) -> bool:
        """Resolve a boolean flag from features or extra_flags."""
        # Check extra_flags first
        if extra_flags and flag_name in extra_flags:
            return extra_flags[flag_name]

        # Check features (PromptFeatures)
        if features is not None and hasattr(features, flag_name):
            value = getattr(features, flag_name)
            if isinstance(value, bool):
                return value
            return bool(value)

        # Default for "no_xxx" flags → True means the "no" condition is active
        if flag_name.startswith("no_"):
            return False
        return False

    @staticmethod
    def _compare_version(actual: float, op: str, target: float) -> bool:
        if op == ">=":
            return actual >= target
        if op == "<=":
            return actual <= target
        if op == "==":
            return abs(actual - target) < 0.01
        if op == "!=":
            return abs(actual - target) >= 0.01
        if op == ">":
            return actual > target
        if op == "<":
            return actual < target
        return False

    @staticmethod
    def _compare_value(actual, op: str, target) -> bool:
        if op == ">=":
            return actual >= target
        if op == "<=":
            return actual <= target
        if op == "==":
            return actual == target
        if op == "!=":
            return actual != target
        if op == ">":
            return actual > target
        if op == "<":
            return actual < target
        return False


class RulesEngine:
    """
    Loads all model YAML profiles and provides lookup + rule evaluation.
    """

    def __init__(self, data_dir: Path | None = None):
        self._data_dir = data_dir or DATA_DIR
        self._profiles: dict[str, ModelProfile] = {}
        self._condition_evaluator = ConditionEvaluator()
        self._load_all()

    def _load_all(self) -> None:
        for yaml_file in self._data_dir.glob("*.yaml"):
            with yaml_file.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if data and "model_id" in data:
                profile = ModelProfile.from_dict(data)
                self._profiles[profile.model_id] = profile

    def list_models(self) -> list[str]:
        return sorted(self._profiles.keys())

    def get_profile(self, model_id: str) -> ModelProfile:
        if model_id not in self._profiles:
            available = ", ".join(self.list_models())
            raise ValueError(
                f"Unknown model '{model_id}'. Available models: {available}"
            )
        return self._profiles[model_id]

    def get_warnings(self, model_id: str) -> list[str]:
        return self.get_profile(model_id).warnings

    def get_preferences(self, model_id: str) -> dict[str, Any]:
        return self.get_profile(model_id).preferences

    def get_sub_family(self, model_id: str, variant: str) -> SubFamilyProfile | None:
        """
        Resolve a sub-family profile for a given model variant.

        Matches variant against sub_family keys:
          - Exact match: "claude-3.5-sonnet" → checks if any key is a prefix
          - Falls back to first sub_family if no match

        Args:
            model_id: The model family ID (e.g. "claude").
            variant: The model variant string (e.g. "claude-3.5-sonnet").

        Returns:
            The matching SubFamilyProfile, or None if no sub_families defined or no match.
        """
        profile = self.get_profile(model_id)
        sub_families = profile.sub_families
        if not sub_families:
            return None

        # Try prefix match: "claude-3.5-sonnet" → "claude-3.5"
        for key in sorted(sub_families.keys(), key=len, reverse=True):
            if variant.startswith(key):
                return sub_families[key]

        # Try exact match
        if variant in sub_families:
            return sub_families[variant]

        # Try matching with version number prefix: "3.5" → "claude-3.5"
        # (for variants that are just version numbers like "3", "3.5", "4")
        for key in sub_families:
            if variant in key:
                return sub_families[key]

        return None

    def evaluate_condition(
        self,
        condition: str | None,
        model_id: str,
        variant: str | None,
        features: Any | None = None,
        prompt_char_count: int = 0,
        extra_flags: dict[str, bool] | None = None,
    ) -> bool:
        """
        Evaluate a transformation condition against the current model context.

        Args:
            condition: The condition string.
            model_id: The model family ID.
            variant: The model variant string.
            features: PromptFeatures object.
            prompt_char_count: Character count of the prompt.
            extra_flags: Additional boolean flags.

        Returns:
            True if the condition is satisfied.
        """
        profile = self.get_profile(model_id)
        sub_family = self.get_sub_family(model_id, variant) if variant else None

        # Determine model_version from variant string
        model_version: float | None = None
        if variant:
            # Try to extract version number from variant
            import re as _re
            m = _re.search(r"(\d+(?:\.\d+)?)", variant)
            if m:
                model_version = float(m.group(1))

        return self._condition_evaluator.evaluate(
            condition,
            model_variant=variant,
            model_version=model_version,
            features=features,
            prompt_char_count=prompt_char_count,
            extra_flags=extra_flags,
        )

    def get_sub_family_profile(
        self, model_id: str, variant: str | None
    ) -> SubFamilyProfile | None:
        """Get sub-family profile for a model variant. Convenience wrapper."""
        if variant is None:
            return None
        return self.get_sub_family(model_id, variant)
