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
        )


class RulesEngine:
    """
    Loads all model YAML profiles and provides lookup + rule evaluation.
    """

    def __init__(self, data_dir: Path | None = None):
        self._data_dir = data_dir or DATA_DIR
        self._profiles: dict[str, ModelProfile] = {}
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
