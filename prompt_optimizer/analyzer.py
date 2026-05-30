"""
analyzer.py
Detects structural and stylistic features of a prompt via regex and heuristics.
No LLMs — deterministic analysis only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# Patterns for feature detection
_LLAMA2_TOKENS = re.compile(r"\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>")
_LLAMA3_TOKENS = re.compile(r"<\|im_start\||<\|im_end\||<\|begin_of_text\||<\|eot_id\|")
_CHATML_TOKENS = re.compile(r"<\|im_start\|>|<\|im_end\|>")
_ALPACA_HEADERS = re.compile(r"###\s*(Instruction|Input|Response)\s*:", re.IGNORECASE)
_XML_TAGS = re.compile(r"<[a-zA-Z][a-zA-Z0-9_\-]*>.*?</[a-zA-Z][a-zA-Z0-9_\-]*>", re.DOTALL)
_MARKDOWN_HEADERS = re.compile(r"^#{1,4}\s+\w", re.MULTILINE)
_BULLET_LIST = re.compile(r"^\s*[-*•]\s+\w", re.MULTILINE)
_NUMBERED_LIST = re.compile(r"^\s*\d+\.\s+\w", re.MULTILINE)
_BOLD_TEXT = re.compile(r"\*\*[^*]+\*\*")
_COT_PHRASES = re.compile(
    r"think\s+(step\s+by\s+step|carefully|through\s+this)|"
    r"let['']s\s+think|reason\s+step\s+by\s+step",
    re.IGNORECASE,
)
_LENGTH_INSTRUCTIONS = re.compile(
    r"\b(brief|concise|short|verbose|detailed|in \d+ (words|sentences|paragraphs)|"
    r"no more than|at most|under \d+)\b",
    re.IGNORECASE,
)
_ROLE_INSTRUCTION = re.compile(
    r"\b(you are|act as|your role is|you will serve as)\b", re.IGNORECASE
)


@dataclass
class PromptFeatures:
    """Detected features of the input prompt."""

    raw: str
    char_count: int = 0
    word_count: int = 0

    # Template/token markers
    has_llama2_tokens: bool = False
    has_llama3_tokens: bool = False
    has_chatml_tokens: bool = False
    has_alpaca_headers: bool = False
    has_xml_tags: bool = False

    # Formatting style
    has_markdown_headers: bool = False
    has_bullet_list: bool = False
    has_numbered_list: bool = False
    has_bold_text: bool = False

    # Instruction style
    has_cot_instruction: bool = False
    has_length_instruction: bool = False
    has_role_instruction: bool = False

    # Inferred nesting depth (rough heuristic)
    instruction_nesting_depth: int = 0

    # Misc
    detected_tokens: list[str] = field(default_factory=list)
    cot_phrases_found: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Length: {self.char_count} chars / {self.word_count} words",
            f"Special tokens: llama2={self.has_llama2_tokens}, llama3={self.has_llama3_tokens}, "
            f"chatml={self.has_chatml_tokens}, alpaca={self.has_alpaca_headers}",
            f"XML tags: {self.has_xml_tags}",
            f"Markdown: headers={self.has_markdown_headers}, bullets={self.has_bullet_list}, "
            f"bold={self.has_bold_text}",
            f"CoT instruction: {self.has_cot_instruction}",
            f"Length instruction: {self.has_length_instruction}",
            f"Role instruction: {self.has_role_instruction}",
            f"Nesting depth estimate: {self.instruction_nesting_depth}",
        ]
        return "\n".join(lines)


class PromptAnalyzer:
    """
    Analyzes a prompt string and returns a PromptFeatures object.
    All analysis is done with regex and heuristics — no LLM calls.
    """

    def analyze(self, prompt: str) -> PromptFeatures:
        features = PromptFeatures(raw=prompt)
        features.char_count = len(prompt)
        features.word_count = len(prompt.split())

        # Token detection
        features.has_llama2_tokens = bool(_LLAMA2_TOKENS.search(prompt))
        features.has_llama3_tokens = bool(_LLAMA3_TOKENS.search(prompt))
        features.has_chatml_tokens = bool(_CHATML_TOKENS.search(prompt))
        features.has_alpaca_headers = bool(_ALPACA_HEADERS.search(prompt))
        features.has_xml_tags = bool(_XML_TAGS.search(prompt))

        # Formatting
        features.has_markdown_headers = bool(_MARKDOWN_HEADERS.search(prompt))
        features.has_bullet_list = bool(_BULLET_LIST.search(prompt))
        features.has_numbered_list = bool(_NUMBERED_LIST.search(prompt))
        features.has_bold_text = bool(_BOLD_TEXT.search(prompt))

        # Instruction style
        cot_matches = _COT_PHRASES.findall(prompt)
        features.has_cot_instruction = bool(cot_matches)
        features.cot_phrases_found = cot_matches
        features.has_length_instruction = bool(_LENGTH_INSTRUCTIONS.search(prompt))
        features.has_role_instruction = bool(_ROLE_INSTRUCTION.search(prompt))

        # Rough nesting depth: count indentation levels in bullet/numbered lines
        indented_lines = [
            line for line in prompt.splitlines()
            if re.match(r"^\s{2,}[-*\d]", line)
        ]
        if indented_lines:
            max_indent = max(len(line) - len(line.lstrip()) for line in indented_lines)
            features.instruction_nesting_depth = max_indent // 2
        else:
            features.instruction_nesting_depth = 0

        return features
