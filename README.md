# prompt-optimizer

> **Work in progress.** Core architecture is in place; contributions welcome.

A rule-based, **LLM-free** tool for adapting and linting prompts so they work as well as possible on a specific target model. The same agent system prompt that works beautifully on Claude can behave poorly on Llama or Mistral. Not because the intent is wrong, but because each model has different formatting expectations, special token requirements, and stylistic preferences. This tool makes those differences explicit and correctable.

---

## The problem

Every major LLM family has its own conventions:

| Model family | System prompt delivery | Special tokens | CoT benefit | Markdown |
|---|---|---|---|---|
| Claude | `system=` API field | None needed | Minimal on v3+ | Moderate |
| GPT / OpenAI | `{"role": "system"}` message | None needed | Minimal on reasoning models | Good |
| Llama 2 | Inline `[INST] <<SYS>>` tokens | Required | Yes | Limited |
| Llama 3 | Inline `<\|start_header_id\|>` tokens | Required | Yes | Limited |
| Mistral Instruct | Inline `[INST]` (no `<<SYS>>`) | Required | Yes | Moderate |
| Gemini | `system_instruction=` API field | None needed | Minimal on Pro/Ultra | Excellent |
| Qwen | ChatML `<\|im_start\|>` tokens | Required | Yes (QwQ) | Moderate |

Manually tracking and adapting for all of these is tedious, error-prone, and poorly documented in one place. `prompt-optimizer` encodes this knowledge as data and applies it automatically.

---

## How it works

No LLM is used at any stage. The pipeline is:

```
Input prompt
     │
     ▼
┌─────────────┐     Detects: special tokens, XML tags, markdown,
│  Analyzer   │     CoT phrases, length instructions, nesting depth
└─────────────┘
     │
     ▼
┌──────────────┐    Loads YAML profile for target model:
│ Rules Engine │    preferences, transformations, warnings
└──────────────┘
     │
     ▼
┌─────────────┐     Applies: token stripping, phrase injection/removal,
│ Transformer │     template wrapping, lint warnings
└─────────────┘
     │
     ▼
Adapted prompt + diff summary
```

Model knowledge lives in `data/models/*.yaml`: plain files that are easy to read, update, and contribute to.

---

## Sub-family architecture

Each model family can have **sub-families** — version groups with their own prompting behavior. For example, Claude 3 and Claude 4 have different XML tag expectations, and Llama 2 and Llama 3 use completely different chat templates.

```yaml
sub_families:
  claude-3:
    prompt_style: xml_block
    xml_tags:
      context: "<context>"
      instructions: "<instructions>"
    cot_policy: strip

  claude-4:
    prompt_style: agent_directives
    xml_tags:
      system_directives: "<system_directives>"
      documents: "<documents>"
    agent_mode: true
```

Use `--model-variant` to select a sub-family:

```bash
prompt-optimizer --input prompt.txt --target claude --model-variant claude-4-opus
prompt-optimizer --input prompt.txt --target llama --model-variant llama-3.1-405b
```

The rules engine matches the variant against sub-family keys and applies the correct transformations.

---

## Quickstart

```bash
pip install -e ".[dev]"
```

```bash
# List supported models
prompt-optimizer --list-models

# Adapt a prompt for Llama 3
prompt-optimizer --input my_agent_prompt.txt --target llama --show-diff

# Pipe directly
echo "You are a helpful assistant. Think step by step." | prompt-optimizer --target claude

# Analyze a prompt without adapting it
prompt-optimizer --input prompt.txt --analyze

# Write output to a file
prompt-optimizer --input prompt.txt --target mistral --output adapted.txt
```

---

## Python API

```python
from prompt_optimizer import PromptAnalyzer, RulesEngine, PromptTransformer

prompt = """
[INST] <<SYS>>
You are a customer support agent. Think step by step.
<</SYS>>

Answer the user's question. [/INST]
"""

engine = RulesEngine()
profile = engine.get_profile("claude")

features = PromptAnalyzer().analyze(prompt)
result = PromptTransformer().transform(prompt, profile, features)

print(result.adapted)
print(result.diff_summary())
```

---

## Supported models

| Model ID | Family | Vendor |
|---|---|---|
| `claude` | Claude 2 / 3 / 3.5 / 4 | Anthropic |
| `gpt` | GPT-3.5 / 4 / 4o / 4.5 / 5 / o1 / o3 | OpenAI |
| `llama` | Llama 2 / 3 / 3.1 / 3.2 / 3.3 / 4 | Meta |
| `mistral` | Mistral 7B / Small / Medium / Large / Mixtral / Ministral / Codestral / Devstral / Magistral | Mistral AI |
| `gemini` | Gemini 1.5 / 2.0 / 2.5 / Gemma 3 | Google DeepMind |
| `qwen` | Qwen2 / Qwen2.5 / Qwen3 / QwQ | Alibaba |
| `deepseek` | DeepSeek V2 / V3 / V4 / R1 | DeepSeek |
| `kimi` | Kimi K2 / K2.5 / K2.6 | Moonshot AI |
| `glm` | GLM-4 / GLM-4.6V / GLM-Z1 | Zhipu AI |
| `minimax` | MiniMax-Text / M1 / M2 / M2.1 | MiniMax |
| `grok` | Grok 3 / Grok 3 Mini | xAI |
| `phi` | Phi-4 / Phi-4-Mini | Microsoft |
| `cohere` | Command R+ / Command A | Cohere |

---

## Adding a new model

Create a file in `data/models/<model_id>.yaml`. The required fields are:

```yaml
model_id: my_model
display_name: My Model (family description)
vendor: Vendor Name
api_type: messages | chat_completions | varies

chat_template:
  system_delivery: api_field | messages_role | inline_tokens

sub_families:
  my-model-v1:
    prompt_style: default
    cot_policy: inject
    raw:
      description: "My Model v1"
      supports_reasoning: false

preferences:
  formatting: explicit | structured | declarative | moderate
  chain_of_thought: helpful | optional | not_recommended
  markdown: excellent | good | moderate | limited | poor
  verbosity: high | medium | medium-low | low

warnings:
  - Any known gotchas or pitfalls for this model

transformations:
  - id: my_transform
    type: regex_replace | phrase_remove | lint_warning | phrase_inject | chat_template_wrap | xml_to_headers | cot_policy | prompt_style | example_inject | xml_rewrite | doc_wrap
    description: What this does
    pattern: "regex here"       # for regex_replace
    replacement: ""             # for regex_replace

sources:
  - https://link-to-official-docs
```

Then open a PR. No code changes needed for adding model support.

---

## Transformation types

| Type | What it does |
|---|---|
| `regex_replace` | Find/replace via regex in the prompt text |
| `phrase_remove` | Remove a list of specific phrases |
| `phrase_inject` | Insert a phrase at a specified position |
| `chat_template_wrap` | Wrap the prompt in model-specific chat tokens |
| `xml_to_headers` | Convert `<tag>content</tag>` to `TAG:\ncontent` plain text |
| `lint_warning` | Check a condition and emit a warning without modifying the prompt |
| `cot_policy` | Apply sub-family CoT policy (strip/keep/inject) |
| `prompt_style` | Apply sub-family prompt style (xml_block, agent_directives, etc.) |
| `example_inject` | Insert `<example>` blocks for structured output |
| `xml_rewrite` | Rewrite text into specific XML tag structures |
| `doc_wrap` | Wrap source documents in tagged blocks |

---

## License

MIT
