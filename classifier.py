import json
import os
import re
from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL, VALID_LABELS, DATA_PATH, TRAIN_FILE, LABELS_FILE

_client = Groq(api_key=GROQ_API_KEY)


def load_labeled_examples() -> list[dict]:
    """
    Load the training episodes and merge them with the student's labels.

    Returns a list of dicts, each with:
      - "id"          : episode ID
      - "title"       : episode title
      - "podcast"     : podcast name
      - "description" : episode description
      - "label"       : the label from my_labels.json (may be None if not yet annotated)

    Only returns episodes where the label is a valid, non-null string.
    Episodes with null labels are silently skipped.
    """
    train_path = os.path.join(DATA_PATH, TRAIN_FILE)
    labels_path = os.path.join(DATA_PATH, LABELS_FILE)

    with open(train_path, encoding="utf-8") as f:
        episodes = {ep["id"]: ep for ep in json.load(f)}

    with open(labels_path, encoding="utf-8") as f:
        labels = {entry["id"]: entry["label"] for entry in json.load(f)}

    labeled = []
    for ep_id, ep in episodes.items():
        label = labels.get(ep_id)
        if label in VALID_LABELS:
            labeled.append({**ep, "label": label})

    return labeled


def build_few_shot_prompt(labeled_examples: list[dict], description: str) -> str:
    """
    Build a few-shot classification prompt using the student's labeled training examples.

    TODO — Milestone 2:

    Your prompt needs to:
      1. Describe the task and the four valid labels
      2. Show the labeled training examples so the LLM can learn the pattern
      3. Present the new description and ask for a classification

    The LLM should return a single label from VALID_LABELS (exactly as written)
    plus a brief explanation of its reasoning. Think carefully about the output
    format you request — you'll need to parse it in classify_episode().

    Before writing code, complete specs/classifier-spec.md.
    """
    valid_labels = ", ".join(VALID_LABELS)
    lines = [
        "You are classifying podcast episodes by format.",
        "",
        "Choose exactly one of these lowercase labels:",
        "- interview: a conversation between a host and one or more guests; the guest's knowledge or experience drives the episode.",
        "- solo: a single host speaking from memory, experience, or opinion; no guests and no assembled external sources as structural elements.",
        "- panel: multiple guests with roughly equal speaking time, often debating or discussing a topic together.",
        "- narrative: reported or documentary storytelling assembled from sources such as interviews, archival audio, or reporting, with a clear story arc.",
        "",
        "Classify by episode structure, not by topic, tone, title style, or marketing language.",
        f"The label must be one of: {valid_labels}.",
        "",
        "Labeled examples:",
    ]

    if labeled_examples:
        for index, example in enumerate(labeled_examples, start=1):
            title = str(example.get("title", "")).strip() or "Untitled"
            example_description = str(example.get("description", "")).strip()
            label = str(example.get("label", "")).strip().lower()
            lines.extend(
                [
                    "",
                    f"Example {index}",
                    f"Title: {title}",
                    f"Description: {example_description}",
                    f"Label: {label}",
                ]
            )
    else:
        lines.extend(
            [
                "",
                "No labeled examples were provided. Use the taxonomy above and classify cautiously from the available format cues.",
            ]
        )

    lines.extend(
        [
            "",
            "New episode",
            f"Description: {str(description).strip()}",
            "",
            "Return only a JSON object with exactly these two keys:",
            '{"label": "interview", "reasoning": "Brief explanation based on episode structure."}',
            "",
            "Use one lowercase valid label. Keep reasoning to one sentence. If the description is short, classify from the available structural cues and keep the reasoning cautious.",
        ]
    )

    return "\n".join(lines)


def _strip_markdown_fences(response_text: str) -> str:
    """
    Remove a single surrounding markdown code fence from an LLM response.

    Input:
      - response_text: raw text returned by the LLM

    Output:
      - response text without wrapping ``` or ```json fences, if present
    """
    text = response_text.strip()
    fence_match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()
    return text


def _normalize_label(label: object) -> str:
    """
    Convert a parsed label candidate into the format used by VALID_LABELS.

    Input:
      - label: any value extracted from the LLM response as the label candidate

    Output:
      - lowercase string with surrounding whitespace, punctuation, quotes, and
        markdown markers removed
    """
    normalized = str(label or "").strip().lower()
    normalized = normalized.strip(" \t\r\n`*_\"'.,:;()[]{}")
    return normalized


def _parse_classifier_response(response_text: str) -> dict:
    """
    Extract a label and reasoning string from the raw LLM response.

    Input:
      - response_text: raw text returned by the LLM

    Output:
      - dict with "label" and "reasoning" keys; the label is normalized, but
        validation against VALID_LABELS happens in classify_episode()
    """
    text = _strip_markdown_fences(response_text)
    parsed = None

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                parsed = None

    if isinstance(parsed, dict):
        return {
            "label": _normalize_label(parsed.get("label", "")),
            "reasoning": str(parsed.get("reasoning", "")).strip(),
        }

    label = ""
    reasoning = ""
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        if not separator:
            continue

        normalized_key = key.strip().lower().strip("*` ")
        if normalized_key == "label":
            label = value.strip()
        elif normalized_key in {"reasoning", "reason", "explanation"}:
            reasoning = value.strip()

    if not label:
        non_empty_lines = [line.strip() for line in text.splitlines() if line.strip()]
        if non_empty_lines:
            label = non_empty_lines[0]

    return {
        "label": _normalize_label(label),
        "reasoning": reasoning or text,
    }


def classify_episode(description: str, labeled_examples: list[dict]) -> dict:
    """
    Classify a single podcast episode description using the few-shot LLM classifier.

    TODO — Milestone 2 (complete after build_few_shot_prompt):

    Steps:
      1. Call build_few_shot_prompt() to construct the prompt
      2. Send it to the LLM via _client.chat.completions.create()
      3. Parse the response to extract a label and reasoning
      4. Validate the label — if it's not in VALID_LABELS, set it to "unknown"
      5. Return a dict with "label" and "reasoning" keys

    Handle the case where the LLM returns something unparseable gracefully —
    don't let a bad response crash the whole evaluation.

    Before writing code, complete specs/classifier-spec.md.
    """
    try:
        prompt = build_few_shot_prompt(labeled_examples, description)
        response = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )
        response_text = response.choices[0].message.content or ""

        if os.getenv("PODCLASSIFIER_DEBUG_RAW"):
            print("\n--- Raw classifier response ---")
            print(response_text)
            print("--- End raw classifier response ---\n")

        result = _parse_classifier_response(response_text)
        label = result["label"]
        reasoning = result["reasoning"].strip() or "No reasoning provided."

        if label not in VALID_LABELS:
            return {
                "label": "unknown",
                "reasoning": f"LLM response did not contain a valid label. Raw response: {response_text}",
            }

        return {
            "label": label,
            "reasoning": reasoning,
        }
    except Exception as error:
        return {
            "label": "unknown",
            "reasoning": f"Classification failed: {error}",
        }
