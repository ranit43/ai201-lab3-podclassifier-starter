# Classifier Spec — Pod Classifier

Complete this spec **before** writing any code for Milestone 2.

Use Plan or Ask mode to think through each blank field. When you're done,
your answers here become the blueprint for `build_few_shot_prompt()` and
`classify_episode()` in `classifier.py`.

---

## build_few_shot_prompt(labeled_examples, description)

### What it does
Constructs a prompt string for the LLM that includes the task instructions,
all labeled training examples, and the new episode description to classify.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `labeled_examples` | `list[dict]` | Each dict has `"title"`, `"description"`, `"label"` (and others). These are the examples you labeled in Milestone 1. |
| `description` | `str` | The episode description to classify. |

### Output

| Return value | Type | Description |
|---|---|---|
| prompt | `str` | A complete prompt string ready to send to the LLM. |

---

### Spec fields — fill these in before writing code

**Task instruction (what should the LLM know about the task?):**

```
You are classifying podcast episodes by their format. Classify the episode
into exactly one of these four labels:

- interview: a conversation between a host and one or more guests
- solo: a single host speaking from memory, experience, or opinion — no guests,
  no assembled external sources
- panel: multiple guests with roughly equal speaking time, often debating or
  discussing a topic together
- narrative: a story assembled from external sources — interviews, archival
  audio, reporting — with a clear narrative arc

Return only the label and your reasoning. Do not explain the taxonomy.
```

---

**How should labeled examples be formatted in the prompt?**

```
Each example should include the episode title, a brief excerpt or the full
description, and the correct label. Separate examples with a blank line or
a delimiter like "---". Include all fields that help the model see why the
label was applied — title and description are both useful; other fields
(like episode ID) are not needed.
```

---

**Example block sketch (write one concrete example):**

```
Title: {title}
Description: {description}
Label: {label}
```

---

**How should the new episode (to be classified) be presented?**

```
Present it in the same format as the labeled examples, but omit the Label
line and replace it with an instruction to classify. For example:

Title: {title}
Description: {description}
Label: ?

Then add a line like: "Classify the episode above. Return your answer in
the format below:" followed by the output format you chose.
```

---

**What output format should you request from the LLM?**

```
Request a compact JSON object with exactly two keys:

{
  "label": "interview",
  "reasoning": "The description presents a host asking one expert guest about their work."
}

JSON gives named fields, so parsing does not depend on line order. It also
supports both required outputs: one fixed label and a brief explanation. The
tradeoff is that the LLM may wrap the JSON in markdown fences or add a sentence
around it, so classify_episode() should strip code fences and, if needed,
extract the first JSON-looking object before parsing.

Do not use Groq's native JSON mode for this milestone. The lab is practicing
robust parsing of ordinary LLM text, so the prompt should request JSON while the
code handles messy real-world output.
```

---

**Edge cases to handle in the prompt:**

```
If labeled_examples is empty, still produce a valid prompt with the taxonomy and
an explicit note that no labeled examples were provided. The normal UI blocks
this case, but the function should not return an empty prompt.

If the description is very short, tell the model to classify from available
format cues and keep the reasoning cautious. The prompt should emphasize that
labels are based on episode structure, not subject matter, tone, or marketing
language.

Include all labeled examples because these 20 examples are the full few-shot
training signal for the lab.
```

---

## classify_episode(description, labeled_examples)

### What it does
Classifies a single podcast episode description using the few-shot LLM classifier.
Returns a dict with a label and reasoning.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `description` | `str` | The episode description to classify. |
| `labeled_examples` | `list[dict]` | Labeled training examples from `load_labeled_examples()`. |

### Output

| Return value | Type | Description |
|---|---|---|
| result | `dict` | Must have keys `"label"` and `"reasoning"`. `"label"` must be one of `VALID_LABELS` or `"unknown"`. |

---

### Spec fields — fill these in before writing code

**Step 1 — Build the prompt:**

```
Call build_few_shot_prompt(labeled_examples, description) and store the
returned string in a variable (e.g., prompt). Pass through both arguments
exactly as received — no modification needed before calling.
```

---

**Step 2 — Send to the LLM:**

```
Call _client.chat.completions.create() with:
  - model: the model name from config (LLM_MODEL)
  - messages: a list with one dict — {"role": "user", "content": prompt}
    (system-design.md shows an optional system message too — either shape works)
  - max_tokens: a reasonable limit (e.g., 200–300) to keep responses concise

Extract the response text from:
  response.choices[0].message.content
```

---

**Step 3 — Parse the response:**

```
Read response.choices[0].message.content and strip surrounding whitespace.
First try to parse it as JSON. Before parsing, remove common markdown code
fences such as ```json ... ```. If direct JSON parsing fails, find the first
substring from "{" through the matching-looking final "}" and try json.loads()
on that substring.

If JSON parsing still fails, use a text fallback: scan for lines like
"Label: interview" and "Reasoning: ...". If no labeled line is present, use the
first non-empty line as a candidate label and keep the full response as the
reasoning.
```

---

**Step 4 — Validate the label:**

```
Normalize the candidate label before validation: convert it to a string, strip
whitespace, remove wrapping punctuation/quotes/asterisks/backticks, and
lowercase it. If the normalized label is one of VALID_LABELS, return it.
Otherwise set the label to "unknown" and keep a reasoning string explaining that
the response did not contain a valid label.
```

---

**Step 5 — Handle errors gracefully:**

```
The API call can fail because of a missing key, network error, rate limit, or
unexpected response shape. Parsing can fail if the LLM ignores the requested
format. Wrap the LLM call and parsing in error handling so evaluation does not
crash on one bad call.

On failure, return:

{
    "label": "unknown",
    "reasoning": "Classification failed: <error>"
}
```

---

### Return value structure

```python
{
    "label": str,      # one of VALID_LABELS, or "unknown" if invalid/error
    "reasoning": str,  # brief explanation from the LLM
}
```

---

## Notes on label quality

The classifier is only as good as your labels. If your training examples have
inconsistent or ambiguous labels, the LLM will learn the wrong pattern.

Before implementing the classifier, re-read `data/taxonomy.md` and double-check
any labels you're unsure about. Annotation quality is part of the lab.

---

## Implementation Notes

*Fill this in after implementing and testing both functions.*

**Test: what does the raw LLM response look like for one episode?**

```
Episode tested: The Aral Sea: A Disaster in Four Acts
Raw response text: {"label": "narrative", "reasoning": "The episode describes a story told in multiple parts, with a clear structure and an attempt to convey a story arc, suggesting a narrative format."}
```

**How did you parse the label out of the response?**

```
Strip surrounding whitespace and markdown code fences, then try json.loads().
If that fails, extract the first JSON-looking {...} substring and parse that.
If JSON still fails, look for Label: and Reasoning: lines. Normalize the
candidate label by stripping quotes, punctuation, markdown markers, and
lowercasing before checking VALID_LABELS.
```

**Did any episodes return `"unknown"`? If so, why?**

```
No for the live UI checkpoint examples. Error-path testing confirmed that API
or connection failures return "unknown" with an explanatory reason instead of
crashing the caller.
```

**One thing about the output format that surprised you:**

```
For the tested response, the model followed the JSON instruction exactly with
no markdown fence or extra prose. The fallback parser is still useful because
the lab warns that later responses may vary across calls.
```
