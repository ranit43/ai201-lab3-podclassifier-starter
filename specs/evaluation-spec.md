# Evaluation Spec — Pod Classifier

Complete this spec **before** writing any code for Milestone 3.

Use Plan or Ask mode to think through each blank field. When you're done,
your answers here become the blueprint for `compute_accuracy()` and
`compute_per_class_accuracy()` in `evaluate.py`.

---

## Background: What is evaluation?

After building a classifier, we need to know how well it works. Evaluation answers:
- **Overall:** What fraction of episodes did we classify correctly?
- **Per-class:** Are we better at some labels than others?

Both functions take the same inputs: a list of predicted labels and a list of
ground-truth labels, in the same order.

---

## compute_accuracy(predictions, ground_truth)

### What it does
Returns the fraction of predictions that exactly match the ground truth.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `predictions` | `list[str]` | Labels predicted by `classify_episode()`, one per episode. |
| `ground_truth` | `list[str]` | The correct labels, in the same order as `predictions`. |

### Output

| Return value | Type | Description |
|---|---|---|
| accuracy | `float` | A value between 0.0 and 1.0. |

---

### Spec fields — fill these in before writing code

**Formula:**

```
accuracy = number of exact prediction/ground-truth matches / number of
ground-truth examples

A prediction counts as correct only when the predicted label exactly equals the
ground-truth label at the same list position.
```

---

**Step-by-step logic:**

```
1. If there are no ground-truth examples, return 0.0.
2. Pair each prediction with the matching ground-truth label using list order.
3. Count the number of pairs where predicted == truth.
4. Divide the correct count by the number of ground-truth examples.
5. Return the resulting float.
```

---

**Edge case — what if both lists are empty?**

```
Return 0.0. With no examples, there is no measured accuracy, and reporting
100% would be misleading.
```

---

**Worked example:**

```
predictions  = ["interview", "solo", "panel", "interview"]
ground_truth = ["interview", "solo", "solo",  "narrative"]

Correct:
- interview == interview
- solo == solo

Incorrect:
- panel != solo
- interview != narrative

Result: 2 correct / 4 examples = 0.5
```

---

## compute_per_class_accuracy(predictions, ground_truth)

### What it does
Returns accuracy broken down by each label. For each label in `VALID_LABELS`,
reports how many episodes with that ground-truth label were classified correctly.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `predictions` | `list[str]` | Labels predicted by `classify_episode()`. |
| `ground_truth` | `list[str]` | Correct labels, in the same order. |

### Output

A `dict` keyed by label. Each value is a dict with three keys:

```python
{
    "interview": {"correct": int, "total": int, "accuracy": float},
    "solo":      {"correct": int, "total": int, "accuracy": float},
    "panel":     {"correct": int, "total": int, "accuracy": float},
    "narrative": {"correct": int, "total": int, "accuracy": float},
}
```

---

### Spec fields — fill these in before writing code

**What does "correct" mean for a given class?**

```
An episode counts as correct for a class when its ground-truth label is that
class and the predicted label exactly matches that same class.

For example, an interview episode is correct only when truth == "interview" and
predicted == "interview".
```

---

**What does "total" mean for a given class?**

```
For a class, total is the number of examples whose ground-truth label is that
class. It is not the total number of predictions across the whole test set.
```

---

**Step-by-step logic:**

```
1. Initialize one stats dictionary for each label in VALID_LABELS with correct
   = 0, total = 0, and accuracy = 0.0.
2. Loop over prediction/ground-truth pairs in list order.
3. For each pair, ignore it if the ground-truth label is not in VALID_LABELS.
4. Otherwise, increment total for the ground-truth class.
5. If predicted == truth, increment correct for that same ground-truth class.
6. After the loop, compute accuracy = correct / total for each class with at
   least one example.
7. Leave accuracy as 0.0 for classes with no examples.
8. Return the stats dictionary.
```

---

**Edge case — what if a class has no examples in ground_truth (total == 0)?**

```
Set accuracy to 0.0. There are no examples for that class, so there is no
measured accuracy to report.
```

---

**Worked example:**

```
predictions  = ["interview", "interview", "solo", "panel", "panel"]
ground_truth = ["interview", "solo",      "solo", "panel", "narrative"]

label       correct  total  accuracy
----------  -------  -----  --------
interview   1        1      1.0
solo        1        2      0.5
panel       1        1      1.0
narrative   0        1      0.0
```

---

## Reflection questions (discuss at the checkpoint)

1. Your overall accuracy might be decent even if one class has very low accuracy.
   Why is per-class accuracy a more informative metric than overall accuracy alone?

2. If `panel` episodes consistently get misclassified as `interview`, what does
   that tell you about your training labels or your prompt?

3. You labeled 20 training episodes and evaluated on 20 test episodes (5 per class).
   How might the evaluation results change if you had labeled 100 training episodes?
   What if you had 200 test episodes?
