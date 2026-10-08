# Findings: OpenAI Decisions API on LegalBench

*Plain-language explainer. Numbers from results/metrics.json. Run date 2026-10-08. Token counts include the 60-row pilot.*

## What we ran

We took 544 yes/no legal questions from LegalBench, a public benchmark whose
labels were written by lawyers. Six tasks: three contract-clause tasks from
CUAD (anti-assignment, non-compete, change of control), two legal-reasoning
tasks (hearsay, personal jurisdiction) and one consumer-contract Q&A task.
Four tasks had 100 items each, balanced 50/50; hearsay and personal
jurisdiction have fewer test items so we used all of them (94 and 50).

Each item went to OpenAI's Decisions API once, model gpt-6-luna, as one
"predicate" question: the clause or fact pattern as input, the task's own
question as the instruction. The API returns a single number, the probability
that the answer is yes. No prompt engineering, no examples, no retries on the
answer. One change from the plan: LegalBench's one-line hearsay prompt made the
model answer yes on almost every "no" item in the pilot, so for the full run we
gave it a four-sentence statement of the hearsay rule instead. Both wordings
are in data/prompts.json.

Cost: about 192,000 input tokens, $0.019. Wall time for 544 calls at
concurrency 8: about four minutes.

## What it found

| | value |
|---|---|
| Items answered | 523 of 544 (21 refusals, all hearsay) |
| Overall accuracy | 83.2% |
| AUROC | 0.894 |
| ECE (10 bins) | 0.103 |
| Brier | 0.140 |
| Accuracy at confidence >= 0.9 | 87.5% on 72.1% of items (47 wrong) |
| Accuracy at confidence >= 0.99 | 90.3% on 49.1% of items |
| Accuracy below 0.9 | 71.9% on 27.9% of items |
| Worst task | hearsay, 74.0% accuracy, AUROC 0.79 |

Per task:

| task | n | acc | AUROC | ECE | cov >= 0.9 | acc >= 0.9 |
|---|---|---|---|---|---|---|
| consumer_contracts_qa | 100 | 93.0% | 0.976 | 0.060 | 87% | 97.7% |
| cuad_non-compete | 100 | 88.0% | 0.949 | 0.086 | 83% | 90.4% |
| personal_jurisdiction | 50 | 84.0% | 0.913 | 0.115 | 16% | 100% |
| cuad_anti-assignment | 100 | 83.0% | 0.952 | 0.170 | 85% | 87.1% |
| cuad_change_of_control | 100 | 75.0% | 0.863 | 0.268 | 79% | 75.9% |
| hearsay | 73 | 74.0% | 0.786 | 0.129 | 48% | 80.0% |

Four things stand out.

**1. Confidence does not buy much accuracy.** Going from "all items" to "only
items the model is at least 90% sure about" raises accuracy from 83% to 88%,
and you give up 28% of the volume to get it. Even at 99% confidence, which
covers half the items, one in ten answers is still wrong. There is no threshold
in this data where the model's confident answers can be left unchecked.

**2. The wrong answers are confident "no"s.** The reliability diagram shows
where the errors live. 242 of 523 items got a predicted probability near zero.
17% of those were "yes" according to the lawyers. On the three contract tasks
the model's average probability on true "no" clauses was 0.01 to 0.08, and on
true "yes" clauses only 0.46 to 0.78. It misses clauses that are there far more
often than it invents ones that are not. For change-of-control clauses it
missed about half. In a review workflow this is the expensive direction: a
missed clause is a missed risk, and a confident miss never reaches a human.

**3. The model is better at knowing it is unsure on reasoning tasks than on
classification tasks.** On personal jurisdiction it only reached 90%
confidence on 16% of items, and every one of those was right. On the contract
tasks it was 90% confident on about 80% of items, and 10 to 24% of those were
wrong. The confidence number means something different depending on what you
ask it.

**4. Refusals are a silent failure mode.** 21 of 94 hearsay fact patterns, 22%,
came back as a refusal rather than an answer. Hearsay examples in a law exam
describe assaults, thefts and threats, which trips a content filter. 18 of the
21 refused items were "no" items. If your pipeline treats a refusal as "flag
for review" that is fine; if it treats it as "no" or drops it, the error rate
on that task roughly doubles. The refusals also mean the reported hearsay
accuracy is on the 73 easier-to-stomach items, not all 94.

Also worth noting: prompt wording moved the hearsay result from 1 right in 9
(LegalBench's one-liner) to 3 right in 8 on the same ten pilot items, and 74%
on the full run with the fuller rule. The API answers the question you wrote,
not the question you meant.

## What it means if you are deciding whether to put this in a review workflow

- As a first-pass filter that routes items to a human, it is useful and nearly
  free: $0.02 for 544 clauses, four minutes. Ranking quality is good (AUROC
  0.89 overall, 0.95 or better on three tasks).
- As an auto-decider at any confidence threshold, it is not ready for this
  kind of question. 12% error on the "confident" 72% is not a number a legal
  review team would sign off on, and the errors skew toward missing things.
- The usable pattern is asymmetric: treat a high-probability "yes" as a
  reliable flag (few false positives), and treat a confident "no" as
  "unverified", not "clear". Calibrate the threshold per task, not globally;
  the same 0.9 means 98% on consumer contracts and 76% on change of control.
- Count refusals as review items, and measure them. A task where a fifth of
  the inputs come back blank has a different cost model.

## What it does not prove

- One model, one day, one call per item, no examples in the prompt, no
  instructions beyond the task's own question. A tuned prompt or few-shot
  examples would likely move these numbers, possibly a lot (the hearsay
  rewrite shows how sensitive it is).
- 523 items across six tasks. Per-task numbers rest on 50 to 100 items, so a
  per-task accuracy has roughly a plus or minus 7 to 10 point margin.
- LegalBench items are short and self-contained. Real contract review means
  long documents, cross-references and defined terms. This test says nothing
  about that.
- The labels are the benchmark's. Lawyers disagree with each other too; we did
  not audit the items the model got "wrong".
- Nothing here measures the Responses API or any other model, so it does not
  say whether Decisions is better or worse than the alternatives, only whether
  its probability can be trusted on its own.
