# Crisis Bargaining Lab


The lab tests whether an LLM changes its crisis advice when a fictional state’s threat becomes more credible. It uses a 2×2 treatment design:

| | Low mobilization cost | High mobilization cost |
|---|---|---|
| **Private threat** | cheap/private | costly/private |
| **Public threat** | cheap/public | costly/public |

For every treatment, the model advises a fictional state to `back_down`, `negotiate`, or `stand_firm`, estimates the chance that its rival will concede, and reports confidence.

## Research question

Do LLMs reason in ways consistent with two classic mechanisms in crisis bargaining?

- **Audience costs:** public commitments can make backing down more costly.
- **Costly signaling:** visible military mobilization can make resolve more credible.

The experiment does not assume that escalation is wise or morally desirable. It measures comparative behavior: whether changing one theoretically relevant variable changes the model’s recommendation while everything else stays fixed.

## Hypotheses

- **H1 — Audience-cost effect:** public threats produce a higher `stand_firm` rate than private threats.
- **H2 — Costly-signal effect:** high-cost mobilization produces a higher `stand_firm` rate than low-cost mobilization.
- **H3 — Credibility forecast:** the model predicts a higher probability of rival concession when threats are both public and costly.

Treat these hypotheses as pre-registered before running a new model.

## Quick start

Python 3.9+ is enough; there are no package dependencies.

```bash
python3 experiment.py --provider demo
```

The `demo` provider is a deterministic pipeline check, not an LLM result. It creates all output files without credentials.

To evaluate a real model through an OpenAI-compatible endpoint:

```bash
export LLM_API_KEY="your-key"
python3 experiment.py \
  --provider openai-compatible \
  --base-url https://api.openai.com/v1 \
  --model YOUR_MODEL \
  --repetitions 5
```

## Experimental design

[`data/scenarios.json`](data/scenarios.json) contains 12 fictional interstate crises spanning maritime access, border demarcation, water, airspace, trade routes, and resource disputes. Fictional actors reduce contamination from memorized facts and current political preferences.

Each base scenario is rendered four times. Only two passages change:

1. whether the threat was made privately or publicly; and
2. whether mobilization is low-cost and reversible or costly and difficult to fake.

Job order is randomized using `--seed`. Use `--repetitions` to measure within-model variation when temperature is above zero.

The system asks for strict JSON:

```json
{"action":"negotiate","opponent_concession_probability":0.55,"confidence":0.78,"rationale":"brief strategic explanation"}
```

## Outputs

- `results/responses.jsonl` — raw prompt, response, treatment, and parsed fields
- `results/summary.json` — treatment-level rates and estimated effects
- `results/report.md` — readable results and hypothesis checks

The primary effects are differences in `stand_firm` rates. The report also shows negotiation and back-down rates, mean predicted concession probability, confidence, and JSON validity.

## Interpretation

A positive audience-cost or costly-signal effect is consistent with the theory, but it does not prove that the model has an internal game-theoretic representation. It may be pattern-matching language associated with resolve. Conversely, a null result may reflect ambiguity, safety tuning, or low statistical power.

This is a behavioral probe—not policy advice, a conflict simulator, or evidence that any recommended action would be effective in the real world.

## Extensions

- Compare base and instruction-tuned models.
- Add democratic vs. autocratic audience-cost treatments.
- Test finite versus indefinite repeated interaction.
- Add asymmetric stakes and privately known resolve.
- Have a second model classify rationales, while keeping action choice as the primary outcome.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

## License

MIT
