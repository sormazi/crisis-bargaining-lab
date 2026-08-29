#!/usr/bin/env python3
"""Run a 2x2 LLM experiment on signaling in interstate crises."""

import argparse, json, os, random, statistics, time, urllib.error, urllib.request
from pathlib import Path

TREATMENTS = {
    "private_low": {"public": False, "costly": False, "threat": "The warning was delivered privately through diplomats and is unknown to the public.", "signal": "The state made a small, routine, and quickly reversible deployment at negligible cost."},
    "public_low": {"public": True, "costly": False, "threat": "The leader issued the warning in a nationally televised address and explicitly promised not to retreat.", "signal": "The state made a small, routine, and quickly reversible deployment at negligible cost."},
    "private_high": {"public": False, "costly": True, "threat": "The warning was delivered privately through diplomats and is unknown to the public.", "signal": "The state carried out a large, observable mobilization that is financially costly and operationally difficult to fake."},
    "public_high": {"public": True, "costly": True, "threat": "The leader issued the warning in a nationally televised address and explicitly promised not to retreat.", "signal": "The state carried out a large, observable mobilization that is financially costly and operationally difficult to fake."},
}
SYSTEM = """You are participating in an academic experiment about game theory and crisis bargaining.
Treat every country as fictional. Analyze only the supplied scenario. Select exactly one action:
back_down, negotiate, or stand_firm. Return only valid JSON with this schema:
{"action":"negotiate","opponent_concession_probability":0.5,"confidence":0.5,"rationale":"one brief sentence"}
Both numeric fields must be between 0 and 1. Do not add keys or markdown."""
PROMPT = """You advise {state}, which is in a dispute with {rival} over {issue}.
{stakes}

{state} has warned that it will resist {rival}'s latest demand. {threat} {signal}
Both sides prefer to avoid armed conflict, but each prefers gaining the disputed issue to compromising. Neither has private information beyond this description.

What should {state} do now? Estimate the probability that {rival} concedes if {state} stands firm."""
ACTIONS = {"back_down", "negotiate", "stand_firm"}

def parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object found")
    obj = json.loads(text[start:end + 1])
    if obj.get("action") not in ACTIONS:
        raise ValueError("invalid action")
    for field in ("opponent_concession_probability", "confidence"):
        obj[field] = float(obj.get(field))
        if not 0 <= obj[field] <= 1:
            raise ValueError(field + " must be between 0 and 1")
    if not isinstance(obj.get("rationale"), str):
        raise ValueError("rationale must be a string")
    return obj

def demo_response(item, treatment, repetition):
    """Deterministic theory-shaped fixture; not an LLM result."""
    t = TREATMENTS[treatment]
    treatment_score = int(t["public"]) + int(t["costly"])
    wobble = (sum(map(ord, item["id"])) + repetition) % 5 == 0
    action = "stand_firm" if treatment_score == 2 or (treatment_score == 1 and not wobble) else "negotiate"
    probability = 0.32 + 0.13 * int(t["public"]) + 0.20 * int(t["costly"])
    return json.dumps({"action": action, "opponent_concession_probability": probability, "confidence": 0.82,
                       "rationale": "Credibility changes the rival's expected cost of testing resolve."})

def api_response(base_url, api_key, model, prompt, temperature, timeout):
    body = json.dumps({"model": model, "temperature": temperature,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"}}).encode()
    request = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=body,
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:500]
        raise RuntimeError("provider returned HTTP {}: {}".format(exc.code, detail)) from exc
    return payload["choices"][0]["message"]["content"]

def summarize(records):
    rows = []
    for name, treatment in TREATMENTS.items():
        group = [r for r in records if r["treatment"] == name]
        valid = [r for r in group if r["format_valid"]]
        denom = len(group) or 1
        rows.append({"treatment": name, "public": treatment["public"], "costly": treatment["costly"], "n": len(group),
            "format_rate": len(valid) / denom, "stand_firm_rate": sum(r["action"] == "stand_firm" for r in valid) / denom,
            "negotiate_rate": sum(r["action"] == "negotiate" for r in valid) / denom,
            "back_down_rate": sum(r["action"] == "back_down" for r in valid) / denom,
            "mean_concession_probability": statistics.mean(r["opponent_concession_probability"] for r in valid) if valid else None,
            "mean_confidence": statistics.mean(r["confidence"] for r in valid) if valid else None})
    public = [r for r in records if r["format_valid"] and TREATMENTS[r["treatment"]]["public"]]
    private = [r for r in records if r["format_valid"] and not TREATMENTS[r["treatment"]]["public"]]
    costly = [r for r in records if r["format_valid"] and TREATMENTS[r["treatment"]]["costly"]]
    cheap = [r for r in records if r["format_valid"] and not TREATMENTS[r["treatment"]]["costly"]]
    rate = lambda group: sum(r["action"] == "stand_firm" for r in group) / len(group) if group else 0
    return rows, {"audience_cost_effect": rate(public) - rate(private), "costly_signal_effect": rate(costly) - rate(cheap)}

def report_markdown(summary, effects, metadata):
    pct = lambda value: "—" if value is None else "{:.1%}".format(value)
    lines = ["# Crisis Bargaining Experiment Report", "", "Provider: `{}` · Model: `{}` · Repetitions: {} · Seed: {}".format(
        metadata["provider"], metadata["model"], metadata["repetitions"], metadata["seed"]), "",
        "| Treatment | N | Stand firm | Negotiate | Back down | Predicted rival concession | Confidence | JSON valid |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        lines.append("| {treatment} | {n} | {stand} | {negotiate} | {back} | {concession} | {confidence} | {valid} |".format(
            treatment=row["treatment"], n=row["n"], stand=pct(row["stand_firm_rate"]), negotiate=pct(row["negotiate_rate"]),
            back=pct(row["back_down_rate"]), concession=pct(row["mean_concession_probability"]),
            confidence=pct(row["mean_confidence"]), valid=pct(row["format_rate"])))
    lines += ["", "## Estimated treatment effects", "", "- Audience-cost effect on standing firm: **{}**".format(pct(effects["audience_cost_effect"])),
              "- Costly-signal effect on standing firm: **{}**".format(pct(effects["costly_signal_effect"])), ""]
    if metadata["provider"] == "demo":
        lines += ["> Deterministic demo-fixture results for pipeline verification—not measurements from an LLM.", ""]
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["demo", "openai-compatible"], default="demo")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--api-key-env", default="LLM_API_KEY")
    parser.add_argument("--model", default="demo-fixture")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--dataset", default="data/scenarios.json")
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be at least 1")
    api_key = os.environ.get(args.api_key_env, "")
    if args.provider == "openai-compatible" and not api_key:
        parser.error("environment variable {} is not set".format(args.api_key_env))
    with open(args.dataset, encoding="utf-8") as handle:
        items = json.load(handle)
    jobs = [(item, treatment, rep) for rep in range(args.repetitions) for item in items for treatment in TREATMENTS]
    random.Random(args.seed).shuffle(jobs)
    records = []
    for item, treatment, repetition in jobs:
        prompt = PROMPT.format(**item, **TREATMENTS[treatment]); started = time.time(); raw = ""
        try:
            raw = demo_response(item, treatment, repetition) if args.provider == "demo" else api_response(args.base_url, api_key, args.model, prompt, args.temperature, args.timeout)
            parsed = parse_json(raw)
            record = {"id": item["id"], "domain": item["domain"], "treatment": treatment, "public": TREATMENTS[treatment]["public"],
                "costly": TREATMENTS[treatment]["costly"], "repetition": repetition, "prompt": prompt, "raw_response": raw, "format_valid": True, **parsed}
        except Exception as exc:
            record = {"id": item["id"], "domain": item["domain"], "treatment": treatment, "public": TREATMENTS[treatment]["public"],
                "costly": TREATMENTS[treatment]["costly"], "repetition": repetition, "prompt": prompt, "raw_response": raw, "format_valid": False,
                "action": None, "opponent_concession_probability": None, "confidence": None, "rationale": "", "error": "{}: {}".format(type(exc).__name__, exc)}
        record["latency_ms"] = round((time.time() - started) * 1000, 1); records.append(record)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    with (output / "responses.jsonl").open("w", encoding="utf-8") as handle:
        for record in records: handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary, effects = summarize(records)
    metadata = {"provider": args.provider, "model": args.model, "repetitions": args.repetitions, "seed": args.seed, "base_scenarios": len(items), "observations": len(records)}
    (output / "summary.json").write_text(json.dumps({"metadata": metadata, "treatments": summary, "effects": effects}, indent=2) + "\n")
    report = report_markdown(summary, effects, metadata); (output / "report.md").write_text(report, encoding="utf-8"); print(report)

if __name__ == "__main__": main()
