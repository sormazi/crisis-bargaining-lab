import unittest
from experiment import parse_json, summarize

class ExperimentTests(unittest.TestCase):
    def test_parse_json_accepts_fenced_output(self):
        parsed = parse_json('```json\n{"action":"negotiate","opponent_concession_probability":0.4,"confidence":0.8,"rationale":"test"}\n```')
        self.assertEqual(parsed["action"], "negotiate")

    def test_parse_json_rejects_unknown_action(self):
        with self.assertRaises(ValueError):
            parse_json('{"action":"attack","opponent_concession_probability":0.4,"confidence":0.8,"rationale":"test"}')

    def test_summary_estimates_positive_effects(self):
        records = []
        actions = {"private_low": "negotiate", "public_low": "stand_firm", "private_high": "stand_firm", "public_high": "stand_firm"}
        for treatment, action in actions.items():
            records.append({"treatment": treatment, "format_valid": True, "action": action, "opponent_concession_probability": 0.5, "confidence": 0.8})
        _, effects = summarize(records)
        self.assertGreater(effects["audience_cost_effect"], 0)
        self.assertGreater(effects["costly_signal_effect"], 0)

if __name__ == "__main__": unittest.main()
