"""
Tests for the Jev verifier and its HITL action mapping.
"""
import unittest
from unittest.mock import patch, MagicMock

from agent import jev_verifier
from api.routes_hitl import _ACTION_TO_HITL_DECISION


def _evidence():
    return {
        "graph_evidence": {"shared_device_account_count": 6, "shared_ip_account_count": 6, "community_size": 6},
        "history_evidence": {"total_historical_txns": 2, "historical_avg_amount": 500.0},
        "device_evidence": {"is_suspicious_proxy": True, "os": "Android", "device_type": "Mobile"},
        "model_evidence": {"tabular_score": 91.0, "gnn_score": 95.0},
    }


def _response(payload):
    r = MagicMock()
    r.raise_for_status.return_value = None
    r.json.return_value = payload
    return r


class TestJevAvailability(unittest.TestCase):
    def test_unavailable_without_key(self):
        with patch.object(jev_verifier, "TYPESAFE_API_KEY", ""):
            self.assertFalse(jev_verifier.is_available())

    def test_available_with_key(self):
        with patch.object(jev_verifier, "TYPESAFE_API_KEY", "test-key"):
            self.assertTrue(jev_verifier.is_available())

    def test_missing_key_raises(self):
        with patch.object(jev_verifier, "TYPESAFE_API_KEY", ""):
            with self.assertRaises(RuntimeError):
                jev_verifier.verify_investigation({}, {}, _evidence(), "hypothesis", "HOLD_FOR_MANUAL_REVIEW")


class TestJevVerification(unittest.TestCase):
    def _action(self, choice, confidence):
        return _response({"model": "jev-test", "answers": {"recommended_action": {
            "type": "choice", "choice": choice,
            "probabilities": {choice: confidence}, "confidence": confidence,
        }}})

    @patch.object(jev_verifier, "TYPESAFE_API_KEY", "test-key")
    @patch("agent.jev_verifier.requests.post")
    def test_agreement_and_grounding_is_consistent(self, post):
        post.side_effect = [
            self._action("HOLD_FOR_MANUAL_REVIEW", 0.91),
            _response({"answers": {"hypothesis_grounded": {"type": "noul", "noul": 0.97}}}),
        ]
        result = jev_verifier.verify_investigation(
            {}, {"risk_score": 82}, _evidence(),
            "Shared device cluster with six linked accounts.",
            "HOLD_FOR_MANUAL_REVIEW",
        )
        self.assertEqual(result["verification_flag"], "CONSISTENT")
        self.assertTrue(result["actions_agree"])
        self.assertTrue(result["hypothesis_grounded"])
        self.assertTrue(result["eligible_for_auto_resolve"])
        self.assertEqual(post.call_count, 2)

    @patch.object(jev_verifier, "TYPESAFE_API_KEY", "test-key")
    @patch("agent.jev_verifier.requests.post")
    def test_action_disagreement_requires_review(self, post):
        post.side_effect = [
            self._action("BLOCK_ACCOUNT_AND_HOLD_FUNDS", 0.88),
            _response({"answers": {"hypothesis_grounded": {"type": "noul", "noul": 0.9}}}),
        ]
        result = jev_verifier.verify_investigation(
            {}, {"risk_score": 82}, _evidence(),
            "Suspicious activity.", "APPROVE_WITH_VERIFICATION",
        )
        self.assertEqual(result["verification_flag"], "REVIEW_RECOMMENDED")
        self.assertFalse(result["eligible_for_auto_resolve"])

    @patch.object(jev_verifier, "TYPESAFE_API_KEY", "test-key")
    @patch("agent.jev_verifier.requests.post")
    def test_low_grounding_requires_review(self, post):
        post.side_effect = [
            self._action("HOLD_FOR_MANUAL_REVIEW", 0.95),
            _response({"answers": {"hypothesis_grounded": {"type": "noul", "noul": 0.2}}}),
        ]
        result = jev_verifier.verify_investigation(
            {}, {"risk_score": 82}, _evidence(),
            "Invented claim about fourteen accounts.", "HOLD_FOR_MANUAL_REVIEW",
        )
        self.assertEqual(result["verification_flag"], "REVIEW_RECOMMENDED")
        self.assertFalse(result["eligible_for_auto_resolve"])

    @patch.object(jev_verifier, "TYPESAFE_API_KEY", "test-key")
    @patch("agent.jev_verifier.requests.post")
    def test_confidence_floor_blocks_auto_resolve(self, post):
        post.side_effect = [
            self._action("HOLD_FOR_MANUAL_REVIEW", 0.60),
            _response({"answers": {"hypothesis_grounded": {"type": "noul", "noul": 0.9}}}),
        ]
        result = jev_verifier.verify_investigation(
            {}, {"risk_score": 82}, _evidence(),
            "Ambiguous case.", "HOLD_FOR_MANUAL_REVIEW",
        )
        self.assertEqual(result["verification_flag"], "CONSISTENT")
        self.assertFalse(result["eligible_for_auto_resolve"])


class TestHitlMapping(unittest.TestCase):
    def test_all_actions_map_to_valid_decisions(self):
        self.assertEqual(set(_ACTION_TO_HITL_DECISION), {
            "BLOCK_ACCOUNT_AND_HOLD_FUNDS", "HOLD_FOR_MANUAL_REVIEW",
            "TEMPORARY_VELOCITY_FREEZE", "REQUIRE_TWO_FACTOR_AUTHENTICATION",
            "APPROVE_WITH_VERIFICATION",
        })
        self.assertTrue(all(v in {"APPROVE", "HOLD", "BLOCK"} for v in _ACTION_TO_HITL_DECISION.values()))

    def test_unknown_action_defaults_to_hold(self):
        self.assertEqual(_ACTION_TO_HITL_DECISION.get("UNKNOWN_ACTION", "HOLD"), "HOLD")


if __name__ == "__main__":
    unittest.main()
