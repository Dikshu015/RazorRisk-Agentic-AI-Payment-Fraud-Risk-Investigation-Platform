import unittest

from data.generate_synthetic_data import generate_dataset
from ml.graph_builder import graph_builder
from ml.risk_graph import build_user_graph, detect_communities
from ml.train_tabular_model import train_tabular_model, predict_tabular_fraud_prob
from ml.train_gnn import train_gnn, GraphSAGEInference
from ml.risk_aggregator import calculate_composite_risk_score, train_stacker, _LiveModels
from db.database import get_raw_sqlite_connection
from agent.graph_agent import investigation_agent
from config import APP_LOG_PATH, RISK_ENGINE_LOG_PATH, AGENT_LOG_PATH


class TestRazorRiskEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n--- Running System Integration Tests for RazorRisk ---")
        generate_dataset(num_users=300, num_transactions=2000, seed=42)
        _LiveModels.reset()
        train_stacker()  # trains tabular -> GNN -> stacker in sequence

    def test_01_risk_graph(self):
        """Canonical User-only risk graph used by the GNN/community detection."""
        conn = get_raw_sqlite_connection()
        G = build_user_graph(conn)
        communities, community_size = detect_communities(G)
        self.assertGreater(len(communities), 0, "Community detection returned empty results.")
        self.assertGreaterEqual(
            G.degree("USER_RING1_1"), 3,
            "Fraud Ring 1 user should have degree >= 3 in the user-only risk graph (shared device/IP edges)."
        )

    def test_02_dashboard_graph_builder(self):
        """The separate richer User-Device-IP-Merchant graph used only by the dashboard's visual explorer."""
        graph_builder.build_graph()
        comm = graph_builder.detect_communities()
        self.assertGreater(len(comm), 0)
        feat = graph_builder.extract_user_graph_features("USER_RING1_1")
        self.assertGreaterEqual(feat["shared_device_accounts"], 3)

    def test_03_tabular_model(self):
        sample_features = {
            "amount_log": 11.35, "hour_of_day": 3, "day_of_week": 2,
            "velocity_1h": 10, "amount_zscore_prior": 3.5, "merchant_fraud_rate": 0.3,
        }
        tab_score = predict_tabular_fraud_prob(sample_features)
        self.assertGreaterEqual(tab_score, 0.0)
        self.assertLessEqual(tab_score, 1.0)

    def test_04_gnn_model_flags_known_ring_member(self):
        gnn = GraphSAGEInference()
        conn = get_raw_sqlite_connection()
        G = build_user_graph(conn)
        communities, community_size = detect_communities(G)
        from ml.risk_graph import fetch_node_features, build_adjacency
        user_ids, X = fetch_node_features(conn, G, community_size)
        A = build_adjacency(G, user_ids)
        scores = gnn.score_all(X, A)
        idx = user_ids.index("USER_RING1_1")
        self.assertGreaterEqual(scores[idx], 0.5, "GNN should flag a known fraud-ring member as high risk.")

    def test_05_risk_aggregator_and_agent(self):
        sample_txn = {
            "transaction_id": "TXN_TEST_001", "user_id": "USER_RING1_1",
            "device_id": "DEV_FRAUD_RING1", "ip_address": "185.220.101.44",
            "merchant_id": "MCH_042", "amount": 95000, "velocity_1h": 12,
            "is_vpn_proxy": True,
        }
        risk_res = calculate_composite_risk_score(sample_txn)
        self.assertGreaterEqual(risk_res["risk_score"], 70.0, "Risk score should exceed the high-risk threshold.")
        self.assertIn(risk_res["risk_tier"], ["HIGH", "CRITICAL"])

        agent_res = investigation_agent.investigate(sample_txn, risk_res)
        self.assertIsNotNone(agent_res["summary_report"])
        self.assertIn(agent_res["agent_mode"], ["deterministic_fallback"] + [f"llm:{p}" for p in ("anthropic", "groq", "openai")])
        # No LLM key is expected in a clean test environment, but this
        # assertion is mode-agnostic on purpose — it should still pass if
        # the test runs somewhere with ANTHROPIC_API_KEY set.
        self.assertTrue(len(agent_res["fraud_hypothesis"]) > 20)

    def test_06_logging_file_generation(self):
        self.assertTrue(APP_LOG_PATH.exists(), "app.log should be created.")
        self.assertTrue(RISK_ENGINE_LOG_PATH.exists(), "risk_engine.log should be created.")
        self.assertTrue(AGENT_LOG_PATH.exists(), "agent_investigations.log should be created.")


if __name__ == "__main__":
    unittest.main()
