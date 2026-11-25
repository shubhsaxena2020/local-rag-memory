import tempfile
import unittest
from pathlib import Path

from storage import MemoryStorage
from lifecycle import FactLifecycleManager


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.storage = MemoryStorage(Path(self.temp.name) / "facts.db")
        self.storage.initialize_schema()
        self.addCleanup(self.storage.close)
        self.calls = []
        self.manager = FactLifecycleManager(self.storage, conflict_judge=self.judge)

    def judge(self, action, old_fact, new_fact, context):
        self.calls.append((action, old_fact["id"], new_fact["id"] if new_fact else None, context))
        return context == "approve"

    def test_compatible_addition_does_not_call_judge(self):
        fact_id = self.manager.create_fact("Rain is measured in millimeters.")
        self.assertEqual(self.storage.get_fact(fact_id)["status"], "active")
        self.assertEqual(self.calls, [])

    def test_accepted_and_rejected_supersede_keep_rows_and_audit(self):
        old_id = self.manager.create_fact("Status is red", entity_key="item.status")
        new_id = self.manager.create_fact("Status is blue", entity_key="item.status", context="approve")
        old = self.storage.get_fact(old_id)
        self.assertEqual(old["status"], "superseded")
        self.assertEqual(old["superseded_by"], new_id)
        self.assertEqual(self.storage.get_fact(new_id)["status"], "active")
        self.assertEqual([x["action"] for x in self.manager.audit_history(old_id)], ["created", "superseded"])

        third_id = self.manager.create_fact("Status is green", entity_key="item.status", context="reject")
        self.assertEqual(self.storage.get_fact(third_id)["status"], "active")
        self.assertEqual(self.storage.get_fact(new_id)["status"], "active")
        self.assertEqual(self.calls[-1][0], "supersede")
        self.assertEqual(self.calls[-1][-1], "reject")
        self.assertEqual(self.storage.conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0], 3)

    def test_retract_approve_and_reject_with_explicit_context(self):
        fact_id = self.manager.create_fact("Use local service")
        self.assertFalse(self.manager.retract(fact_id, context="reject"))
        self.assertEqual(self.storage.get_fact(fact_id)["status"], "active")
        self.assertTrue(self.manager.retract(fact_id, context="approve"))
        self.assertEqual(self.storage.get_fact(fact_id)["status"], "retracted")
        self.assertEqual([c[0] for c in self.calls], ["retract", "retract"])
        self.assertEqual([x["action"] for x in self.manager.audit_history(fact_id)], ["created", "retracted"])

    def test_rejected_retract_does_not_change_fact_or_audit(self):
        fact_id = self.manager.create_fact("Keep this fact")
        before = self.manager.audit_history(fact_id)
        self.assertFalse(self.manager.retract(fact_id, context="reject"))
        self.assertEqual(self.storage.get_fact(fact_id)["status"], "active")
        self.assertEqual(self.manager.audit_history(fact_id), before)

    def test_invalid_judge_result_rolls_back_new_fact(self):
        self.manager.conflict_judge = lambda *args: "yes"
        self.manager.create_fact("Initial status", entity_key="state")
        before = self.storage.conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        with self.assertRaises(ValueError):
            self.manager.create_fact("Updated status", entity_key="state")
        self.assertEqual(self.storage.conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0], before)


if __name__ == "__main__":
    unittest.main()
