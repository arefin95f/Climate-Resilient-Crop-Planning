import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.accounts import (
    admin_update_user,
    assign_review_request,
    authenticate,
    complete_review_request,
    create_account,
    ensure_default_admin,
    list_review_events,
    list_review_requests,
    list_user_review_requests,
    list_verified_agronomists,
    start_review_request,
    submit_review_request,
)


class ExpertReviewWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_override = patch.dict(
            os.environ,
            {
                "AGRISIGHT_ACCOUNTS_DB": str(
                    Path(self.temp_dir.name) / "accounts.sqlite"
                )
            },
        )
        self.db_override.start()
        ensure_default_admin()
        self.admin, error = authenticate("admin", "admin123")
        self.assertFalse(error)
        self.user, error = authenticate("user", "user1234")
        self.assertFalse(error)
        self.requester, error = create_account(
            "Field worker",
            "field@example.org",
            "fieldpass123",
            "District office",
            "user",
            [],
            [],
        )
        self.assertFalse(error)

    def tearDown(self):
        self.db_override.stop()
        self.temp_dir.cleanup()

    def _approve_agronomist(self):
        error = admin_update_user(
            self.admin["id"],
            self.user["id"],
            "agronomist",
            True,
            [],
            [],
            qualifications="Agronomy degree · verified with issuing authority",
            expert_verified=True,
        )
        self.assertEqual(error, "")
        self.user, error = authenticate("user", "user1234")
        self.assertFalse(error)
        self.assertTrue(self.user["expert_verified"])

    def _submit_case(self):
        request_id, error = submit_review_request(
            self.requester["id"],
            "boro_rice",
            "Boro rice",
            "Bogura",
            "yield_estimate",
            4.2,
            4.0,
            {"rainfall_mm": 420},
            "Please review this yield estimate.",
        )
        self.assertEqual(error, "")
        return request_id

    def test_assignment_review_response_and_audit_history(self):
        self._approve_agronomist()
        request_id = self._submit_case()

        admin_cases = list_review_requests(self.admin["id"])
        self.assertEqual(admin_cases[0]["status"], "submitted")
        self.assertEqual(admin_cases[0]["details"]["rainfall_mm"], 420)
        self.assertEqual(
            len(list_verified_agronomists(self.admin["id"])),
            1,
        )
        self_case_id, error = submit_review_request(
            self.user["id"],
            "boro_rice",
            "Boro rice",
            "Bogura",
            "yield_estimate",
            4.1,
            4.0,
            {},
            "Please review my own estimate.",
        )
        self.assertEqual(error, "")
        self.assertIn(
            "cannot review their own",
            assign_review_request(
                self.admin["id"],
                self_case_id,
                self.user["id"],
            ),
        )

        error = assign_review_request(
            self.admin["id"],
            request_id,
            self.user["id"],
        )
        self.assertEqual(error, "")
        self.assertEqual(
            list_review_requests(self.user["id"])[0]["status"],
            "assigned",
        )

        error = start_review_request(self.user["id"], request_id)
        self.assertEqual(error, "")
        error = complete_review_request(
            self.user["id"],
            request_id,
            "Check the crop in the field and compare the estimate with local "
            "harvest records before making input decisions.",
        )
        self.assertEqual(error, "")

        case = list_user_review_requests(self.requester["id"])[0]
        self.assertEqual(case["status"], "reviewed")
        self.assertIn("Check the crop in the field", case["response"])
        history = list_review_events(self.requester["id"], request_id)
        self.assertEqual(
            [event["action"] for event in history],
            ["submitted", "assigned", "review_started", "review_completed"],
        )

    def test_unverified_or_unassigned_accounts_cannot_review_cases(self):
        request_id = self._submit_case()
        self.assertIn(
            "documented qualifications",
            admin_update_user(
                self.admin["id"],
                self.user["id"],
                "agronomist",
                True,
                [],
                [],
                qualifications="",
                expert_verified=False,
            ),
        )
        with self.assertRaises(PermissionError):
            list_review_requests(self.user["id"])
        self.assertIn(
            "active, admin-verified",
            assign_review_request(
                self.admin["id"],
                request_id,
                self.user["id"],
            ),
        )

    def test_requester_must_be_signed_in_and_question_is_required(self):
        request_id, error = submit_review_request(
            self.user["id"],
            "boro_rice",
            "Boro rice",
            "Bogura",
            "yield_estimate",
            4.2,
            4.0,
            {},
            " ",
        )
        self.assertIsNone(request_id)
        self.assertIn("Describe", error)

        request_id, error = submit_review_request(
            -1,
            "boro_rice",
            "Boro rice",
            "Bogura",
            "yield_estimate",
            4.2,
            4.0,
            {},
            "Please review.",
        )
        self.assertIsNone(request_id)
        self.assertIn("active account", error)

    def test_existing_review_cases_are_migrated_without_false_expert_claims(self):
        legacy_path = Path(self.temp_dir.name) / "legacy.sqlite"
        with sqlite3.connect(legacy_path) as connection:
            connection.executescript(
                """
                CREATE TABLE users (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    organization TEXT NOT NULL DEFAULT '',
                    role TEXT NOT NULL,
                    districts TEXT NOT NULL DEFAULT '[]',
                    active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE review_requests (
                    id INTEGER PRIMARY KEY,
                    requester_name TEXT NOT NULL DEFAULT '',
                    contact_email TEXT NOT NULL,
                    crop_id TEXT NOT NULL,
                    crop_name TEXT NOT NULL,
                    district TEXT NOT NULL,
                    predicted_yield REAL NOT NULL,
                    usual_yield REAL NOT NULL,
                    details TEXT NOT NULL DEFAULT '{}',
                    question TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                INSERT INTO review_requests (
                    requester_name, contact_email, crop_id, crop_name,
                    district, predicted_yield, usual_yield, details,
                    question, status, created_at, updated_at
                ) VALUES (
                    'Field worker', 'field@example.org', 'boro_rice',
                    'Boro rice', 'Bogura', 4.2, 4.0, '{}',
                    'Review this estimate', 'expert_reviewed', '2026-01-01',
                    '2026-01-01'
                );
                """
            )
        connection.close()
        with patch.dict(
            os.environ,
            {"AGRISIGHT_ACCOUNTS_DB": str(legacy_path)},
        ):
            ensure_default_admin()
            admin, error = authenticate("admin", "admin123")
            self.assertFalse(error)
            cases = list_review_requests(admin["id"])
        self.assertEqual(cases[0]["status"], "submitted")
        self.assertEqual(cases[0]["response"], "")


if __name__ == "__main__":
    unittest.main()
