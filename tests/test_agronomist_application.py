import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.accounts import (
    authenticate,
    create_account,
    ensure_default_admin,
    latest_agronomist_application,
    list_agronomist_applications,
    review_agronomist_application,
    submit_agronomist_application,
)


class AgronomistApplicationTests(unittest.TestCase):
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

    def tearDown(self):
        self.db_override.stop()
        self.temp_dir.cleanup()

    def _payload(self):
        return "card.jpg", b"\xff\xd8\xff fake-jpeg-bytes"

    def test_user_applies_and_admin_approves(self):
        filename, payload = self._payload()
        application_id, error = submit_agronomist_application(
            self.user["id"],
            "Field Officer",
            "DAE Bogura",
            "BSc Agriculture, DAE extension officer",
            "work_id",
            "DAE-8821",
            filename,
            payload,
            "Posted to Bogura sadar.",
        )
        self.assertEqual(error, "", error)
        latest = latest_agronomist_application(self.user["id"])
        self.assertEqual(latest["status"], "pending")
        self.assertEqual(latest["document_type"], "work_id")

        _, error = submit_agronomist_application(
            self.user["id"],
            "Field Officer",
            "DAE Bogura",
            "BSc Agriculture, DAE extension officer",
            "work_id",
            "DAE-8821",
            filename,
            payload,
        )
        self.assertIn("already waiting", error)

        pending, error = list_agronomist_applications(self.admin["id"], "pending")
        self.assertEqual(error, "")
        self.assertEqual(len(pending), 1)

        error = review_agronomist_application(
            self.admin["id"],
            application_id,
            "approved",
        )
        self.assertEqual(error, "")
        approved = authenticate("user", "user1234")[0]
        self.assertEqual(approved["role"], "agronomist")
        self.assertTrue(approved["expert_verified"])
        self.assertIn("BSc Agriculture", approved["qualifications"])
        self.assertEqual(
            latest_agronomist_application(self.user["id"])["status"],
            "approved",
        )

    def test_admin_can_reject_and_user_may_reapply(self):
        filename, payload = self._payload()
        application_id, error = submit_agronomist_application(
            self.user["id"],
            "Field Officer",
            "DAE Bogura",
            "BSc Agriculture, DAE extension officer",
            "nid",
            "1234567890",
            filename,
            payload,
        )
        self.assertEqual(error, "")
        error = review_agronomist_application(
            self.admin["id"],
            application_id,
            "rejected",
        )
        self.assertIn("reason", error)
        error = review_agronomist_application(
            self.admin["id"],
            application_id,
            "rejected",
            "Card photo is unreadable.",
        )
        self.assertEqual(error, "")
        rejected = authenticate("user", "user1234")[0]
        self.assertEqual(rejected["role"], "user")
        self.assertFalse(rejected["expert_verified"])

        second_id, error = submit_agronomist_application(
            self.user["id"],
            "Field Officer",
            "DAE Bogura",
            "BSc Agriculture, DAE extension officer",
            "nid",
            "1234567890",
            filename,
            payload,
        )
        self.assertEqual(error, "")
        self.assertNotEqual(second_id, application_id)

    def test_nid_and_file_rules(self):
        filename, payload = self._payload()
        _, error = submit_agronomist_application(
            self.user["id"],
            "Field Officer",
            "DAE Bogura",
            "BSc Agriculture",
            "nid",
            "12345",
            filename,
            payload,
        )
        self.assertIn("National ID", error)
        _, error = submit_agronomist_application(
            self.user["id"],
            "Field Officer",
            "DAE Bogura",
            "BSc Agriculture",
            "work_id",
            "DAE-8821",
            "card.txt",
            b"not-an-image",
        )
        self.assertIn("JPG, PNG, or PDF", error)
        outsider, error = create_account(
            "Outsider",
            "out@example.org",
            "password123",
            "NGO",
            "user",
            [],
            [],
        )
        self.assertFalse(error)
        _, error = list_agronomist_applications(outsider["id"])
        self.assertIn("Only an admin", error)


if __name__ == "__main__":
    unittest.main()
