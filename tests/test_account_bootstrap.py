import os
import sqlite3
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import src.accounts as accounts


class AccountBootstrapTests(unittest.TestCase):
    def test_concurrent_bootstrap_creates_each_builtin_account_once(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "accounts.sqlite"
            with patch.dict(
                os.environ,
                {"AGRISIGHT_ACCOUNTS_DB": str(db_path)},
            ):
                conn = accounts._connect()
                conn.execute("DELETE FROM users")
                conn.commit()
                conn.close()

                barrier = threading.Barrier(2)
                original_hash = accounts._hash_password

                def synchronized_hash(password, salt):
                    barrier.wait(timeout=10)
                    return original_hash(password, salt)

                def connect_to_ready_db():
                    conn = sqlite3.connect(db_path, timeout=10)
                    conn.row_factory = sqlite3.Row
                    return conn

                with (
                    patch.object(
                        accounts,
                        "_connect",
                        side_effect=connect_to_ready_db,
                    ),
                    patch.object(
                        accounts,
                        "_hash_password",
                        side_effect=synchronized_hash,
                    ),
                ):
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        futures = [
                            pool.submit(accounts.ensure_default_admin)
                            for _ in range(2)
                        ]
                        for future in futures:
                            future.result(timeout=30)

                conn = sqlite3.connect(db_path)
                try:
                    rows = conn.execute(
                        "SELECT email, role FROM users ORDER BY email"
                    ).fetchall()
                finally:
                    conn.close()

        self.assertEqual(rows, [("admin", "admin"), ("user", "user")])


if __name__ == "__main__":
    unittest.main()
