"""Local accounts for AgriSight-BD.

The crop models, maps, and district tables stay public. An account
remembers who is signed in: the districts they cover, visit notes,
and saved prediction runs. The first account is the admin.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from src.config import PROJECT_ROOT


DB_PATH = PROJECT_ROOT / "data" / "accounts.sqlite"
PBKDF2_ROUNDS = 120_000
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
USERNAME_PATTERN = re.compile(r"[a-z0-9._-]{3,40}")
DEFAULT_ADMIN_LOGIN = "admin"
DEFAULT_ADMIN_PASSWORD = "admin123"
DEFAULT_USER_LOGIN = "user"
DEFAULT_USER_PASSWORD = "user1234"

ROLE_LABELS = {
    "user": "User",
    "admin": "Admin",
}
# Public signup always creates a user. Admin is not self-assigned.
SIGNUP_ROLES = ["user"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    salt TEXT NOT NULL,
    organization TEXT NOT NULL DEFAULT '',
    role TEXT NOT NULL,
    districts TEXT NOT NULL DEFAULT '[]',
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS saved_runs (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    kind TEXT NOT NULL,
    crop_id TEXT NOT NULL,
    crop_name TEXT NOT NULL,
    district TEXT NOT NULL,
    scenario TEXT NOT NULL DEFAULT '',
    predicted_yield REAL NOT NULL,
    usual_yield REAL NOT NULL,
    details TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    crop_id TEXT NOT NULL,
    district TEXT NOT NULL,
    body TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (user_id, crop_id, district),
    FOREIGN KEY (user_id) REFERENCES users(id)
);
CREATE INDEX IF NOT EXISTS idx_runs_user ON saved_runs(user_id, id);
CREATE INDEX IF NOT EXISTS idx_notes_user ON notes(user_id);
CREATE TABLE IF NOT EXISTS admin_events (
    id INTEGER PRIMARY KEY,
    actor_id INTEGER NOT NULL,
    actor_name TEXT NOT NULL,
    action TEXT NOT NULL,
    detail TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def db_path() -> Path:
    """SQLite file for accounts. Tests can override it with an env var."""

    override = os.environ.get("AGRISIGHT_ACCOUNTS_DB")
    if override:
        return Path(override)
    return DB_PATH


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _connect() -> sqlite3.Connection:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(SCHEMA)
    conn.execute(
        """
        UPDATE users
        SET role = 'user'
        WHERE role NOT IN ('user', 'admin')
        """
    )
    conn.commit()
    return conn


def _hash_password(password: str, salt: str) -> str:
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        PBKDF2_ROUNDS,
    )
    return digest.hex()


def _clean_label(text: str, limit: int) -> str:
    cleaned = " ".join(str(text).split())
    for char in "*_[]<>":
        cleaned = cleaned.replace(char, "")
    return cleaned[:limit].strip()


def _clean_email(email: str) -> Optional[str]:
    cleaned = " ".join(str(email).split()).lower()
    if not EMAIL_PATTERN.match(cleaned):
        return None
    return cleaned


def _clean_login(value: str) -> Optional[str]:
    """Email or a short username such as admin."""

    cleaned = " ".join(str(value).split()).lower()
    if EMAIL_PATTERN.match(cleaned) or USERNAME_PATTERN.fullmatch(cleaned):
        return cleaned
    return None


def _chosen_districts(
    districts: Iterable[str],
    allowed: Iterable[str],
) -> list[str]:
    allowed_names = set(allowed)
    chosen: list[str] = []
    for name in districts:
        label = _clean_label(str(name), 80)
        if label in allowed_names and label not in chosen:
            chosen.append(label)
    return chosen


def _plain(value):
    """JSON-ready copy of a run's details."""

    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float, str)):
        return value
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _user_from_row(row: sqlite3.Row) -> dict:
    try:
        districts = json.loads(row["districts"])
    except json.JSONDecodeError:
        districts = []
    if not isinstance(districts, list):
        districts = []
    return {
        "id": int(row["id"]),
        "name": row["name"],
        "email": row["email"],
        "organization": row["organization"],
        "role": row["role"],
        "districts": [str(name) for name in districts],
        "active": bool(row["active"]),
        "created_at": row["created_at"],
    }


def user_count() -> int:
    conn = _connect()
    try:
        row = conn.execute("SELECT COUNT(*) AS n FROM users").fetchone()
        return int(row["n"])
    finally:
        conn.close()


def get_user(user_id: int) -> Optional[dict]:
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM users WHERE id = ?",
            (int(user_id),),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return _user_from_row(row)


def list_users() -> list[dict]:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT * FROM users ORDER BY name COLLATE NOCASE, id"
        ).fetchall()
    finally:
        conn.close()
    return [_user_from_row(row) for row in rows]


def create_account(
    name: str,
    email: str,
    password: str,
    organization: str,
    role: str,
    districts: Iterable[str],
    allowed_districts: Iterable[str],
) -> tuple:
    """Create an account. Returns (user, error). Error is empty on success."""

    name = _clean_label(name, 80)
    if len(name) < 2:
        return None, "Enter your name."
    cleaned_email = _clean_email(email)
    if cleaned_email is None:
        return None, "Enter a valid email address."
    if len(password) < 8 or not password.strip():
        return None, "Use at least 8 characters for the password."
    organization = _clean_label(organization, 120)
    chosen = _chosen_districts(districts, allowed_districts)

    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        count = int(
            conn.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"]
        )
        if count == 0:
            stored_role = "admin"
        else:
            stored_role = "user"
        salt = secrets.token_hex(16)
        try:
            cursor = conn.execute(
                """
                INSERT INTO users (
                    name, email, password_hash, salt, organization,
                    role, districts, active, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?)
                """,
                (
                    name,
                    cleaned_email,
                    _hash_password(password, salt),
                    salt,
                    organization,
                    stored_role,
                    json.dumps(chosen),
                    _now(),
                ),
            )
        except sqlite3.IntegrityError:
            conn.rollback()
            return None, "An account with this email already exists."
        conn.commit()
        user_id = int(cursor.lastrowid)
    finally:
        conn.close()
    return get_user(user_id), ""


def _ensure_builtin(
    login: str,
    password: str,
    name: str,
    role: str,
    organization: str,
    reset_password: bool,
) -> None:
    """Create one built-in account when it is missing."""

    conn = _connect()
    try:
        row = conn.execute(
            "SELECT id FROM users WHERE email = ?",
            (login,),
        ).fetchone()
        salt = secrets.token_hex(16)
        digest = _hash_password(password, salt)
        if row is None:
            conn.execute(
                """
                INSERT INTO users (
                    name, email, password_hash, salt, organization,
                    role, districts, active, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, '[]', 1, ?)
                """,
                (name, login, digest, salt, organization, role, _now()),
            )
        elif reset_password:
            conn.execute(
                """
                UPDATE users
                SET password_hash = ?, salt = ?, role = ?, active = 1
                WHERE id = ?
                """,
                (digest, salt, role, int(row["id"])),
            )
        conn.commit()
    finally:
        conn.close()


def ensure_default_admin(reset_password: bool = False) -> None:
    """Keep the built-in admin and the built-in field user."""

    _ensure_builtin(
        DEFAULT_ADMIN_LOGIN,
        DEFAULT_ADMIN_PASSWORD,
        "Admin",
        "admin",
        "",
        reset_password,
    )
    _ensure_builtin(
        DEFAULT_USER_LOGIN,
        DEFAULT_USER_PASSWORD,
        "Field user",
        "user",
        "DAE",
        reset_password,
    )


def authenticate(email: str, password: str) -> tuple:
    """Return (user, error). The same message is used for an unknown login."""

    cleaned_login = _clean_login(email)
    if cleaned_login is None or not password:
        return None, "Those sign-in details do not match."
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?",
            (cleaned_login,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None, "Those sign-in details do not match."
    digest = _hash_password(password, row["salt"])
    if not hmac.compare_digest(digest, row["password_hash"]):
        return None, "Those sign-in details do not match."
    user = _user_from_row(row)
    if not user["active"]:
        return None, "This account is turned off. Ask an admin to turn it on."
    return user, ""


def update_profile(
    user_id: int,
    name: str,
    organization: str,
    districts: Iterable[str],
    allowed_districts: Iterable[str],
) -> str:
    """Save name, organization, and covered districts. Empty string means ok."""

    name = _clean_label(name, 80)
    if len(name) < 2:
        return "Enter your name."
    organization = _clean_label(organization, 120)
    chosen = _chosen_districts(districts, allowed_districts)
    conn = _connect()
    try:
        cursor = conn.execute(
            """
            UPDATE users
            SET name = ?, organization = ?, districts = ?
            WHERE id = ?
            """,
            (name, organization, json.dumps(chosen), int(user_id)),
        )
        conn.commit()
        if cursor.rowcount == 0:
            return "That account was not found."
    finally:
        conn.close()
    return ""


def change_password(user_id: int, current: str, new_password: str) -> str:
    if len(new_password) < 8 or not new_password.strip():
        return "Use at least 8 characters for the password."
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM users WHERE id = ?",
            (int(user_id),),
        ).fetchone()
        if row is None:
            return "That account was not found."
        digest = _hash_password(current, row["salt"])
        if not hmac.compare_digest(digest, row["password_hash"]):
            return "The current password does not match."
        salt = secrets.token_hex(16)
        conn.execute(
            """
            UPDATE users
            SET password_hash = ?, salt = ?
            WHERE id = ?
            """,
            (_hash_password(new_password, salt), salt, int(user_id)),
        )
        conn.commit()
    finally:
        conn.close()
    return ""


def _active_admin_count(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        """
        SELECT COUNT(*) AS n
        FROM users
        WHERE role = 'admin' AND active = 1
        """
    ).fetchone()
    return int(row["n"])


def admin_update_user(
    actor_id: int,
    user_id: int,
    role: str,
    active: bool,
    districts: Iterable[str],
    allowed_districts: Iterable[str],
    new_password: str = "",
) -> str:
    """Admin changes to role, active flag, districts, and an optional password."""

    if role not in ROLE_LABELS:
        return "Choose a role."
    if new_password and (len(new_password) < 8 or not new_password.strip()):
        return "Use at least 8 characters for the new password."
    chosen = _chosen_districts(districts, allowed_districts)
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        actor = conn.execute(
            "SELECT * FROM users WHERE id = ?",
            (int(actor_id),),
        ).fetchone()
        if (
            actor is None
            or actor["role"] != "admin"
            or not actor["active"]
        ):
            conn.rollback()
            return "Only an admin can change accounts."
        target = conn.execute(
            "SELECT * FROM users WHERE id = ?",
            (int(user_id),),
        ).fetchone()
        if target is None:
            conn.rollback()
            return "That account was not found."
        if int(actor_id) == int(user_id) and not active:
            conn.rollback()
            return "Turn on another admin before turning your own account off."
        removing_admin = (
            target["role"] == "admin"
            and target["active"]
            and (role != "admin" or not active)
        )
        if removing_admin and _active_admin_count(conn) <= 1:
            conn.rollback()
            return "Keep at least one active admin."
        conn.execute(
            """
            UPDATE users
            SET role = ?, active = ?, districts = ?
            WHERE id = ?
            """,
            (
                role,
                1 if active else 0,
                json.dumps(chosen),
                int(user_id),
            ),
        )
        if new_password:
            salt = secrets.token_hex(16)
            conn.execute(
                """
                UPDATE users
                SET password_hash = ?, salt = ?
                WHERE id = ?
                """,
                (
                    _hash_password(new_password, salt),
                    salt,
                    int(user_id),
                ),
            )
        conn.commit()
    finally:
        conn.close()
    return ""


def save_run(
    user_id: int,
    kind: str,
    crop_id: str,
    crop_name: str,
    district: str,
    scenario: str,
    predicted_yield: float,
    usual_yield: float,
    details: dict,
) -> str:
    if kind not in ("playground", "what_if"):
        return "Unknown run type."
    conn = _connect()
    try:
        owner = conn.execute(
            "SELECT id FROM users WHERE id = ? AND active = 1",
            (int(user_id),),
        ).fetchone()
        if owner is None:
            return "Sign in again to save this run."
        conn.execute(
            """
            INSERT INTO saved_runs (
                user_id, kind, crop_id, crop_name, district, scenario,
                predicted_yield, usual_yield, details, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(user_id),
                kind,
                _clean_label(crop_id, 80),
                _clean_label(crop_name, 80),
                _clean_label(district, 80),
                _clean_label(scenario, 160),
                float(predicted_yield),
                float(usual_yield),
                json.dumps(_plain(details), ensure_ascii=False),
                _now(),
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return ""


def list_runs(user_id: int) -> list[dict]:
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT * FROM saved_runs
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT 100
            """,
            (int(user_id),),
        ).fetchall()
    finally:
        conn.close()
    runs = []
    for row in rows:
        try:
            details = json.loads(row["details"])
        except json.JSONDecodeError:
            details = {}
        if not isinstance(details, dict):
            details = {}
        runs.append(
            {
                "id": int(row["id"]),
                "kind": row["kind"],
                "crop_id": row["crop_id"],
                "crop_name": row["crop_name"],
                "district": row["district"],
                "scenario": row["scenario"],
                "predicted_yield": float(row["predicted_yield"]),
                "usual_yield": float(row["usual_yield"]),
                "details": details,
                "created_at": row["created_at"],
            }
        )
    return runs


def delete_run(user_id: int, run_id: int) -> None:
    conn = _connect()
    try:
        conn.execute(
            "DELETE FROM saved_runs WHERE id = ? AND user_id = ?",
            (int(run_id), int(user_id)),
        )
        conn.commit()
    finally:
        conn.close()


def get_note(user_id: int, crop_id: str, district: str) -> str:
    conn = _connect()
    try:
        row = conn.execute(
            """
            SELECT body FROM notes
            WHERE user_id = ? AND crop_id = ? AND district = ?
            """,
            (int(user_id), crop_id, district),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return ""
    return str(row["body"])


def _clean_note(body: str) -> str:
    lines = []
    for line in str(body).replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        lines.append(" ".join(line.split()))
    return "\n".join(lines).strip()[:2000]


def save_note(
    user_id: int,
    crop_id: str,
    district: str,
    body: str,
) -> None:
    text = _clean_note(body)
    conn = _connect()
    try:
        if not text:
            conn.execute(
                """
                DELETE FROM notes
                WHERE user_id = ? AND crop_id = ? AND district = ?
                """,
                (int(user_id), crop_id, district),
            )
        else:
            conn.execute(
                """
                INSERT INTO notes (
                    user_id, crop_id, district, body, updated_at
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id, crop_id, district)
                DO UPDATE SET
                    body = excluded.body,
                    updated_at = excluded.updated_at
                """,
                (int(user_id), crop_id, district, text, _now()),
            )
        conn.commit()
    finally:
        conn.close()


def list_notes(user_id: int) -> list[dict]:
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT * FROM notes
            WHERE user_id = ?
            ORDER BY updated_at DESC, id DESC
            """,
            (int(user_id),),
        ).fetchall()
    finally:
        conn.close()
    return [
        {
            "id": int(row["id"]),
            "crop_id": row["crop_id"],
            "district": row["district"],
            "body": row["body"],
            "updated_at": row["updated_at"],
        }
        for row in rows
    ]


def delete_note(user_id: int, crop_id: str, district: str) -> None:
    save_note(user_id, crop_id, district, "")


def log_admin_event(
    actor_id: int,
    actor_name: str,
    action: str,
    detail: str,
) -> None:
    """Append one operations-console event."""

    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO admin_events (
                actor_id, actor_name, action, detail, created_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                int(actor_id),
                _clean_label(actor_name, 80) or "Admin",
                _clean_label(action, 40),
                " ".join(str(detail).split())[:500],
                _now(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def list_admin_events(limit: int = 80) -> list[dict]:
    """Newest operations-console events first."""

    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT * FROM admin_events
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    finally:
        conn.close()
    return [
        {
            "id": int(row["id"]),
            "actor_id": int(row["actor_id"]),
            "actor_name": row["actor_name"],
            "action": row["action"],
            "detail": row["detail"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]
