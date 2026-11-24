"""Local accounts for AgriSight-BD.

The crop models, maps, and district tables stay public. An account
remembers who is signed in: the districts they cover, visit notes,
and saved prediction runs. The first account is the admin.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from src.config import CROP_CATALOG, PROJECT_ROOT


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
    "agronomist": "Verified agronomist",
}
REVIEW_CATEGORIES = {
    "yield_estimate": "Yield estimate",
    "pest_disease": "Pest or disease concern",
    "pesticide": "Pesticide question",
    "fertilizer": "Fertilizer question",
    "other": "Other crop-planning question",
}
# Public signup always creates a user. Admin is not self-assigned.
SIGNUP_ROLES = ["user"]
AGRONOMIST_DOCUMENT_TYPES = {
    "nid": "National ID (NID)",
    "work_id": "Work / DAE identity card",
    "student_id": "University student ID",
    "professional_reg": "Professional registration or license",
}
ALLOWED_APPLICATION_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".pdf": "application/pdf",
}
APPLICATION_MAX_BYTES = 5 * 1024 * 1024

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
    qualifications TEXT NOT NULL DEFAULT '',
    expert_verified INTEGER NOT NULL DEFAULT 0,
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
CREATE TABLE IF NOT EXISTS review_requests (
    id INTEGER PRIMARY KEY,
    requester_name TEXT NOT NULL DEFAULT '',
    contact_email TEXT NOT NULL,
    crop_id TEXT NOT NULL,
    crop_name TEXT NOT NULL,
    district TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'yield_estimate',
    predicted_yield REAL NOT NULL,
    usual_yield REAL NOT NULL,
    details TEXT NOT NULL DEFAULT '{}',
    question TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'submitted',
    requester_user_id INTEGER,
    assigned_to INTEGER,
    response TEXT NOT NULL DEFAULT '',
    reviewed_at TEXT,
    closed_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (requester_user_id) REFERENCES users(id),
    FOREIGN KEY (assigned_to) REFERENCES users(id)
);
CREATE TABLE IF NOT EXISTS review_events (
    id INTEGER PRIMARY KEY,
    request_id INTEGER NOT NULL,
    actor_id INTEGER,
    action TEXT NOT NULL,
    from_status TEXT NOT NULL DEFAULT '',
    to_status TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    FOREIGN KEY (request_id) REFERENCES review_requests(id),
    FOREIGN KEY (actor_id) REFERENCES users(id)
);
CREATE INDEX IF NOT EXISTS idx_runs_user ON saved_runs(user_id, id);
CREATE INDEX IF NOT EXISTS idx_notes_user ON notes(user_id);
CREATE INDEX IF NOT EXISTS idx_review_requests_status
    ON review_requests(status, id);
CREATE INDEX IF NOT EXISTS idx_review_events_request
    ON review_events(request_id, id);
CREATE TABLE IF NOT EXISTS admin_events (
    id INTEGER PRIMARY KEY,
    actor_id INTEGER NOT NULL,
    actor_name TEXT NOT NULL,
    action TEXT NOT NULL,
    detail TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS agronomist_applications (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    full_name TEXT NOT NULL,
    organization TEXT NOT NULL,
    qualifications TEXT NOT NULL,
    document_type TEXT NOT NULL,
    document_number TEXT NOT NULL,
    document_filename TEXT NOT NULL DEFAULT '',
    document_path TEXT NOT NULL DEFAULT '',
    notes TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending',
    review_note TEXT NOT NULL DEFAULT '',
    reviewed_by INTEGER,
    submitted_at TEXT NOT NULL,
    reviewed_at TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (reviewed_by) REFERENCES users(id)
);
CREATE INDEX IF NOT EXISTS idx_agronomist_applications_user
    ON agronomist_applications(user_id, id);
CREATE INDEX IF NOT EXISTS idx_agronomist_applications_status
    ON agronomist_applications(status, id);
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
    _ensure_column(
        conn,
        "users",
        "qualifications",
        "TEXT NOT NULL DEFAULT ''",
    )
    _ensure_column(conn, "users", "expert_verified", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(conn, "review_requests", "requester_user_id", "INTEGER")
    _ensure_column(conn, "review_requests", "assigned_to", "INTEGER")
    _ensure_column(
        conn,
        "review_requests",
        "category",
        "TEXT NOT NULL DEFAULT 'yield_estimate'",
    )
    _ensure_column(
        conn,
        "review_requests",
        "response",
        "TEXT NOT NULL DEFAULT ''",
    )
    _ensure_column(conn, "review_requests", "reviewed_at", "TEXT")
    _ensure_column(conn, "review_requests", "closed_at", "TEXT")
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_review_requests_assigned
        ON review_requests(assigned_to, status, id)
        """
    )
    conn.execute(
        """
        UPDATE users
        SET role = 'user'
        WHERE role NOT IN ('user', 'admin', 'agronomist')
        """
    )
    conn.execute(
        "UPDATE review_requests SET status = 'submitted' WHERE status = 'pending'"
    )
    conn.execute(
        """
        UPDATE review_requests SET status = 'submitted'
        WHERE status = 'expert_reviewed'
        """
    )
    conn.commit()
    return conn


def _ensure_column(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    definition: str,
) -> None:
    columns = {
        row["name"]
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }
    if column not in columns:
        conn.execute(
            f"ALTER TABLE {table} ADD COLUMN {column} {definition}"
        )


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
        "qualifications": row["qualifications"],
        "expert_verified": bool(row["expert_verified"]),
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
                ON CONFLICT(email) DO NOTHING
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
    qualifications: str = "",
    expert_verified: bool = False,
) -> str:
    """Admin changes to role, active flag, districts, and an optional password."""

    if role not in ROLE_LABELS:
        return "Choose a role."
    if new_password and (len(new_password) < 8 or not new_password.strip()):
        return "Use at least 8 characters for the new password."
    qualifications = _clean_label(qualifications, 300)
    if role == "agronomist" and (
        not qualifications or not expert_verified
    ):
        return (
            "An agronomist needs documented qualifications and explicit "
            "admin verification."
        )
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
            SET role = ?, active = ?, districts = ?, qualifications = ?,
                expert_verified = ?
            WHERE id = ?
            """,
            (
                role,
                1 if active else 0,
                json.dumps(chosen),
                qualifications,
                1 if role == "agronomist" and expert_verified else 0,
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
        conn.execute(
            """
            INSERT INTO admin_events (
                actor_id, actor_name, action, detail, created_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                int(actor_id),
                actor["name"],
                "account_updated",
                (
                    f"Updated {target['email']}; role={role}; "
                    f"active={bool(active)}; "
                    f"agronomist_verified="
                    f"{bool(role == 'agronomist' and expert_verified)}"
                ),
                _now(),
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return ""


def application_dir() -> Path:
    """Local folder for agronomist identity documents."""

    override = os.environ.get("AGRISIGHT_ACCOUNTS_DB")
    if override:
        return Path(override).parent / "agronomist_applications"
    return PROJECT_ROOT / "data" / "agronomist_applications"


def _application_from_row(row: sqlite3.Row) -> dict:
    keys = row.keys()
    return {
        "id": int(row["id"]),
        "user_id": int(row["user_id"]),
        "full_name": row["full_name"],
        "organization": row["organization"],
        "qualifications": row["qualifications"],
        "document_type": row["document_type"],
        "document_type_label": AGRONOMIST_DOCUMENT_TYPES.get(
            row["document_type"],
            row["document_type"],
        ),
        "document_number": row["document_number"],
        "document_filename": row["document_filename"],
        "document_path": row["document_path"],
        "notes": row["notes"],
        "status": row["status"],
        "review_note": row["review_note"],
        "reviewed_by": (
            int(row["reviewed_by"]) if row["reviewed_by"] is not None else None
        ),
        "submitted_at": row["submitted_at"],
        "reviewed_at": row["reviewed_at"],
        "email": row["email"] if "email" in keys else "",
        "account_name": row["account_name"] if "account_name" in keys else "",
        "account_role": row["account_role"] if "account_role" in keys else "",
    }


def _clean_document_number(document_type: str, number: str) -> tuple[str, str]:
    raw = " ".join(str(number).split())
    if document_type == "nid":
        digits = re.sub(r"\D", "", raw)
        if len(digits) not in (10, 13, 17):
            return "", "Enter a 10, 13, or 17-digit National ID number."
        return digits, ""
    cleaned = re.sub(r"[^A-Za-z0-9/-]", "", raw).upper()
    if len(cleaned) < 4 or len(cleaned) > 40:
        return "", "Enter the ID or registration number on the card."
    return cleaned, ""


def _store_application_file(
    application_id: int,
    filename: str,
    payload: bytes,
) -> tuple[str, str, str]:
    suffix = Path(filename or "").suffix.lower()
    if suffix not in ALLOWED_APPLICATION_TYPES:
        return "", "", "Upload a JPG, PNG, or PDF of the identity card."
    if not payload or len(payload) > APPLICATION_MAX_BYTES:
        return "", "", "The identity file must be 5 MB or smaller."
    folder = application_dir()
    folder.mkdir(parents=True, exist_ok=True)
    stored_name = f"{application_id}{suffix}"
    path = folder / stored_name
    path.write_bytes(payload)
    original = _clean_label(Path(filename).name, 80) or stored_name
    return original, str(path), ""


def latest_agronomist_application(user_id: int) -> Optional[dict]:
    conn = _connect()
    try:
        row = conn.execute(
            """
            SELECT a.*, u.email AS email, u.name AS account_name,
                   u.role AS account_role
            FROM agronomist_applications AS a
            JOIN users AS u ON u.id = a.user_id
            WHERE a.user_id = ?
            ORDER BY a.id DESC
            LIMIT 1
            """,
            (int(user_id),),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return _application_from_row(row)


def list_agronomist_applications(
    actor_id: int,
    status: str = "",
) -> tuple[list[dict], str]:
    conn = _connect()
    try:
        actor = conn.execute(
            "SELECT * FROM users WHERE id = ?",
            (int(actor_id),),
        ).fetchone()
        if actor is None or actor["role"] != "admin" or not actor["active"]:
            return [], "Only an admin can read agronomist applications."
        query = """
            SELECT a.*, u.email AS email, u.name AS account_name,
                   u.role AS account_role
            FROM agronomist_applications AS a
            JOIN users AS u ON u.id = a.user_id
        """
        params: list = []
        if status:
            query += " WHERE a.status = ?"
            params.append(status)
        query += " ORDER BY CASE a.status WHEN 'pending' THEN 0 ELSE 1 END, a.id DESC"
        rows = conn.execute(query, params).fetchall()
    finally:
        conn.close()
    return [_application_from_row(row) for row in rows], ""


def application_document_bytes(
    actor_id: int,
    application_id: int,
) -> tuple[Optional[bytes], str, str]:
    applications, error = list_agronomist_applications(actor_id)
    if error:
        return None, "", error
    match = next(
        (item for item in applications if item["id"] == int(application_id)),
        None,
    )
    if match is None or not match["document_path"]:
        return None, "", "That identity file was not found."
    path = Path(match["document_path"])
    if not path.is_file():
        return None, "", "That identity file was not found."
    return path.read_bytes(), match["document_filename"] or path.name, ""


def submit_agronomist_application(
    user_id: int,
    full_name: str,
    organization: str,
    qualifications: str,
    document_type: str,
    document_number: str,
    filename: str,
    payload: bytes,
    notes: str = "",
) -> tuple[Optional[int], str]:
    """A signed-in user asks an admin to verify them as an agronomist."""

    if document_type not in AGRONOMIST_DOCUMENT_TYPES:
        return None, "Choose the type of identity document."
    name = _clean_label(full_name, 80)
    if len(name) < 2:
        return None, "Enter the name printed on the identity document."
    organization = _clean_label(organization, 120)
    if len(organization) < 2:
        return None, "Enter the university, DAE office, or employer."
    qualifications = _clean_label(qualifications, 300)
    if len(qualifications) < 8:
        return None, "Describe the agronomy qualification or posting."
    number, number_error = _clean_document_number(document_type, document_number)
    if number_error:
        return None, number_error
    notes = _clean_label(notes, 500)
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        user = conn.execute(
            "SELECT * FROM users WHERE id = ?",
            (int(user_id),),
        ).fetchone()
        if user is None or not user["active"]:
            conn.rollback()
            return None, "Sign in with an active account to apply."
        if user["role"] == "agronomist" and user["expert_verified"]:
            conn.rollback()
            return None, "This account is already a verified agronomist."
        if user["role"] == "admin":
            conn.rollback()
            return None, "Admin accounts do not apply for agronomist review."
        pending = conn.execute(
            """
            SELECT id FROM agronomist_applications
            WHERE user_id = ? AND status = 'pending'
            """,
            (int(user_id),),
        ).fetchone()
        if pending is not None:
            conn.rollback()
            return None, "An application is already waiting for admin review."
        cursor = conn.execute(
            """
            INSERT INTO agronomist_applications (
                user_id, full_name, organization, qualifications,
                document_type, document_number, notes, status, submitted_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)
            """,
            (
                int(user_id),
                name,
                organization,
                qualifications,
                document_type,
                number,
                notes,
                _now(),
            ),
        )
        application_id = int(cursor.lastrowid)
        stored_name, stored_path, file_error = _store_application_file(
            application_id,
            filename,
            payload,
        )
        if file_error:
            conn.rollback()
            return None, file_error
        conn.execute(
            """
            UPDATE agronomist_applications
            SET document_filename = ?, document_path = ?
            WHERE id = ?
            """,
            (stored_name, stored_path, application_id),
        )
        conn.commit()
    finally:
        conn.close()
    return application_id, ""


def review_agronomist_application(
    actor_id: int,
    application_id: int,
    decision: str,
    review_note: str = "",
) -> str:
    """Admin approves or rejects a submitted agronomist application."""

    if decision not in ("approved", "rejected"):
        return "Choose approve or reject."
    review_note = _clean_label(review_note, 400)
    if decision == "rejected" and len(review_note) < 4:
        return "Add a short reason when rejecting an application."
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        actor = conn.execute(
            "SELECT * FROM users WHERE id = ?",
            (int(actor_id),),
        ).fetchone()
        if actor is None or actor["role"] != "admin" or not actor["active"]:
            conn.rollback()
            return "Only an admin can decide an agronomist application."
        application = conn.execute(
            "SELECT * FROM agronomist_applications WHERE id = ?",
            (int(application_id),),
        ).fetchone()
        if application is None:
            conn.rollback()
            return "That application was not found."
        if application["status"] != "pending":
            conn.rollback()
            return "This application has already been decided."
        target = conn.execute(
            "SELECT * FROM users WHERE id = ?",
            (int(application["user_id"]),),
        ).fetchone()
        if target is None or not target["active"]:
            conn.rollback()
            return "That account is no longer active."
        stamp = _now()
        conn.execute(
            """
            UPDATE agronomist_applications
            SET status = ?, review_note = ?, reviewed_by = ?, reviewed_at = ?
            WHERE id = ?
            """,
            (
                decision,
                review_note,
                int(actor_id),
                stamp,
                int(application_id),
            ),
        )
        if decision == "approved":
            conn.execute(
                """
                UPDATE users
                SET role = 'agronomist', qualifications = ?, expert_verified = 1
                WHERE id = ?
                """,
                (
                    application["qualifications"],
                    int(application["user_id"]),
                ),
            )
        conn.execute(
            """
            INSERT INTO admin_events (
                actor_id, actor_name, action, detail, created_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                int(actor_id),
                actor["name"],
                "agronomist_application_" + decision,
                f"Application #{application_id} for {target['email']}",
                stamp,
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


def submit_review_request(
    requester_id: int,
    crop_id: str,
    crop_name: str,
    district: str,
    category: str,
    predicted_yield: float,
    usual_yield: float,
    details: dict,
    question: str,
) -> tuple[Optional[int], str]:
    """Create a trackable review case for a signed-in requester."""

    if crop_id not in CROP_CATALOG:
        return None, "Choose a supported crop."
    crop_name = _clean_label(crop_name, 80)
    district = _clean_label(district, 80)
    if not crop_name or not district:
        return None, "Crop and district are required."
    if category not in REVIEW_CATEGORIES:
        return None, "Choose a valid review category."
    if not isinstance(details, dict):
        return None, "The prediction context is invalid."
    cleaned_question = _clean_note(question)
    if not cleaned_question:
        return None, "Describe what you would like reviewed."
    try:
        predicted_yield = float(predicted_yield)
        usual_yield = float(usual_yield)
    except (TypeError, ValueError):
        return None, "The prediction values are invalid."
    if not all(
        math.isfinite(value)
        for value in (predicted_yield, usual_yield)
    ):
        return None, "The prediction values are invalid."

    conn = _connect()
    try:
        requester = conn.execute(
            """
            SELECT name, email FROM users
            WHERE id = ? AND active = 1
            """,
            (int(requester_id),),
        ).fetchone()
        if requester is None:
            return None, "Sign in with an active account to submit a review request."
        now = _now()
        cursor = conn.execute(
            """
            INSERT INTO review_requests (
                requester_name, contact_email, crop_id, crop_name, district,
                category, predicted_yield, usual_yield, details, question, status,
                requester_user_id, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'submitted', ?, ?, ?)
            """,
            (
                requester["name"],
                requester["email"],
                _clean_label(crop_id, 80),
                crop_name,
                district,
                category,
                predicted_yield,
                usual_yield,
                json.dumps(_plain(details), ensure_ascii=False),
                cleaned_question,
                int(requester_id),
                now,
                now,
            ),
        )
        _log_review_event(
            conn,
            int(cursor.lastrowid),
            int(requester_id),
            "submitted",
            "",
            "submitted",
        )
        request_id = int(cursor.lastrowid)
        conn.commit()
    finally:
        conn.close()
    return request_id, ""


def _log_review_event(
    conn: sqlite3.Connection,
    request_id: int,
    actor_id: Optional[int],
    action: str,
    from_status: str,
    to_status: str,
) -> None:
    conn.execute(
        """
        INSERT INTO review_events (
            request_id, actor_id, action, from_status, to_status, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            int(request_id),
            None if actor_id is None else int(actor_id),
            str(action)[:40],
            str(from_status)[:40],
            str(to_status)[:40],
            _now(),
        ),
    )


def _active_user(conn: sqlite3.Connection, user_id: int) -> Optional[sqlite3.Row]:
    row = conn.execute(
        """
        SELECT * FROM users WHERE id = ? AND active = 1
        """,
        (int(user_id),),
    ).fetchone()
    return row


def _verified_agronomist(
    conn: sqlite3.Connection,
    user_id: int,
) -> Optional[sqlite3.Row]:
    row = conn.execute(
        """
        SELECT * FROM users
        WHERE id = ? AND role = 'agronomist' AND active = 1
            AND expert_verified = 1
        """,
        (int(user_id),),
    ).fetchone()
    return row


def _review_request_dict(row: sqlite3.Row) -> dict:
    try:
        details = json.loads(row["details"])
    except json.JSONDecodeError:
        details = {}
    if not isinstance(details, dict):
        details = {}
    return {
        "id": int(row["id"]),
        "requester_name": row["requester_name"],
        "contact_email": row["contact_email"],
        "requester_user_id": (
            int(row["requester_user_id"])
            if row["requester_user_id"] is not None
            else None
        ),
        "crop_id": row["crop_id"],
        "crop_name": row["crop_name"],
        "district": row["district"],
        "category": row["category"],
        "predicted_yield": float(row["predicted_yield"]),
        "usual_yield": float(row["usual_yield"]),
        "details": details,
        "question": row["question"],
        "status": row["status"],
        "assigned_to": (
            int(row["assigned_to"]) if row["assigned_to"] is not None else None
        ),
        "assigned_name": row["assigned_name"],
        "response": row["response"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "reviewed_at": row["reviewed_at"],
        "closed_at": row["closed_at"],
    }


REVIEW_REQUEST_SELECT = """
    SELECT r.*, u.name AS assigned_name
    FROM review_requests AS r
    LEFT JOIN users AS u ON u.id = r.assigned_to
"""


def list_review_requests(actor_id: int) -> list[dict]:
    """Admins see all cases; verified agronomists see only assigned cases."""

    conn = _connect()
    try:
        actor = _active_user(conn, actor_id)
        if actor is None:
            raise PermissionError("An active account is required.")
        if actor["role"] == "admin":
            where = ""
            params = ()
        elif _verified_agronomist(conn, actor_id) is not None:
            where = "WHERE r.assigned_to = ?"
            params = (int(actor_id),)
        else:
            raise PermissionError(
                "Only admins and verified agronomists can view review cases."
            )
        rows = conn.execute(
            REVIEW_REQUEST_SELECT
            + f"""
            {where}
            ORDER BY CASE r.status
                WHEN 'submitted' THEN 0
                WHEN 'assigned' THEN 1
                WHEN 'in_review' THEN 2
                WHEN 'reviewed' THEN 3
                ELSE 4
            END, r.id DESC
            """,
            params,
        ).fetchall()
    finally:
        conn.close()
    return [_review_request_dict(row) for row in rows]


def list_user_review_requests(user_id: int) -> list[dict]:
    """Return an account's own review requests and expert responses."""

    conn = _connect()
    try:
        if _active_user(conn, user_id) is None:
            raise PermissionError("An active account is required.")
        rows = conn.execute(
            REVIEW_REQUEST_SELECT
            + """
            WHERE r.requester_user_id = ?
            ORDER BY r.id DESC
            """,
            (int(user_id),),
        ).fetchall()
    finally:
        conn.close()
    return [_review_request_dict(row) for row in rows]


def list_review_events(actor_id: int, request_id: int) -> list[dict]:
    """Return case history only to its requester, assigned expert, or admin."""

    conn = _connect()
    try:
        actor = _active_user(conn, actor_id)
        request = conn.execute(
            "SELECT requester_user_id, assigned_to FROM review_requests WHERE id = ?",
            (int(request_id),),
        ).fetchone()
        if actor is None or request is None:
            raise PermissionError("The review case is not available.")
        authorized = (
            actor["role"] == "admin"
            or (
                request["requester_user_id"] is not None
                and int(request["requester_user_id"]) == int(actor_id)
            )
            or (
                request["assigned_to"] is not None
                and int(request["assigned_to"]) == int(actor_id)
                and _verified_agronomist(conn, actor_id) is not None
            )
        )
        if not authorized:
            raise PermissionError("You cannot view this review case history.")
        rows = conn.execute(
            """
            SELECT e.*, u.name AS actor_name
            FROM review_events AS e
            LEFT JOIN users AS u ON u.id = e.actor_id
            WHERE e.request_id = ?
            ORDER BY e.id
            """,
            (int(request_id),),
        ).fetchall()
    finally:
        conn.close()
    return [
        {
            "action": row["action"],
            "from_status": row["from_status"],
            "to_status": row["to_status"],
            "created_at": row["created_at"],
            "actor_name": row["actor_name"] or "System",
        }
        for row in rows
    ]


def list_verified_agronomists(actor_id: int) -> list[dict]:
    conn = _connect()
    try:
        actor = _active_user(conn, actor_id)
        if actor is None or actor["role"] != "admin":
            raise PermissionError("Only an active admin can manage reviewers.")
        rows = conn.execute(
            """
            SELECT id, name, email, organization, qualifications
            FROM users
            WHERE role = 'agronomist' AND active = 1
                AND expert_verified = 1
            ORDER BY name COLLATE NOCASE, id
            """
        ).fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows]


def assign_review_request(
    actor_id: int,
    request_id: int,
    agronomist_id: int,
) -> str:
    """Assign or reassign a case to an active, verified agronomist."""

    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        actor = _active_user(conn, actor_id)
        if actor is None or actor["role"] != "admin":
            return "Only an active admin can assign review cases."
        expert = _verified_agronomist(conn, agronomist_id)
        if expert is None:
            return "Choose an active, admin-verified agronomist."
        request = conn.execute(
            """
            SELECT status, assigned_to, requester_user_id
            FROM review_requests WHERE id = ?
            """,
            (int(request_id),),
        ).fetchone()
        if request is None:
            return "That review request was not found."
        if request["status"] in ("reviewed", "closed"):
            return "A completed or closed case cannot be reassigned."
        if (
            request["requester_user_id"] is not None
            and int(request["requester_user_id"]) == int(agronomist_id)
        ):
            return "A requester cannot review their own case."
        old_status = request["status"]
        action = (
            "reassigned"
            if request["assigned_to"] is not None
            else "assigned"
        )
        conn.execute(
            """
            UPDATE review_requests
            SET assigned_to = ?, status = 'assigned', response = '',
                reviewed_at = NULL, closed_at = NULL, updated_at = ?
            WHERE id = ?
            """,
            (int(agronomist_id), _now(), int(request_id)),
        )
        _log_review_event(
            conn,
            request_id,
            actor_id,
            action,
            old_status,
            "assigned",
        )
        conn.commit()
    finally:
        conn.close()
    return ""


def start_review_request(actor_id: int, request_id: int) -> str:
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        if _verified_agronomist(conn, actor_id) is None:
            return "Only a verified agronomist can start a review."
        request = conn.execute(
            """
            SELECT status FROM review_requests
            WHERE id = ? AND assigned_to = ?
            """,
            (int(request_id), int(actor_id)),
        ).fetchone()
        if request is None:
            return "This case is not assigned to you."
        if request["status"] == "in_review":
            return ""
        if request["status"] != "assigned":
            return "Only an assigned case can be started."
        conn.execute(
            """
            UPDATE review_requests
            SET status = 'in_review', updated_at = ?
            WHERE id = ?
            """,
            (_now(), int(request_id)),
        )
        _log_review_event(
            conn,
            request_id,
            actor_id,
            "review_started",
            "assigned",
            "in_review",
        )
        conn.commit()
    finally:
        conn.close()
    return ""


def complete_review_request(
    actor_id: int,
    request_id: int,
    response: str,
) -> str:
    """Submit an expert response for a case assigned to the reviewer."""

    cleaned_response = _clean_note(response)
    if not cleaned_response:
        return "Enter the expert's review before completing the case."
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        if _verified_agronomist(conn, actor_id) is None:
            return "Only a verified agronomist can complete a review."
        request = conn.execute(
            """
            SELECT status FROM review_requests
            WHERE id = ? AND assigned_to = ?
            """,
            (int(request_id), int(actor_id)),
        ).fetchone()
        if request is None:
            return "This case is not assigned to you."
        if request["status"] != "in_review":
            return "Start the assigned case before submitting a review."
        now = _now()
        conn.execute(
            """
            UPDATE review_requests
            SET status = 'reviewed', response = ?, reviewed_at = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (cleaned_response, now, now, int(request_id)),
        )
        _log_review_event(
            conn,
            request_id,
            actor_id,
            "review_completed",
            "in_review",
            "reviewed",
        )
        conn.commit()
    finally:
        conn.close()
    return ""


def close_review_request(actor_id: int, request_id: int) -> str:
    """Allow an admin to close an uncompleted or abandoned case."""

    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        actor = _active_user(conn, actor_id)
        if actor is None or actor["role"] != "admin":
            return "Only an active admin can close a case."
        request = conn.execute(
            "SELECT status FROM review_requests WHERE id = ?",
            (int(request_id),),
        ).fetchone()
        if request is None:
            return "That review request was not found."
        if request["status"] == "reviewed":
            return "A reviewed case cannot be closed as unreviewed."
        if request["status"] == "closed":
            return ""
        now = _now()
        conn.execute(
            """
            UPDATE review_requests
            SET status = 'closed', closed_at = ?, updated_at = ?
            WHERE id = ?
            """,
            (now, now, int(request_id)),
        )
        _log_review_event(
            conn,
            request_id,
            actor_id,
            "case_closed",
            request["status"],
            "closed",
        )
        conn.commit()
    finally:
        conn.close()
    return ""


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
