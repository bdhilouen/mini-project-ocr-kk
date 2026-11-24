"""SQLite database schema and CRUD helpers for Kartu Keluarga (KK) records."""

from contextlib import contextmanager
import os
import sqlite3
from typing import Any, Dict, Iterable, List, Optional

IDENTITAS_FIELDS = (
    "no_kk",
    "nama_kepala_keluarga",
    "alamat",
    "rt_rw",
    "desa_kelurahan",
    "kecamatan",
    "kabupaten_kota",
    "kode_pos",
    "provinsi",
)

ANGGOTA_FIELDS = (
    "nik",
    "nama_lengkap",
    "jenis_kelamin",
    "tempat_lahir",
    "tanggal_lahir",
    "agama",
    "pendidikan",
    "jenis_pekerjaan",
    "status_perkawinan",
    "status_hubungan_dalam_keluarga",
    "kewarganegaraan",
    "nama_ayah",
    "nama_ibu",
)


def get_connection(db_path: str) -> sqlite3.Connection:
    """Open a SQLite connection with row dict access and foreign keys enabled."""
    parent = os.path.dirname(os.path.abspath(db_path)) if db_path != ":memory:" else ""
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


@contextmanager
def _managed_connection(db_path: str):
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: str) -> None:
    """Create tbl_identitasKK and tbl_anggotaKeluargaKK if they do not exist."""
    with _managed_connection(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS tbl_identitasKK (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                no_kk TEXT NOT NULL UNIQUE,
                nama_kepala_keluarga TEXT NOT NULL,
                alamat TEXT DEFAULT '',
                rt_rw TEXT DEFAULT '',
                desa_kelurahan TEXT DEFAULT '',
                kecamatan TEXT DEFAULT '',
                kabupaten_kota TEXT DEFAULT '',
                kode_pos TEXT DEFAULT '',
                provinsi TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS tbl_anggotaKeluargaKK (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                id_kk INTEGER NOT NULL,
                nik TEXT NOT NULL,
                nama_lengkap TEXT NOT NULL,
                jenis_kelamin TEXT DEFAULT '',
                tempat_lahir TEXT DEFAULT '',
                tanggal_lahir TEXT DEFAULT '',
                agama TEXT DEFAULT '',
                pendidikan TEXT DEFAULT '',
                jenis_pekerjaan TEXT DEFAULT '',
                status_perkawinan TEXT DEFAULT '',
                status_hubungan_dalam_keluarga TEXT DEFAULT '',
                kewarganegaraan TEXT DEFAULT 'WNI',
                nama_ayah TEXT DEFAULT '',
                nama_ibu TEXT DEFAULT '',
                FOREIGN KEY (id_kk) REFERENCES tbl_identitasKK(id) ON DELETE CASCADE
            );
            """
        )


def _clean_record(data: Dict[str, Any], fields: Iterable[str], defaults: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    defaults = defaults or {}
    cleaned: Dict[str, str] = {}
    for field in fields:
        raw = data.get(field)
        value = str(raw).strip() if raw is not None else ""
        if not value and field in defaults and field not in data:
            value = defaults[field]
        cleaned[field] = value
    return cleaned


def create_kk_with_anggota(
    db_path: str,
    identitas: Dict[str, Any],
    anggota_list: List[Dict[str, Any]],
) -> int:
    """Insert a KK identity record and all associated family members in one transaction."""
    clean_kk = _clean_record(identitas, IDENTITAS_FIELDS)
    kk_cols = ", ".join(IDENTITAS_FIELDS)
    kk_placeholders = ", ".join("?" for _ in IDENTITAS_FIELDS)

    anggota_cols = ", ".join(("id_kk",) + ANGGOTA_FIELDS)
    anggota_placeholders = ", ".join("?" for _ in range(1 + len(ANGGOTA_FIELDS)))

    with _managed_connection(db_path) as conn:
        cur = conn.execute(
            f"INSERT INTO tbl_identitasKK ({kk_cols}) VALUES ({kk_placeholders})",
            tuple(clean_kk[f] for f in IDENTITAS_FIELDS),
        )
        kk_id = int(cur.lastrowid)

        for member in anggota_list:
            clean_member = _clean_record(member, ANGGOTA_FIELDS, defaults={"kewarganegaraan": "WNI"})
            conn.execute(
                f"INSERT INTO tbl_anggotaKeluargaKK ({anggota_cols}) VALUES ({anggota_placeholders})",
                (kk_id,) + tuple(clean_member[f] for f in ANGGOTA_FIELDS),
            )
        return kk_id


def list_all_kk(db_path: str) -> List[Dict[str, Any]]:
    """Return all KK identity records ordered by newest first."""
    with _managed_connection(db_path) as conn:
        rows = conn.execute("SELECT * FROM tbl_identitasKK ORDER BY id DESC").fetchall()
        return [dict(row) for row in rows]


def get_kk_by_id(db_path: str, kk_id: int) -> Optional[Dict[str, Any]]:
    """Return a single KK identity record by primary key, or None if not found."""
    with _managed_connection(db_path) as conn:
        row = conn.execute("SELECT * FROM tbl_identitasKK WHERE id = ?", (kk_id,)).fetchone()
        return dict(row) if row else None


def update_kk(db_path: str, kk_id: int, identitas: Dict[str, Any]) -> bool:
    """Update a KK identity record by ID. Returns True if a row was updated."""
    clean_kk = _clean_record(identitas, IDENTITAS_FIELDS)
    assignments = ", ".join(f"{field} = ?" for field in IDENTITAS_FIELDS)
    with _managed_connection(db_path) as conn:
        cur = conn.execute(
            f"UPDATE tbl_identitasKK SET {assignments} WHERE id = ?",
            tuple(clean_kk[f] for f in IDENTITAS_FIELDS) + (kk_id,),
        )
        return cur.rowcount > 0


def delete_kk(db_path: str, kk_id: int) -> bool:
    """Delete a KK identity record (and cascade-delete its family members)."""
    with _managed_connection(db_path) as conn:
        cur = conn.execute("DELETE FROM tbl_identitasKK WHERE id = ?", (kk_id,))
        return cur.rowcount > 0


def list_anggota_by_kk(db_path: str, kk_id: int) -> List[Dict[str, Any]]:
    """Return all family members belonging to a KK ordered by ID."""
    with _managed_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM tbl_anggotaKeluargaKK WHERE id_kk = ? ORDER BY id ASC",
            (kk_id,),
        ).fetchall()
        return [dict(row) for row in rows]


def get_anggota_by_id(db_path: str, kk_id: int, anggota_id: int) -> Optional[Dict[str, Any]]:
    """Return one family member scoped to the specified KK ID."""
    with _managed_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM tbl_anggotaKeluargaKK WHERE id = ? AND id_kk = ?",
            (anggota_id, kk_id),
        ).fetchone()
        return dict(row) if row else None


def create_anggota(db_path: str, kk_id: int, data: Dict[str, Any]) -> int:
    """Insert a new family member for an existing KK and return the new ID."""
    clean_member = _clean_record(data, ANGGOTA_FIELDS, defaults={"kewarganegaraan": "WNI"})
    cols = ", ".join(("id_kk",) + ANGGOTA_FIELDS)
    placeholders = ", ".join("?" for _ in range(1 + len(ANGGOTA_FIELDS)))
    with _managed_connection(db_path) as conn:
        cur = conn.execute(
            f"INSERT INTO tbl_anggotaKeluargaKK ({cols}) VALUES ({placeholders})",
            (kk_id,) + tuple(clean_member[f] for f in ANGGOTA_FIELDS),
        )
        return int(cur.lastrowid)


def update_anggota(db_path: str, kk_id: int, anggota_id: int, data: Dict[str, Any]) -> bool:
    """Update a family member scoped to kk_id. Returns True if a row was updated."""
    clean_member = _clean_record(data, ANGGOTA_FIELDS, defaults={"kewarganegaraan": "WNI"})
    assignments = ", ".join(f"{field} = ?" for field in ANGGOTA_FIELDS)
    with _managed_connection(db_path) as conn:
        cur = conn.execute(
            f"UPDATE tbl_anggotaKeluargaKK SET {assignments} WHERE id = ? AND id_kk = ?",
            tuple(clean_member[f] for f in ANGGOTA_FIELDS) + (anggota_id, kk_id),
        )
        return cur.rowcount > 0


def delete_anggota(db_path: str, kk_id: int, anggota_id: int) -> bool:
    """Delete a family member scoped to kk_id. Returns True if a row was deleted."""
    with _managed_connection(db_path) as conn:
        cur = conn.execute(
            "DELETE FROM tbl_anggotaKeluargaKK WHERE id = ? AND id_kk = ?",
            (anggota_id, kk_id),
        )
        return cur.rowcount > 0
