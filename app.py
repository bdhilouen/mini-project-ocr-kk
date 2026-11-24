"""Flask web application for Indonesian Kartu Keluarga (KK) OCR and CRUD management."""

import os
import re
import sqlite3
from typing import Any, Dict, List, Optional
from uuid import uuid4

from flask import (
    Flask,
    abort,
    flash,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from models.database import (
    ANGGOTA_FIELDS,
    IDENTITAS_FIELDS,
    create_anggota,
    create_kk_with_anggota,
    delete_anggota,
    delete_kk,
    get_anggota_by_id,
    get_kk_by_id,
    init_db,
    list_all_kk,
    list_anggota_by_kk,
    update_anggota,
    update_kk,
)
from models.ocr_kk import empty_anggota, empty_identitas, extract_kk_from_image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

JENIS_KELAMIN_OPTIONS = ("LAKI-LAKI", "PEREMPUAN")
AGAMA_OPTIONS = (
    "ISLAM",
    "KRISTEN",
    "KATOLIK",
    "HINDU",
    "BUDDHA",
    "KONGHUCU",
    "KEPERCAYAAN TERHADAP TUHAN YME",
)
STATUS_PERKAWINAN_OPTIONS = (
    "BELUM KAWIN",
    "KAWIN",
    "KAWIN TERCATAT",
    "KAWIN BELUM TERCATAT",
    "CERAI HIDUP",
    "CERAI MATI",
)
HUBUNGAN_OPTIONS = (
    "KEPALA KELUARGA",
    "SUAMI",
    "ISTRI",
    "ANAK",
    "MENANTU",
    "CUCU",
    "ORANG TUA",
    "MERTUA",
    "FAMILI LAIN",
    "PEMBANTU",
    "LAINNYA",
)
KEWARGANEGARAAN_OPTIONS = ("WNI", "WNA")


def is_allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def is_valid_16_digits(value: str) -> bool:
    return bool(re.fullmatch(r"\d{16}", (value or "").strip()))


def validate_identitas(data: Dict[str, Any]) -> List[str]:
    """Validate required fields and 16-digit No. KK format."""
    errors: List[str] = []
    no_kk = (data.get("no_kk") or "").strip()
    nama_kk = (data.get("nama_kepala_keluarga") or "").strip()

    if not is_valid_16_digits(no_kk):
        errors.append("Nomor KK wajib berisi tepat 16 digit angka.")
    if not nama_kk:
        errors.append("Nama Kepala Keluarga wajib diisi.")
    return errors


def validate_anggota(data: Dict[str, Any], row_num: Optional[int] = None) -> List[str]:
    """Validate required fields and 16-digit NIK format for one family member."""
    prefix = f"Anggota #{row_num}: " if row_num is not None else ""
    errors: List[str] = []
    nik = (data.get("nik") or "").strip()
    nama = (data.get("nama_lengkap") or "").strip()

    if not is_valid_16_digits(nik):
        errors.append(f"{prefix}NIK wajib berisi tepat 16 digit angka.")
    if not nama:
        errors.append(f"{prefix}Nama Lengkap wajib diisi.")
    return errors


def _parse_identitas_form(form) -> Dict[str, str]:
    return {field: form.get(field, "").strip() for field in IDENTITAS_FIELDS}


def _parse_anggota_single_form(form) -> Dict[str, str]:
    data = {field: form.get(field, "").strip() for field in ANGGOTA_FIELDS}
    if not data["kewarganegaraan"]:
        data["kewarganegaraan"] = "WNI"
    return data


def _parse_anggota_list_form(form) -> List[Dict[str, str]]:
    columns = {field: form.getlist(f"anggota_{field}") for field in ANGGOTA_FIELDS}
    row_count = max((len(vals) for vals in columns.values()), default=0)
    rows: List[Dict[str, str]] = []
    for idx in range(row_count):
        row = {
            field: (columns[field][idx].strip() if idx < len(columns[field]) else "")
            for field in ANGGOTA_FIELDS
        }
        rows.append(row)
    return rows


def _is_blank_anggota(row: Dict[str, str]) -> bool:
    return not any(row.get(field, "").strip() for field in ANGGOTA_FIELDS if field != "kewarganegaraan")


def create_app(test_config: Optional[Dict[str, Any]] = None) -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.config.from_mapping(
        SECRET_KEY=os.environ.get("SECRET_KEY", "dev-secret-key-kk-ocr"),
        DATABASE_PATH=os.path.join(BASE_DIR, "kk_database.db"),
        UPLOAD_FOLDER=os.path.join(BASE_DIR, "uploads"),
        MAX_CONTENT_LENGTH=10 * 1024 * 1024,
    )
    if test_config:
        app.config.update(test_config)

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    init_db(app.config["DATABASE_PATH"])

    @app.context_processor
    def inject_options():
        return {
            "jenis_kelamin_options": JENIS_KELAMIN_OPTIONS,
            "agama_options": AGAMA_OPTIONS,
            "status_perkawinan_options": STATUS_PERKAWINAN_OPTIONS,
            "hubungan_options": HUBUNGAN_OPTIONS,
            "kewarganegaraan_options": KEWARGANEGARAAN_OPTIONS,
        }

    @app.errorhandler(RequestEntityTooLarge)
    def handle_file_too_large(_err):
        flash("Ukuran gambar melebihi batas maksimal 10 MB.", "danger")
        return redirect(url_for("index"))

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/upload", methods=["POST"])
    def upload_kk():
        uploaded = request.files.get("kk_image")
        if not uploaded or not uploaded.filename:
            flash("Pilih file gambar Kartu Keluarga terlebih dahulu.", "danger")
            return redirect(url_for("index"))

        if not is_allowed_file(uploaded.filename):
            flash("Format file tidak didukung. Gunakan PNG, JPG, JPEG, atau WEBP.", "danger")
            return redirect(url_for("index"))

        safe_name = secure_filename(uploaded.filename) or "kk_image.png"
        stored_filename = f"{uuid4().hex}_{safe_name}"
        saved_path = os.path.join(app.config["UPLOAD_FOLDER"], stored_filename)
        uploaded.save(saved_path)

        ocr_result = extract_kk_from_image(saved_path)
        if ocr_result.get("error"):
            flash(
                f"Proses OCR mengalami kendala ({ocr_result['error']}). "
                "Anda tetap dapat mengisi dan mengoreksi data secara manual.",
                "warning",
            )
        elif not ocr_result["identitas"].get("no_kk") and not ocr_result.get("anggota"):
            flash(
                "Sebagian data belum terdeteksi otomatis oleh OCR. "
                "Silakan periksa pratinjau gambar dan lengkapi kolom yang kosong.",
                "warning",
            )
        else:
            flash("Hasil OCR berhasil dimuat. Periksa dan koreksi bila ada kekeliruan sebelum menyimpan.", "info")

        anggota_list = ocr_result.get("anggota") or [empty_anggota()]
        return render_template(
            "konfirmasi.html",
            identitas=ocr_result.get("identitas") or empty_identitas(),
            anggota_list=anggota_list,
            image_filename=stored_filename,
            raw_text=ocr_result.get("raw_text", ""),
            errors=[],
        )

    @app.route("/konfirmasi", methods=["GET"])
    def konfirmasi_manual():
        return render_template(
            "konfirmasi.html",
            identitas=empty_identitas(),
            anggota_list=[empty_anggota()],
            image_filename="",
            raw_text="",
            errors=[],
        )

    @app.route("/simpan", methods=["POST"])
    def simpan_kk():
        db_path = app.config["DATABASE_PATH"]
        identitas = _parse_identitas_form(request.form)
        submitted_rows = _parse_anggota_list_form(request.form)
        image_filename = request.form.get("image_filename", "").strip()
        raw_text = request.form.get("raw_text", "")

        non_blank_rows = [row for row in submitted_rows if not _is_blank_anggota(row)]
        for row in non_blank_rows:
            if not row["kewarganegaraan"]:
                row["kewarganegaraan"] = "WNI"

        errors = validate_identitas(identitas)
        for idx, member in enumerate(non_blank_rows, start=1):
            errors.extend(validate_anggota(member, row_num=idx))

        if errors:
            display_rows = non_blank_rows if non_blank_rows else (submitted_rows or [empty_anggota()])
            return (
                render_template(
                    "konfirmasi.html",
                    identitas=identitas,
                    anggota_list=display_rows,
                    image_filename=image_filename,
                    raw_text=raw_text,
                    errors=errors,
                ),
                400,
            )

        try:
            new_kk_id = create_kk_with_anggota(db_path, identitas, non_blank_rows)
        except sqlite3.IntegrityError:
            display_rows = non_blank_rows if non_blank_rows else [empty_anggota()]
            return (
                render_template(
                    "konfirmasi.html",
                    identitas=identitas,
                    anggota_list=display_rows,
                    image_filename=image_filename,
                    raw_text=raw_text,
                    errors=[
                        "Nomor KK sudah terdaftar di database. "
                        "Gunakan Nomor KK berbeda atau perbarui data yang ada."
                    ],
                ),
                400,
            )

        flash("Data Kartu Keluarga dan anggota keluarga berhasil disimpan.", "success")
        return redirect(url_for("detail_kk", kk_id=new_kk_id))

    @app.route("/kk")
    def daftar_kk():
        kk_list = list_all_kk(app.config["DATABASE_PATH"])
        return render_template("daftar_kk.html", kk_list=kk_list)

    @app.route("/kk/<int:kk_id>")
    def detail_kk(kk_id: int):
        db_path = app.config["DATABASE_PATH"]
        kk = get_kk_by_id(db_path, kk_id)
        if kk is None:
            abort(404)
        anggota_list = list_anggota_by_kk(db_path, kk_id)
        return render_template("detail_kk.html", kk=kk, anggota_list=anggota_list)

    @app.route("/kk/<int:kk_id>/edit", methods=["GET", "POST"])
    def edit_kk(kk_id: int):
        db_path = app.config["DATABASE_PATH"]
        kk = get_kk_by_id(db_path, kk_id)
        if kk is None:
            abort(404)

        if request.method == "POST":
            identitas = _parse_identitas_form(request.form)
            errors = validate_identitas(identitas)
            if errors:
                return render_template("edit_kk.html", kk=kk, identitas=identitas, errors=errors), 400
            try:
                update_kk(db_path, kk_id, identitas)
            except sqlite3.IntegrityError:
                return (
                    render_template(
                        "edit_kk.html",
                        kk=kk,
                        identitas=identitas,
                        errors=["Nomor KK sudah digunakan oleh data lain."],
                    ),
                    400,
                )
            flash("Identitas Kartu Keluarga berhasil diperbarui.", "success")
            return redirect(url_for("detail_kk", kk_id=kk_id))

        return render_template("edit_kk.html", kk=kk, identitas=kk, errors=[])

    @app.route("/kk/<int:kk_id>/hapus", methods=["POST"])
    def hapus_kk(kk_id: int):
        deleted = delete_kk(app.config["DATABASE_PATH"], kk_id)
        if not deleted:
            abort(404)
        flash("Data Kartu Keluarga beserta seluruh anggota keluarganya berhasil dihapus.", "success")
        return redirect(url_for("daftar_kk"))

    @app.route("/kk/<int:kk_id>/anggota/tambah", methods=["GET", "POST"])
    def tambah_anggota(kk_id: int):
        db_path = app.config["DATABASE_PATH"]
        kk = get_kk_by_id(db_path, kk_id)
        if kk is None:
            abort(404)

        if request.method == "POST":
            data = _parse_anggota_single_form(request.form)
            errors = validate_anggota(data)
            if errors:
                return (
                    render_template("anggota_form.html", mode="tambah", kk=kk, anggota=data, errors=errors),
                    400,
                )
            create_anggota(db_path, kk_id, data)
            flash("Anggota keluarga berhasil ditambahkan.", "success")
            return redirect(url_for("detail_kk", kk_id=kk_id))

        return render_template("anggota_form.html", mode="tambah", kk=kk, anggota=empty_anggota(), errors=[])

    @app.route("/kk/<int:kk_id>/anggota/<int:anggota_id>/edit", methods=["GET", "POST"])
    def edit_anggota(kk_id: int, anggota_id: int):
        db_path = app.config["DATABASE_PATH"]
        kk = get_kk_by_id(db_path, kk_id)
        anggota = get_anggota_by_id(db_path, kk_id, anggota_id)
        if kk is None or anggota is None:
            abort(404)

        if request.method == "POST":
            data = _parse_anggota_single_form(request.form)
            errors = validate_anggota(data)
            if errors:
                data["id"] = anggota_id
                return (
                    render_template("anggota_form.html", mode="edit", kk=kk, anggota=data, errors=errors),
                    400,
                )
            update_anggota(db_path, kk_id, anggota_id, data)
            flash("Data anggota keluarga berhasil diperbarui.", "success")
            return redirect(url_for("detail_kk", kk_id=kk_id))

        return render_template("anggota_form.html", mode="edit", kk=kk, anggota=anggota, errors=[])

    @app.route("/kk/<int:kk_id>/anggota/<int:anggota_id>/hapus", methods=["POST"])
    def hapus_anggota(kk_id: int, anggota_id: int):
        deleted = delete_anggota(app.config["DATABASE_PATH"], kk_id, anggota_id)
        if not deleted:
            abort(404)
        flash("Anggota keluarga berhasil dihapus.", "success")
        return redirect(url_for("detail_kk", kk_id=kk_id))

    @app.route("/uploads/<path:filename>")
    def uploaded_file(filename: str):
        return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
