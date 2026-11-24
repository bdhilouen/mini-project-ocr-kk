"""Unit and integration tests for Kartu Keluarga OCR web application."""

import io
import os
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from app import create_app
from models.database import (
    create_anggota,
    create_kk_with_anggota,
    delete_anggota,
    delete_kk,
    get_anggota_by_id,
    get_connection,
    get_kk_by_id,
    init_db,
    list_all_kk,
    list_anggota_by_kk,
    update_anggota,
    update_kk,
)
from models.ocr_kk import parse_kk_ocr_results


def _bbox(x1: float, y1: float, x2: float, y2: float):
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


class DatabaseCrudTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmpdir.name, "test.db")
        init_db(self.db_path)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_schema_and_foreign_keys_enabled(self):
        conn = get_connection(self.db_path)
        try:
            fk_on = conn.execute("PRAGMA foreign_keys;").fetchone()[0]
            self.assertEqual(fk_on, 1)
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            self.assertIn("tbl_identitasKK", tables)
            self.assertIn("tbl_anggotaKeluargaKK", tables)
        finally:
            conn.close()

    def test_crud_and_cascade_delete(self):
        identitas = {
            "no_kk": "3273012345678901",
            "nama_kepala_keluarga": "BUDI SANTOSO",
            "alamat": "JL. MERDEKA NO. 10",
            "rt_rw": "001/002",
            "desa_kelurahan": "CIHAPIT",
            "kecamatan": "BANDUNG WETAN",
            "kabupaten_kota": "KOTA BANDUNG",
            "kode_pos": "40114",
            "provinsi": "JAWA BARAT",
        }
        members = [
            {
                "nik": "3273011111111111",
                "nama_lengkap": "BUDI SANTOSO",
                "jenis_kelamin": "LAKI-LAKI",
                "tempat_lahir": "BANDUNG",
                "tanggal_lahir": "10-05-1980",
                "agama": "ISLAM",
                "pendidikan": "SLTA/SEDERAJAT",
                "jenis_pekerjaan": "KARYAWAN SWASTA",
                "status_perkawinan": "KAWIN",
                "status_hubungan_dalam_keluarga": "KEPALA KELUARGA",
                "kewarganegaraan": "WNI",
                "nama_ayah": "SLAMET",
                "nama_ibu": "SITI",
            },
            {
                "nik": "3273012222222222",
                "nama_lengkap": "ANI WIJAYA",
                "jenis_kelamin": "PEREMPUAN",
                "status_hubungan_dalam_keluarga": "ISTRI",
                "kewarganegaraan": "WNI",
            },
        ]

        kk_id = create_kk_with_anggota(self.db_path, identitas, members)
        self.assertEqual(len(list_all_kk(self.db_path)), 1)
        self.assertEqual(len(list_anggota_by_kk(self.db_path, kk_id)), 2)

        identitas["alamat"] = "JL. ASIA AFRIKA NO. 5"
        self.assertTrue(update_kk(self.db_path, kk_id, identitas))
        self.assertEqual(get_kk_by_id(self.db_path, kk_id)["alamat"], "JL. ASIA AFRIKA NO. 5")

        new_member_id = create_anggota(
            self.db_path,
            kk_id,
            {"nik": "3273013333333333", "nama_lengkap": "CITRA SANTOSO", "kewarganegaraan": "WNI"},
        )
        member = get_anggota_by_id(self.db_path, kk_id, new_member_id)
        self.assertEqual(member["nama_lengkap"], "CITRA SANTOSO")

        member["nama_lengkap"] = "CITRA KUSUMA"
        self.assertTrue(update_anggota(self.db_path, kk_id, new_member_id, member))
        self.assertEqual(get_anggota_by_id(self.db_path, kk_id, new_member_id)["nama_lengkap"], "CITRA KUSUMA")

        self.assertTrue(delete_anggota(self.db_path, kk_id, new_member_id))
        self.assertEqual(len(list_anggota_by_kk(self.db_path, kk_id)), 2)

        self.assertTrue(delete_kk(self.db_path, kk_id))
        self.assertIsNone(get_kk_by_id(self.db_path, kk_id))
        self.assertEqual(list_anggota_by_kk(self.db_path, kk_id), [])


class OcrParserTests(unittest.TestCase):
    def test_spatial_kk_parser_header_and_two_tables(self):
        raw_results = [
            # Top title & No KK (with minor OCR 'O' noise)
            (_bbox(650, 40, 1150, 80), "KARTU KELUARGA", 0.99),
            (_bbox(680, 90, 1120, 125), "No. 3273O12345678901", 0.95),
            # 2-column header
            (_bbox(80, 160, 320, 185), "Nama Kepala Keluarga", 0.96),
            (_bbox(340, 160, 620, 185), ": BUDI SANTOSO", 0.96),
            (_bbox(1020, 160, 1200, 185), "Kecamatan", 0.96),
            (_bbox(1250, 160, 1550, 185), ": BANDUNG WETAN", 0.96),
            (_bbox(80, 195, 220, 220), "Alamat", 0.96),
            (_bbox(340, 195, 700, 220), ": JL. MERDEKA NO. 10", 0.96),
            (_bbox(1020, 195, 1240, 220), "Kabupaten/Kota", 0.96),
            (_bbox(1250, 195, 1550, 220), ": KOTA BANDUNG", 0.96),
            (_bbox(80, 230, 220, 255), "RT/RW", 0.96),
            (_bbox(340, 230, 500, 255), ": 1/2", 0.96),
            (_bbox(1020, 230, 1180, 255), "Kode Pos", 0.96),
            (_bbox(1250, 230, 1420, 255), ": 40114", 0.96),
            (_bbox(80, 265, 280, 290), "Desa/Kelurahan", 0.96),
            (_bbox(340, 265, 560, 290), ": CIHAPIT", 0.96),
            (_bbox(1020, 265, 1180, 290), "Provinsi", 0.96),
            (_bbox(1250, 265, 1520, 290), ": JAWA BARAT", 0.96),
            # Table 1 header
            (_bbox(60, 340, 100, 365), "No", 0.96),
            (_bbox(120, 340, 320, 365), "Nama Lengkap", 0.96),
            (_bbox(350, 340, 560, 365), "NIK", 0.96),
            (_bbox(580, 340, 740, 365), "Jenis Kelamin", 0.96),
            (_bbox(760, 340, 920, 365), "Tempat Lahir", 0.96),
            (_bbox(940, 340, 1100, 365), "Tanggal Lahir", 0.96),
            (_bbox(1120, 340, 1240, 365), "Agama", 0.96),
            (_bbox(1260, 340, 1460, 365), "Pendidikan", 0.96),
            (_bbox(1480, 340, 1740, 365), "Jenis Pekerjaan", 0.96),
            # Table 1 column index row (should be skipped)
            (_bbox(190, 375, 230, 395), "(1)", 0.9),
            (_bbox(430, 375, 470, 395), "(2)", 0.9),
            (_bbox(640, 375, 680, 395), "(3)", 0.9),
            # Table 1 data rows
            (_bbox(65, 410, 95, 435), "1", 0.95),
            (_bbox(120, 410, 320, 435), "BUDI SANTOSO", 0.97),
            (_bbox(350, 410, 560, 435), "3273O11111111111", 0.94),
            (_bbox(580, 410, 740, 435), "LAKI-LAKI", 0.98),
            (_bbox(760, 410, 920, 435), "BANDUNG", 0.97),
            (_bbox(940, 410, 1100, 435), "10/05/1980", 0.96),
            (_bbox(1120, 410, 1240, 435), "ISLAM", 0.98),
            (_bbox(1260, 410, 1460, 435), "DIPLOMA IV/STRATA I", 0.95),
            (_bbox(1480, 410, 1740, 435), "KARYAWAN SWASTA", 0.95),
            (_bbox(65, 450, 95, 475), "2", 0.95),
            (_bbox(120, 450, 320, 475), "ANI WIJAYA", 0.97),
            (_bbox(350, 450, 560, 475), "3273012222222222", 0.97),
            (_bbox(580, 450, 740, 475), "PEREMPUAN", 0.98),
            (_bbox(760, 450, 920, 475), "JAKARTA", 0.97),
            (_bbox(940, 450, 1100, 475), "14-08-1982", 0.96),
            (_bbox(1120, 450, 1240, 475), "ISLAM", 0.98),
            (_bbox(1260, 450, 1460, 475), "SLTA/SEDERAJAT", 0.95),
            (_bbox(1480, 450, 1740, 475), "MENGURUS RUMAH TANGGA", 0.95),
            # Table 2 header
            (_bbox(60, 540, 100, 565), "No", 0.95),
            (_bbox(120, 540, 320, 565), "Status Perkawinan", 0.95),
            (_bbox(340, 540, 520, 565), "Tanggal Perkawinan", 0.95),
            (_bbox(540, 540, 820, 565), "Status Hubungan Dalam Keluarga", 0.95),
            (_bbox(840, 540, 1020, 565), "Kewarganegaraan", 0.95),
            (_bbox(1040, 540, 1180, 565), "No. Paspor", 0.95),
            (_bbox(1200, 540, 1340, 565), "No. KITAP", 0.95),
            (_bbox(1360, 540, 1540, 565), "Ayah", 0.95),
            (_bbox(1560, 540, 1740, 565), "Ibu", 0.95),
            # Table 2 rows
            (_bbox(65, 585, 95, 610), "1", 0.95),
            (_bbox(120, 585, 320, 610), "KAWIN", 0.96),
            (_bbox(340, 585, 520, 610), "05-05-2008", 0.96),
            (_bbox(540, 585, 820, 610), "KEPALA KELUARGA", 0.96),
            (_bbox(840, 585, 1020, 610), "WNI", 0.97),
            (_bbox(1040, 585, 1180, 610), "-", 0.95),
            (_bbox(1200, 585, 1340, 610), "-", 0.95),
            (_bbox(1360, 585, 1540, 610), "SLAMET", 0.97),
            (_bbox(1560, 585, 1740, 610), "SITI", 0.97),
            (_bbox(65, 625, 95, 650), "2", 0.95),
            (_bbox(120, 625, 320, 650), "KAWIN", 0.96),
            (_bbox(340, 625, 520, 650), "05-05-2008", 0.96),
            (_bbox(540, 625, 820, 650), "ISTRI", 0.96),
            (_bbox(840, 625, 1020, 650), "WNI", 0.97),
            (_bbox(1040, 625, 1180, 650), "-", 0.95),
            (_bbox(1200, 625, 1340, 650), "-", 0.95),
            (_bbox(1360, 625, 1540, 650), "HENDRA", 0.97),
            (_bbox(1560, 625, 1740, 650), "RATNA", 0.97),
        ]

        parsed = parse_kk_ocr_results(raw_results, image_width=1800, image_height=1000)
        identitas = parsed["identitas"]
        self.assertEqual(identitas["no_kk"], "3273012345678901")
        self.assertEqual(identitas["nama_kepala_keluarga"], "BUDI SANTOSO")
        self.assertEqual(identitas["alamat"], "JL. MERDEKA NO. 10")
        self.assertEqual(identitas["rt_rw"], "001/002")
        self.assertEqual(identitas["desa_kelurahan"], "CIHAPIT")
        self.assertEqual(identitas["kecamatan"], "BANDUNG WETAN")
        self.assertEqual(identitas["kabupaten_kota"], "KOTA BANDUNG")
        self.assertEqual(identitas["kode_pos"], "40114")
        self.assertEqual(identitas["provinsi"], "JAWA BARAT")

        self.assertEqual(len(parsed["anggota"]), 2)
        m1, m2 = parsed["anggota"]
        self.assertEqual(m1["nik"], "3273011111111111")
        self.assertEqual(m1["nama_lengkap"], "BUDI SANTOSO")
        self.assertEqual(m1["tanggal_lahir"], "10-05-1980")
        self.assertEqual(m1["status_hubungan_dalam_keluarga"], "KEPALA KELUARGA")
        self.assertEqual(m1["nama_ayah"], "SLAMET")
        self.assertEqual(m1["nama_ibu"], "SITI")

        self.assertEqual(m2["nik"], "3273012222222222")
        self.assertEqual(m2["jenis_kelamin"], "PEREMPUAN")
        self.assertEqual(m2["status_hubungan_dalam_keluarga"], "ISTRI")
        self.assertEqual(m2["nama_ayah"], "HENDRA")
        self.assertEqual(m2["nama_ibu"], "RATNA")

    def test_header_slash_confusion_variants(self):
        raw_results = [
            (_bbox(680, 70, 1120, 110), "No. 3273012345678901", 0.99),
            (_bbox(60, 140, 280, 165), "Alamat", 0.96),
            (_bbox(320, 140, 620, 165), ": JL. MERDEKA NO. 10", 0.96),
            (_bbox(980, 140, 1220, 165), "KabupatenlKota : KOTA BANDUNG", 0.96),
            (_bbox(60, 176, 520, 200), "RTIRW : 001,002", 0.96),
            (_bbox(60, 212, 520, 236), "DesalKelurahan CIHAPIT", 0.96),
        ]
        parsed = parse_kk_ocr_results(raw_results, image_width=1800, image_height=1000)
        ident = parsed["identitas"]
        self.assertEqual(ident["alamat"], "JL. MERDEKA NO. 10")
        self.assertEqual(ident["rt_rw"], "001/002")
        self.assertEqual(ident["desa_kelurahan"], "CIHAPIT")
        self.assertEqual(ident["kabupaten_kota"], "KOTA BANDUNG")


class FlaskAppRoutesTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmpdir.name, "app_test.db")
        self.upload_dir = os.path.join(self.tmpdir.name, "uploads")
        self.app = create_app({
            "TESTING": True,
            "DATABASE_PATH": self.db_path,
            "UPLOAD_FOLDER": self.upload_dir,
        })
        self.client = self.app.test_client()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_upload_validation_and_full_crud_flow(self):
        # 1. Index & manual confirmation page render cleanly.
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/konfirmasi").status_code, 200)

        # 2. Reject invalid file extension.
        bad_resp = self.client.post(
            "/upload",
            data={"kk_image": (io.BytesIO(b"not an image"), "notes.txt")},
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        self.assertEqual(bad_resp.status_code, 200)
        self.assertIn(b"Format file tidak didukung", bad_resp.data)

        # 3. Upload valid PNG with mocked OCR extractor.
        img_buf = io.BytesIO()
        Image.new("RGB", (200, 100), color=(255, 255, 255)).save(img_buf, format="PNG")
        img_buf.seek(0)
        mocked_ocr = {
            "identitas": {
                "no_kk": "3273012345678901",
                "nama_kepala_keluarga": "BUDI SANTOSO",
                "alamat": "JL. MERDEKA NO. 10",
                "rt_rw": "001/002",
                "desa_kelurahan": "CIHAPIT",
                "kecamatan": "BANDUNG WETAN",
                "kabupaten_kota": "KOTA BANDUNG",
                "kode_pos": "40114",
                "provinsi": "JAWA BARAT",
            },
            "anggota": [
                {
                    "nik": "3273011111111111",
                    "nama_lengkap": "BUDI SANTOSO",
                    "jenis_kelamin": "LAKI-LAKI",
                    "tempat_lahir": "BANDUNG",
                    "tanggal_lahir": "10-05-1980",
                    "agama": "ISLAM",
                    "pendidikan": "S1",
                    "jenis_pekerjaan": "WIRASWASTA",
                    "status_perkawinan": "KAWIN",
                    "status_hubungan_dalam_keluarga": "KEPALA KELUARGA",
                    "kewarganegaraan": "WNI",
                    "nama_ayah": "SLAMET",
                    "nama_ibu": "SITI",
                }
            ],
            "raw_text": "KARTU KELUARGA 3273012345678901",
            "error": None,
        }
        with patch("app.extract_kk_from_image", return_value=mocked_ocr):
            upload_resp = self.client.post(
                "/upload",
                data={"kk_image": (img_buf, "kk_sample.png")},
                content_type="multipart/form-data",
            )
        self.assertEqual(upload_resp.status_code, 200)
        self.assertIn(b"3273012345678901", upload_resp.data)
        self.assertIn(b"BUDI SANTOSO", upload_resp.data)

        # 4. Validation rejects invalid 16-digit No KK & NIK while preserving input.
        invalid_save = self.client.post(
            "/simpan",
            data={
                "no_kk": "12345ABC",
                "nama_kepala_keluarga": "BUDI SANTOSO",
                "alamat": "JL. MERDEKA NO. 10",
                "anggota_nik": ["999"],
                "anggota_nama_lengkap": ["BUDI SANTOSO"],
                "anggota_kewarganegaraan": ["WNI"],
            },
        )
        self.assertEqual(invalid_save.status_code, 400)
        self.assertIn(b"Nomor KK wajib berisi tepat 16 digit angka", invalid_save.data)
        self.assertIn(b"Anggota #1: NIK wajib berisi tepat 16 digit angka", invalid_save.data)
        self.assertIn(b"JL. MERDEKA NO. 10", invalid_save.data)

        # 5. Valid save stores KK + anggota and redirects to detail page.
        valid_save = self.client.post(
            "/simpan",
            data={
                "no_kk": "3273012345678901",
                "nama_kepala_keluarga": "BUDI SANTOSO",
                "alamat": "JL. MERDEKA NO. 10",
                "rt_rw": "001/002",
                "desa_kelurahan": "CIHAPIT",
                "kecamatan": "BANDUNG WETAN",
                "kabupaten_kota": "KOTA BANDUNG",
                "kode_pos": "40114",
                "provinsi": "JAWA BARAT",
                "anggota_nik": ["3273011111111111"],
                "anggota_nama_lengkap": ["BUDI SANTOSO"],
                "anggota_jenis_kelamin": ["LAKI-LAKI"],
                "anggota_tempat_lahir": ["BANDUNG"],
                "anggota_tanggal_lahir": ["10-05-1980"],
                "anggota_agama": ["ISLAM"],
                "anggota_pendidikan": ["S1"],
                "anggota_jenis_pekerjaan": ["WIRASWASTA"],
                "anggota_status_perkawinan": ["KAWIN"],
                "anggota_status_hubungan_dalam_keluarga": ["KEPALA KELUARGA"],
                "anggota_kewarganegaraan": ["WNI"],
                "anggota_nama_ayah": ["SLAMET"],
                "anggota_nama_ibu": ["SITI"],
            },
        )
        self.assertEqual(valid_save.status_code, 302)
        detail_url = valid_save.headers["Location"]

        # 6. Daftar KK lists only tbl_identitasKK (not member NIKs).
        daftar_resp = self.client.get("/kk")
        self.assertEqual(daftar_resp.status_code, 200)
        self.assertIn(b"3273012345678901", daftar_resp.data)
        self.assertNotIn(b"3273011111111111", daftar_resp.data)

        # 7. Detail KK shows both identity and family member.
        detail_resp = self.client.get(detail_url)
        self.assertEqual(detail_resp.status_code, 200)
        self.assertIn(b"3273011111111111", detail_resp.data)

        # 8. Edit KK identity.
        kk_id = list_all_kk(self.db_path)[0]["id"]
        edit_kk_resp = self.client.post(
            f"/kk/{kk_id}/edit",
            data={
                "no_kk": "3273012345678901",
                "nama_kepala_keluarga": "BUDI SANTOSO SH",
                "alamat": "JL. DAGO NO. 1",
                "rt_rw": "003/004",
                "desa_kelurahan": "LEBAK GEDE",
                "kecamatan": "COBLONG",
                "kabupaten_kota": "KOTA BANDUNG",
                "kode_pos": "40132",
                "provinsi": "JAWA BARAT",
            },
            follow_redirects=True,
        )
        self.assertEqual(edit_kk_resp.status_code, 200)
        self.assertIn(b"BUDI SANTOSO SH", edit_kk_resp.data)

        # 9. Add, edit, and delete anggota on detail page.
        self.client.post(
            f"/kk/{kk_id}/anggota/tambah",
            data={
                "nik": "3273014444444444",
                "nama_lengkap": "DINA SANTOSO",
                "jenis_kelamin": "PEREMPUAN",
                "kewarganegaraan": "WNI",
            },
            follow_redirects=True,
        )
        members = list_anggota_by_kk(self.db_path, kk_id)
        self.assertEqual(len(members), 2)
        added_id = members[1]["id"]

        self.client.post(
            f"/kk/{kk_id}/anggota/{added_id}/edit",
            data={
                "nik": "3273014444444444",
                "nama_lengkap": "DINA PUTRI",
                "jenis_kelamin": "PEREMPUAN",
                "kewarganegaraan": "WNI",
            },
            follow_redirects=True,
        )
        self.assertEqual(get_anggota_by_id(self.db_path, kk_id, added_id)["nama_lengkap"], "DINA PUTRI")

        self.client.post(f"/kk/{kk_id}/anggota/{added_id}/hapus", follow_redirects=True)
        self.assertEqual(len(list_anggota_by_kk(self.db_path, kk_id)), 1)

        # 10. Deleting KK removes KK and cascades remaining anggota.
        del_kk_resp = self.client.post(f"/kk/{kk_id}/hapus", follow_redirects=True)
        self.assertEqual(del_kk_resp.status_code, 200)
        self.assertEqual(list_all_kk(self.db_path), [])
        self.assertEqual(list_anggota_by_kk(self.db_path, kk_id), [])


if __name__ == "__main__":
    unittest.main()
