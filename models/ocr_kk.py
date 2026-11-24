"""EasyOCR preprocessing and spatial parser for Indonesian Kartu Keluarga (KK) images."""

import re
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

from models.database import ANGGOTA_FIELDS, IDENTITAS_FIELDS

_READER = None

_DIGIT_CONFUSIONS = str.maketrans({
    "O": "0",
    "o": "0",
    "Q": "0",
    "D": "0",
    "I": "1",
    "l": "1",
    "|": "1",
    "S": "5",
    "s": "5",
    "B": "8",
    "Z": "2",
    "z": "2",
})

_RIGHT_HEADER_LABELS = (
    ("kabupaten_kota", (r"kab(?:upaten)?\s*[/\\|Il1.\-]?\s*kota", r"kabupaten", r"kota")),
    ("kode_pos", (r"kode\s*p[o0]s",)),
    ("kecamatan", (r"kecamatan",)),
    ("provinsi", (r"pro[pv]insi",)),
)

_LEFT_HEADER_LABELS = (
    ("nama_kepala_keluarga", (r"nama\s+kepala\s+keluarga", r"nama\s+\S*\s*keluarga", r"kepala\s+keluarga")),
    (
        "desa_kelurahan",
        (
            r"desa\s*[/\\|Il1.\-]?\s*kelurahan",
            r"kelurahan\s*[/\\|Il1.\-]?\s*desa",
            r"kelurahan",
            r"desa",
        ),
    ),
    ("rt_rw", (r"rt\s*[/\\|Il1.\-]?\s*rw",)),
    ("alamat", (r"alamat",)),
)

_AGAMA_MAP = {
    "ISLAM": "ISLAM",
    "KRISTEN": "KRISTEN",
    "PROTESTAN": "KRISTEN",
    "KATOLIK": "KATOLIK",
    "KATHOLIK": "KATOLIK",
    "HINDU": "HINDU",
    "BUDDHA": "BUDDHA",
    "BUDHA": "BUDDHA",
    "KONGHUCU": "KONGHUCU",
    "KHONGHUCU": "KONGHUCU",
}


def empty_identitas() -> Dict[str, str]:
    """Return an empty KK identity dict matching tbl_identitasKK fields."""
    return {field: "" for field in IDENTITAS_FIELDS}


def empty_anggota() -> Dict[str, str]:
    """Return an empty family member dict matching tbl_anggotaKeluargaKK fields."""
    data = {field: "" for field in ANGGOTA_FIELDS}
    data["kewarganegaraan"] = "WNI"
    return data


def _get_reader():
    """Lazily initialize and cache the EasyOCR reader so imports stay fast."""
    global _READER
    if _READER is None:
        import easyocr  # Deferred import so tests/startup don't pay torch load upfront.

        _READER = easyocr.Reader(["id", "en"], gpu=False)
    return _READER


def preprocess_image(image_path: str) -> np.ndarray:
    """Load an image via Pillow, resize to OCR-friendly dimensions, and apply gentle CLAHE."""
    with Image.open(image_path) as pil_img:
        pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
        rgb = np.array(pil_img)

    height, width = rgb.shape[:2]
    if width > 2000:
        scale = 1800.0 / width
        rgb = cv2.resize(rgb, (1800, max(1, int(round(height * scale)))), interpolation=cv2.INTER_AREA)
    elif width < 1000:
        scale = 1400.0 / max(1, width)
        rgb = cv2.resize(rgb, (1400, max(1, int(round(height * scale)))), interpolation=cv2.INTER_CUBIC)

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_chan)
    merged = cv2.merge((l_enhanced, a_chan, b_chan))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def _clean_digits(text: str) -> str:
    """Normalize common OCR character confusions inside numeric strings."""
    cleaned = text.strip().translate(_DIGIT_CONFUSIONS)
    return re.sub(r"\D", "", cleaned)


def _extract_16_digit_candidate(text: str) -> str:
    """Find a 16-digit KK/NIK number in text, including runs glued to adjacent text."""
    direct = re.search(r"\d{16}", text)
    if direct:
        return direct.group(0)

    # Match 16-character digit-like runs even when glued to adjacent column text (e.g. "B204124809750007]PEREMPUAN").
    for m in re.finditer(r"[0-9OoQDIl|SsBZz]{16}", text):
        raw = m.group(0)
        if sum(ch.isdigit() for ch in raw) >= 12:
            digits = _clean_digits(raw)
            if len(digits) == 16:
                return digits

    # Collapse spaces/dashes within numeric runs like "3273 0123 4567 8901"
    for chunk in re.findall(r"[0-9OoQDIl|SsBZz\-\s]{16,24}", text):
        if sum(ch.isdigit() for ch in chunk) >= 12:
            digits = _clean_digits(chunk)
            if len(digits) == 16:
                return digits

    digits = _clean_digits(text)
    if len(digits) == 16:
        return digits
    return ""


def _strip_label_value(text: str) -> str:
    return re.sub(r"^[\s:.\-\[\]|{}]+", "", text).strip()


def _normalize_rt_rw(text: str) -> str:
    cleaned = _strip_label_value(text)
    if not cleaned:
        return ""
    match = re.search(r"([0-9OoIl|SsBZz]{1,3})\s*[/\\\-|,.\s]\s*([0-9OoIl|SsBZz]{1,3})", cleaned)
    if match:
        rt = _clean_digits(match.group(1)).zfill(3)
        rw = _clean_digits(match.group(2)).zfill(3)
        if rt and rw:
            return f"{rt}/{rw}"
    return cleaned


def _normalize_tanggal(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""
    match = re.search(r"([0-9OoIl|]{1,2})\s*[-/.\s]\s*([0-9OoIl|]{1,2})\s*[-/.\s]\s*([0-9OoIl|]{4})", cleaned)
    if match:
        day = _clean_digits(match.group(1)).zfill(2)
        month = _clean_digits(match.group(2)).zfill(2)
        year = _clean_digits(match.group(3))
        if len(day) == 2 and len(month) == 2 and len(year) == 4:
            return f"{day}-{month}-{year}"
    return cleaned


def _normalize_jenis_kelamin(text: str) -> str:
    upper = text.upper()
    if re.search(r"LAKI[\s\-]*LAKI", upper) or "LAKI" in upper:
        return "LAKI-LAKI"
    if "PEREMPUAN" in upper or "WANITA" in upper:
        return "PEREMPUAN"
    return ""


def _normalize_agama(text: str) -> str:
    upper = text.upper()
    if re.search(r"[\[\|I1]?SLAM", upper):
        return "ISLAM"
    letters = re.sub(r"[^A-Z]", "", upper)
    for key, canonical in _AGAMA_MAP.items():
        if key in letters:
            return canonical
    return ""


def _normalize_pendidikan(text: str) -> str:
    cleaned = _clean_cell_placeholder(text)
    if not cleaned:
        return ""
    compact = re.sub(r"[^A-Z0-9]", "", cleaned.upper())
    if "DIPLOMAIV" in compact and "STRATAI" in compact:
        return "DIPLOMA IV/STRATA I"
    if re.fullmatch(r"SL[TI1][AN]?SEDERAJAT", compact):
        return "SLTA/SEDERAJAT"
    if re.fullmatch(r"SLTP[I1]?SEDERAJAT", compact):
        return "SLTP/SEDERAJAT"
    return cleaned


def _normalize_pekerjaan(text: str) -> str:
    cleaned = _clean_cell_placeholder(text)
    if not cleaned:
        return ""
    compact = re.sub(r"[^A-Z]", "", cleaned.upper())
    if "MENGURUSRUMAHTANGGA" in compact:
        return "MENGURUS RUMAH TANGGA"
    if "PELAJAR" in compact and "MAHASISWA" in compact:
        return "PELAJAR/MAHASISWA"
    if "KARYAWANSWASTA" in compact:
        return "KARYAWAN SWASTA"
    if "BELUMTIDAKBEKERJA" in compact:
        return "BELUM/TIDAK BEKERJA"
    return cleaned


def _normalize_boxes(raw_results: List[Any]) -> List[Dict[str, Any]]:
    boxes: List[Dict[str, Any]] = []
    for item in raw_results:
        if not item or len(item) < 2:
            continue
        bbox = item[0]
        text = str(item[1]).strip()
        conf = float(item[2]) if len(item) > 2 else 1.0
        if not text:
            continue
        xs = [float(pt[0]) for pt in bbox]
        ys = [float(pt[1]) for pt in bbox]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        boxes.append({
            "text": text,
            "conf": conf,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "cx": (x1 + x2) / 2.0,
            "cy": (y1 + y2) / 2.0,
            "h": max(1.0, y2 - y1),
        })
    return boxes


def _group_into_lines(boxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not boxes:
        return []
    median_h = median(b["h"] for b in boxes)
    tol = max(6.0, 0.6 * median_h)

    sorted_boxes = sorted(boxes, key=lambda b: (b["cy"], b["x1"]))
    lines: List[Dict[str, Any]] = []

    for box in sorted_boxes:
        placed = False
        for line in reversed(lines):
            if abs(box["cy"] - line["cy"]) <= tol:
                line["boxes"].append(box)
                n = len(line["boxes"])
                line["cy"] = sum(b["cy"] for b in line["boxes"]) / n
                line["y1"] = min(b["y1"] for b in line["boxes"])
                line["y2"] = max(b["y2"] for b in line["boxes"])
                placed = True
                break
        if not placed:
            lines.append({
                "boxes": [box],
                "cy": box["cy"],
                "y1": box["y1"],
                "y2": box["y2"],
            })

    for line in lines:
        line["boxes"].sort(key=lambda b: b["x1"])
        line["text"] = " ".join(b["text"] for b in line["boxes"])
    return lines


def _extract_field_from_segment(text: str, label_specs: Tuple[Tuple[str, Tuple[str, ...]], ...]) -> Optional[Tuple[str, str]]:
    lowered = text.lower()
    for field, patterns in label_specs:
        for pat in patterns:
            pattern = re.compile(rf"(?:^|\b)(?:{pat})\b\s*[:.\-]?\s*(.*)$", re.IGNORECASE)
            match = pattern.search(lowered)
            if match:
                start_idx = match.start(1)
                value = _strip_label_value(text[start_idx:])
                return field, value
    return None


def _is_column_number_line(line: Dict[str, Any]) -> bool:
    tokens = [b["text"].strip() for b in line["boxes"] if b["text"].strip()]
    if len(tokens) < 3:
        return False
    num_tokens = sum(1 for tok in tokens if re.fullmatch(r"\(?\d{1,2}\)?\.?", tok))
    return num_tokens >= max(3, int(0.6 * len(tokens)))


def _detect_table1_header(lines: List[Dict[str, Any]], median_h: float) -> Tuple[Optional[int], float, float, Dict[str, float], Optional[float]]:
    keywords = (
        "nama lengkap",
        "nik",
        "jenis kelamin",
        "kelamin",
        "katamin",
        "kalamin",
        "tempat",
        "tanggal",
        "tanggai",
        "agama",
        "pendidikan",
        "pekerjaan",
        "pekarjaan",
    )
    for idx, line in enumerate(lines):
        low = line["text"].lower()
        has_nik = bool(re.search(r"\bnik\b", low))
        hits = sum(1 for kw in keywords if kw in low)
        if not ((has_nik and hits >= 2) or hits >= 3):
            continue

        band_lines = [line]
        # KK table headers often wrap across two lines (e.g., "Jenis" / "Kelamin"),
        # but never swallow the first member row (starts with row index or contains NIK/gender).
        if idx + 1 < len(lines) and (lines[idx + 1]["cy"] - line["cy"]) <= 1.8 * median_h:
            next_line = lines[idx + 1]
            next_low = next_line["text"].lower()
            looks_like_data = (
                bool(re.match(r"^\d{1,2}\b", next_low))
                or bool(_extract_16_digit_candidate(next_line["text"]))
                or any(d in next_low for d in ("laki-laki", "perempuan", "islam", "kristen", "katolik", "hindu", "buddha"))
            )
            if not looks_like_data and any(
                w in next_low for w in ("kelamin", "katamin", "lahir", "lahlr", "pekerjaan", "pekarjaan", "lengkap", "nik")
            ):
                band_lines.append(next_line)

        band_boxes = [b for ln in band_lines for b in ln["boxes"]]
        y1 = min(b["y1"] for b in band_boxes)
        y2 = max(b["y2"] for b in band_boxes)

        cols: Dict[str, float] = {}
        no_col_x: Optional[float] = None

        # Sort left-to-right so "Tempat" before "Tanggal" and "Lahir" stays unambiguous.
        band_boxes_sorted = sorted(band_boxes, key=lambda b: (b["cx"], b["cy"]))
        for box in band_boxes_sorted:
            t = box["text"].lower().strip()
            cx = box["cx"]
            if re.fullmatch(r"[\[\|]?no\.?", t) and "nik" not in cols:
                no_col_x = cx
            elif "nama" in t and "lengkap" in t:
                cols["nama_lengkap"] = cx
            elif t == "nama" and "nama_lengkap" not in cols:
                cols["nama_lengkap"] = cx
            elif re.search(r"\bnik\b", t):
                cols["nik"] = cx
            elif "pekerjaan" in t or "pekarjaan" in t:
                cols["jenis_pekerjaan"] = cx
            elif "kelamin" in t or "katamin" in t or "kalamin" in t or t == "jenis":
                if "jenis_kelamin" not in cols and (
                    "kelamin" in t
                    or "katamin" in t
                    or "kalamin" in t
                    or ("nik" in cols and cx > cols["nik"] and "agama" not in cols)
                ):
                    cols["jenis_kelamin"] = cx
                elif "jenis_kelamin" in cols and "agama" in cols:
                    cols["jenis_pekerjaan"] = cx
            elif "tempat" in t:
                cols["tempat_lahir"] = cx
            elif "tanggal" in t or "tanggai" in t:
                cols["tanggal_lahir"] = cx
            elif "lahir" in t or "lahlr" in t:
                if "tempat_lahir" in cols and abs(cx - cols["tempat_lahir"]) < 60:
                    cols["tempat_lahir"] = (cols["tempat_lahir"] + cx) / 2.0
                elif "tanggal_lahir" in cols and abs(cx - cols["tanggal_lahir"]) < 60:
                    cols["tanggal_lahir"] = (cols["tanggal_lahir"] + cx) / 2.0
                elif "tempat_lahir" not in cols:
                    cols["tempat_lahir"] = cx
                else:
                    cols["tanggal_lahir"] = cx
            elif "agama" in t:
                cols["agama"] = cx
            elif "pendidikan" in t:
                cols["pendidikan"] = cx

        return idx, y1, y2, cols, no_col_x
    return None, 0.0, 0.0, {}, None


def _detect_table2_header(lines: List[Dict[str, Any]], start_idx: int, median_h: float) -> Tuple[Optional[int], float, float, Dict[str, float], Optional[float]]:
    keywords = (
        "status perkawinan",
        "perkawinan",
        "perkew",
        "hubungan",
        "kewarg",
        "warganeg",
        "dokumen",
        "dokuman",
        "imigrasi",
        "orang tua",
        "paspor",
        "kitap",
        "kitae",
        "ayah",
        "ayeh",
        "ibu",
    )
    for idx in range(start_idx, len(lines)):
        line = lines[idx]
        low = line["text"].lower()
        hits = sum(1 for kw in keywords if kw in low)
        if hits < 2:
            continue

        band_lines = [line]
        for look_ahead in (1, 2):
            if idx + look_ahead < len(lines) and (lines[idx + look_ahead]["cy"] - line["cy"]) <= 2.4 * median_h:
                next_line = lines[idx + look_ahead]
                next_low = next_line["text"].lower()
                looks_like_data = (
                    bool(re.match(r"^\d{1,2}\b", next_low))
                    or (
                        bool(re.search(r"\b(?:wni|wna|kepala keluarga|istri|anak|kawin)\b", next_low))
                        and "perkawinan" not in next_low
                        and "perkew" not in next_low
                    )
                )
                if not looks_like_data and any(
                    w in next_low
                    for w in (
                        "perkawinan",
                        "perkew",
                        "dalam keluarga",
                        "delam keluarga",
                        "paspor",
                        "kitap",
                        "kitae",
                        "ayah",
                        "ayeh",
                        "ibu",
                        "hubungan",
                    )
                ):
                    band_lines.append(next_line)

        band_boxes = [b for ln in band_lines for b in ln["boxes"]]
        y1 = min(b["y1"] for b in band_boxes)
        y2 = max(b["y2"] for b in band_boxes)

        cols: Dict[str, float] = {}
        no_col_x: Optional[float] = None
        for box in sorted(band_boxes, key=lambda b: (b["cx"], b["cy"])):
            t = box["text"].lower().strip()
            cx = box["cx"]
            if re.fullmatch(r"[\[\|i]?no\.?", t) and "status_perkawinan" not in cols:
                no_col_x = cx
            elif ("status" in t and "perkawinan" in t) or t == "status" or "perkew" in t:
                cols["status_perkawinan"] = cx
            elif "tanggal" in t and "perkawinan" in t:
                cols["_tanggal_perkawinan"] = cx
            elif "perkawinan" in t:
                if "status_perkawinan" not in cols:
                    cols["status_perkawinan"] = cx
                else:
                    cols["_tanggal_perkawinan"] = cx
            elif "hubungan" in t or "dalam keluarga" in t or "delam keluarga" in t:
                cols["status_hubungan_dalam_keluarga"] = cx
            elif "kewarg" in t or "warganeg" in t:
                cols["kewarganegaraan"] = cx
            elif "paspor" in t:
                cols["_paspor"] = cx
            elif "kitap" in t or "kitas" in t or "kitae" in t:
                cols["_kitap"] = cx
            elif "ayah" in t or "ayeh" in t:
                cols["nama_ayah"] = cx
            elif re.search(r"\bibu\b", t):
                cols["nama_ibu"] = cx

        return idx, y1, y2, cols, no_col_x
    return None, 0.0, 0.0, {}, None


def _assign_line_to_columns(
    line: Dict[str, Any],
    cols: Dict[str, float],
    no_col_x: Optional[float],
    first_data_col_x: Optional[float],
) -> Tuple[Optional[int], Dict[str, str]]:
    """Assign each box in a table row to the nearest detected column center."""
    row_num: Optional[int] = None
    buckets: Dict[str, List[str]] = {col: [] for col in cols}
    col_items = list(cols.items())

    for box in line["boxes"]:
        txt = box["text"].strip()
        if not txt:
            continue
        cx = box["cx"]
        if re.fullmatch(r"\d{1,2}\.?", txt):
            if no_col_x is not None and first_data_col_x is not None:
                if abs(cx - no_col_x) <= abs(cx - first_data_col_x):
                    row_num = int(txt.rstrip("."))
                    continue
            elif first_data_col_x is not None and cx < first_data_col_x - 35:
                row_num = int(txt.rstrip("."))
                continue

        nearest_col = min(col_items, key=lambda item: abs(cx - item[1]))[0]
        buckets[nearest_col].append(txt)

    values = {col: " ".join(parts).strip() for col, parts in buckets.items() if parts}
    return row_num, values


def _clean_cell_placeholder(value: str) -> str:
    stripped = re.sub(r"^[\[\]|{}\s]+|[\[\]|{}\s]+$", "", value)
    if stripped in {"-", "--", "---", "—", "_"}:
        return ""
    return stripped


def _extract_status_perkawinan(text: str) -> str:
    upper = text.upper()
    for pat, canonical in (
        (r"BELUM\s*KAWIN", "BELUM KAWIN"),
        (r"CERAI\s*HIDUP", "CERAI HIDUP"),
        (r"CERAI\s*MATI", "CERAI MATI"),
        (r"KAWIN\s*BELUM\s*TERCATAT", "KAWIN BELUM TERCATAT"),
        (r"KAWIN\s*TERCATAT", "KAWIN TERCATAT"),
        (r"KAWIN", "KAWIN"),
    ):
        if re.search(pat, upper):
            return canonical
    return ""


def _extract_hubungan_keluarga(text: str) -> str:
    upper = text.upper()
    for pat, canonical in (
        (r"KEPALA\s*KELUARGA", "KEPALA KELUARGA"),
        (r"ORANG\s*TUA", "ORANG TUA"),
        (r"FAMILI\s*LAIN", "FAMILI LAIN"),
        (r"MENANTU", "MENANTU"),
        (r"MERTUA", "MERTUA"),
        (r"PEMBANTU", "PEMBANTU"),
        (r"SUAMI", "SUAMI"),
        (r"ISTRI", "ISTRI"),
        (r"ANAK", "ANAK"),
        (r"CUCU", "CUCU"),
    ):
        if re.search(pat, upper):
            return canonical
    return ""


def _extract_tempat_after_gender(text: str) -> str:
    """Recover tempat_lahir when OCR glues it right after LAKI-LAKI or PEREMPUAN."""
    match = re.search(r"(?:PEREMPUAN|LAKI[\s\-]*LAKI)[\s\]\|]*([A-Z]{3,}(?:\s+[A-Z]{2,})?)", text.upper())
    if match:
        candidate = match.group(1).strip()
        if candidate not in {"ISLAM", "KRISTEN", "KATOLIK", "HINDU", "BUDDHA", "WNI", "WNA"}:
            return candidate
    return ""


def _parse_header_identitas(
    header_lines: List[Dict[str, Any]],
    image_width: int,
    image_height: int,
) -> Dict[str, str]:
    identitas = empty_identitas()

    # 1. Extract 16-digit No. KK near the top of the KK document.
    for line in header_lines:
        if line["cy"] <= 0.35 * image_height or "kartu keluarga" in line["text"].lower() or re.search(r"\bno\b", line["text"].lower()):
            if "nik" in line["text"].lower():
                continue
            cand = _extract_16_digit_candidate(line["text"])
            if cand:
                identitas["no_kk"] = cand
                break

    # 2. Compute left/right column split X from right-side labels (Kecamatan, Kab/Kota, Kode Pos, Provinsi).
    right_label_xs: List[float] = []
    for line in header_lines:
        for box in line["boxes"]:
            low = box["text"].lower()
            if box["x1"] >= 0.35 * image_width and any(
                kw in low for kw in ("kecamatan", "kabupaten", "kode pos", "provinsi", "propinsi")
            ):
                right_label_xs.append(box["x1"])
    split_x = min(right_label_xs) - 2.0 if right_label_xs else 0.52 * image_width

    last_left_field: Optional[str] = None
    for line in header_lines:
        left_boxes = [b for b in line["boxes"] if b["cx"] < split_x]
        right_boxes = [b for b in line["boxes"] if b["cx"] >= split_x]

        left_text = " ".join(b["text"] for b in left_boxes).strip()
        if left_text:
            parsed_left = _extract_field_from_segment(left_text, _LEFT_HEADER_LABELS)
            if parsed_left:
                field, val = parsed_left
                last_left_field = field
                if field == "rt_rw":
                    val = _normalize_rt_rw(val)
                if val:
                    identitas[field] = val
            elif (
                last_left_field == "alamat"
                and not identitas["rt_rw"]
                and not any(kw in left_text.lower() for kw in ("kartu keluarga", "no.", "nomor"))
                and not _extract_16_digit_candidate(left_text)
            ):
                identitas["alamat"] = f"{identitas['alamat']} {left_text}".strip()

        right_text = " ".join(b["text"] for b in right_boxes).strip()
        if right_text:
            parsed_right = _extract_field_from_segment(right_text, _RIGHT_HEADER_LABELS)
            if parsed_right:
                field, val = parsed_right
                if field == "kode_pos":
                    m = re.search(r"\b\d{5}\b", _clean_digits(val) if val else "")
                    if m:
                        val = m.group(0)
                    else:
                        digits = _clean_digits(val)
                        val = digits[:5] if len(digits) >= 5 else val
                if val:
                    identitas[field] = val

    return identitas


def _fallback_extract_anggota(lines: List[Dict[str, Any]], no_kk: str) -> List[Dict[str, str]]:
    """Fallback heuristic when table headers are unreadable: locate 16-digit NIKs."""
    members: List[Dict[str, str]] = []
    for line in lines:
        nik = _extract_16_digit_candidate(line["text"])
        if not nik or nik == no_kk:
            continue
        member = empty_anggota()
        member["nik"] = nik
        tgl_match = re.search(r"\d{1,2}[-/.\s]\d{1,2}[-/.\s]\d{4}", line["text"])
        if tgl_match:
            member["tanggal_lahir"] = _normalize_tanggal(tgl_match.group(0))
        jk = _normalize_jenis_kelamin(line["text"])
        if jk in ("LAKI-LAKI", "PEREMPUAN"):
            member["jenis_kelamin"] = jk
        members.append(member)
    return members


def parse_kk_ocr_results(
    raw_results: List[Any],
    image_width: int,
    image_height: int,
) -> Dict[str, Any]:
    """Parse EasyOCR bounding-box output into structured KK identity and family members."""
    boxes = _normalize_boxes(raw_results)
    lines = _group_into_lines(boxes)
    raw_text = "\n".join(line["text"] for line in lines)

    if not lines:
        return {"identitas": empty_identitas(), "anggota": [], "raw_text": raw_text}

    median_h = median(b["h"] for b in boxes)
    t1_idx, t1_y1, t1_y2, t1_cols, t1_no_x = _detect_table1_header(lines, median_h)

    header_cutoff_y = t1_y1 if t1_idx is not None else 0.45 * image_height
    header_lines = [ln for ln in lines if ln["cy"] < header_cutoff_y]
    identitas = _parse_header_identitas(header_lines, image_width, image_height)

    anggota_list: List[Dict[str, str]] = []
    if t1_idx is not None and t1_cols:
        t2_idx, t2_y1, t2_y2, t2_cols, t2_no_x = _detect_table2_header(lines, t1_idx + 1, median_h)
        t1_end_y = t2_y1 if t2_idx is not None else float("inf")

        t1_first_x = min(t1_cols.values())
        t1_rows: List[Tuple[Optional[int], Dict[str, str]]] = []
        for line in lines[t1_idx + 1 :]:
            if line["cy"] <= t1_y2 + 0.1 * median_h:
                continue
            if line["cy"] >= t1_end_y - 0.1 * median_h:
                break
            if _is_column_number_line(line):
                continue
            if "tidak ada data" in line["text"].lower():
                continue

            row_num, raw_vals = _assign_line_to_columns(line, t1_cols, t1_no_x, t1_first_x)
            if not any(raw_vals.values()):
                continue

            member = empty_anggota()
            member["nama_lengkap"] = _clean_cell_placeholder(raw_vals.get("nama_lengkap", ""))
            member["tempat_lahir"] = _clean_cell_placeholder(raw_vals.get("tempat_lahir", ""))
            member["pendidikan"] = _normalize_pendidikan(raw_vals.get("pendidikan", ""))
            member["jenis_pekerjaan"] = _normalize_pekerjaan(raw_vals.get("jenis_pekerjaan", ""))

            # De-merge NIK + Jenis Kelamin + Tempat Lahir when OCR glues adjacent cells.
            middle_text = " ".join(
                raw_vals.get(k, "") for k in ("nik", "jenis_kelamin", "tempat_lahir") if raw_vals.get(k)
            )
            nik_cand = _extract_16_digit_candidate(middle_text) or _extract_16_digit_candidate(line["text"])
            if nik_cand and nik_cand != identitas["no_kk"]:
                member["nik"] = nik_cand
            elif raw_vals.get("nik"):
                member["nik"] = _clean_cell_placeholder(raw_vals["nik"])

            jk = _normalize_jenis_kelamin(middle_text) or _normalize_jenis_kelamin(line["text"])
            if jk:
                member["jenis_kelamin"] = jk

            if not member["tempat_lahir"]:
                member["tempat_lahir"] = _extract_tempat_after_gender(middle_text)

            # De-merge Tanggal Lahir + Agama (e.g. "15-08-1982[SLAM" or "21-04-2OO4ISLAM").
            date_agama_text = " ".join(
                raw_vals.get(k, "") for k in ("tanggal_lahir", "agama") if raw_vals.get(k)
            )
            tgl = _normalize_tanggal(date_agama_text) if date_agama_text else ""
            if re.fullmatch(r"\d{2}-\d{2}-\d{4}", tgl):
                member["tanggal_lahir"] = tgl
            elif raw_vals.get("tanggal_lahir"):
                member["tanggal_lahir"] = _clean_cell_placeholder(raw_vals["tanggal_lahir"])

            agama = _normalize_agama(date_agama_text) or _normalize_agama(line["text"])
            if agama:
                member["agama"] = agama

            if member["nama_lengkap"] or member["nik"]:
                t1_rows.append((row_num, member))

        t2_rows: List[Tuple[Optional[int], Dict[str, str]]] = []
        if t2_idx is not None and t2_cols:
            t2_first_x = min(t2_cols.values())
            for line in lines[t2_idx + 1 :]:
                if line["cy"] <= t2_y2 + 0.1 * median_h:
                    continue
                low = line["text"].lower()
                if any(stop in low for stop in ("dikeluarkan", "kepala dinas", "lembar", "tanda tangan")):
                    break
                if _is_column_number_line(line):
                    continue

                row_num, raw_vals = _assign_line_to_columns(line, t2_cols, t2_no_x, t2_first_x)
                filtered = {
                    k: _clean_cell_placeholder(v)
                    for k, v in raw_vals.items()
                    if not k.startswith("_")
                }

                # Short left-aligned father names can fall near the _kitap/_ayah midpoint;
                # pure alphabetic strings in _kitap belong to nama_ayah (KITAP/KITAS is '-' or alphanumeric with digits).
                kitap_val = _clean_cell_placeholder(raw_vals.get("_kitap", ""))
                if not filtered.get("nama_ayah") and kitap_val and re.fullmatch(r"[A-Za-z\s.,']{3,}", kitap_val):
                    filtered["nama_ayah"] = kitap_val

                # De-merge Status Perkawinan + Status Hubungan + Kewarganegaraan.
                status_segment = " ".join(
                    filtered.get(k, "")
                    for k in ("status_perkawinan", "status_hubungan_dalam_keluarga", "kewarganegaraan")
                    if filtered.get(k)
                )
                if status_segment:
                    sp = _extract_status_perkawinan(status_segment)
                    if sp:
                        filtered["status_perkawinan"] = sp
                    hub = _extract_hubungan_keluarga(status_segment)
                    if hub:
                        filtered["status_hubungan_dalam_keluarga"] = hub
                    if "WNA" in status_segment.upper():
                        filtered["kewarganegaraan"] = "WNA"
                    elif "WNI" in status_segment.upper() or re.search(r"WN\b", status_segment.upper()):
                        filtered["kewarganegaraan"] = "WNI"

                if any(filtered.values()):
                    t2_rows.append((row_num, filtered))

        # Merge Table 2 attributes onto Table 1 members (prefer matching row_num, fallback to positional index).
        t2_by_num = {num: vals for num, vals in t2_rows if num is not None}
        for idx, (row_num, member) in enumerate(t1_rows):
            t2_vals = None
            if row_num is not None and row_num in t2_by_num:
                t2_vals = t2_by_num[row_num]
            elif idx < len(t2_rows):
                t2_vals = t2_rows[idx][1]

            if t2_vals:
                for field in (
                    "status_perkawinan",
                    "status_hubungan_dalam_keluarga",
                    "kewarganegaraan",
                    "nama_ayah",
                    "nama_ibu",
                ):
                    if t2_vals.get(field):
                        member[field] = t2_vals[field]
            if not member["kewarganegaraan"]:
                member["kewarganegaraan"] = "WNI"
            anggota_list.append(member)
    else:
        anggota_list = _fallback_extract_anggota(lines, identitas["no_kk"])

    # Fallback: Nama Kepala Keluarga matches the head-of-household member (typically row 1).
    if not identitas["nama_kepala_keluarga"] and anggota_list:
        kepala = next(
            (m["nama_lengkap"] for m in anggota_list if "KEPALA KELUARGA" in m.get("status_hubungan_dalam_keluarga", "").upper() and m.get("nama_lengkap")),
            "",
        )
        identitas["nama_kepala_keluarga"] = kepala or anggota_list[0].get("nama_lengkap", "")

    return {
        "identitas": identitas,
        "anggota": anggota_list,
        "raw_text": raw_text,
    }


def extract_kk_from_image(image_path: str) -> Dict[str, Any]:
    """Run preprocessing + EasyOCR + spatial parsing on a Kartu Keluarga image."""
    try:
        processed = preprocess_image(image_path)
        height, width = processed.shape[:2]
        reader = _get_reader()
        raw_results = reader.readtext(processed, detail=1, paragraph=False, width_ths=0.3)
        parsed = parse_kk_ocr_results(raw_results, image_width=width, image_height=height)
        parsed["error"] = None
        return parsed
    except Exception as exc:
        return {
            "identitas": empty_identitas(),
            "anggota": [],
            "raw_text": "",
            "error": str(exc),
        }
