# Image tier helpers shared by prepare_pages.py, crop_figure.py and zoom.py.
#
# All page images are 8-bit grayscale NumPy arrays (0 = ink, 255 = paper) taken
# straight from the 1-bit CCITT scans in the PDF (5100 x 7020 px at 600 dpi).
# Coordinates passed between scripts are percentages of that full bitmap.

import re
from dataclasses import dataclass

import cv2
import numpy as np
import pymupdf
import pytesseract

SCAN_DPI = 600
INK_THRESHOLD = 128

# Illustration IDs printed in the bottom right corner of every framed figure
ID_PATTERN_SPECIFIC = r"^[A-Z]{2}\d{4}$"  # two uppercase letters followed by four digits
MISIDENTIFIED_DIGIT_MAP = {"O": "0", "I": "1", "S": "5", "Z": "2", "B": "8", "G": "6", "Q": "0", "D": "0"}

# Page header: "ENGINE MECHANICAL - Engine Tune-Up   EM-11"
MISIDENTIFIED_LETTER_MAP = {"1": "I", "0": "O", "5": "S", "8": "B", "6": "G", "2": "Z"}


# ----------------------------------------------------------------------------
# Loading pages
# ----------------------------------------------------------------------------


def load_page_bitmap(doc, page_number):
    """
    Load the scanned bitmap of a page without re-rasterizing it.
    :param doc: open pymupdf document
    :param page_number: 1-based PDF page number
    :return: grayscale NumPy array (0 = ink, 255 = paper)
    """
    page = doc[page_number - 1]
    images = page.get_images(full=True)
    if images:
        pix = pymupdf.Pixmap(doc, images[0][0])
        if pix.n != 1 or pix.alpha:
            pix = pymupdf.Pixmap(pymupdf.csGRAY, pix)
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).copy()
        # some pages are stored rotated relative to how they are displayed
        if page.rotation:
            image = rotate_multiple_of_90(image, page.rotation)
        return image
    # fallback for pages that are not a single scanned image
    pix = page.get_pixmap(dpi=SCAN_DPI, colorspace=pymupdf.csGRAY)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).copy()


def scale_to_dpi(image, dpi):
    """Downscale a 600 dpi bitmap to the requested dpi (area interpolation keeps thin lines readable)."""
    if dpi >= SCAN_DPI:
        return image
    factor = dpi / SCAN_DPI
    size = (max(1, round(image.shape[1] * factor)), max(1, round(image.shape[0] * factor)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def rotate_multiple_of_90(image, degrees):
    """Rotate an image clockwise by a multiple of 90 degrees."""
    degrees %= 360
    if degrees == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


# ----------------------------------------------------------------------------
# Geometry: skew, rotation, boxes
# ----------------------------------------------------------------------------


def pct_to_px(box_pct, shape):
    """Convert a (x0, y0, x1, y1) box in page percent to pixel coordinates for an image of the given shape."""
    height, width = shape[:2]
    x0, y0, x1, y1 = box_pct
    return (
        int(round(x0 / 100 * width)),
        int(round(y0 / 100 * height)),
        int(round(x1 / 100 * width)),
        int(round(y1 / 100 * height)),
    )


def px_to_pct(box_px, shape):
    """Convert a (x0, y0, x1, y1) pixel box to page percent (rounded to 0.1)."""
    height, width = shape[:2]
    x0, y0, x1, y1 = box_px
    return [round(x0 / width * 100, 1), round(y0 / height * 100, 1), round(x1 / width * 100, 1),
            round(y1 / height * 100, 1)]


def estimate_skew(image):
    """
    Estimate the skew of a scanned page in degrees (positive = content rotated clockwise).
    Uses the long horizontal rules (header line, frame borders, table lines) and
    falls back to text line orientation.
    :param image: grayscale page bitmap
    :return: angle in degrees, clamped to +-3
    """
    small = scale_to_dpi(image, 100)
    ink = (small < INK_THRESHOLD).astype(np.uint8) * 255
    width = small.shape[1]

    # long horizontal segments
    edges = cv2.Canny(small, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=80, minLineLength=int(width * 0.25), maxLineGap=8)
    angles = []
    if lines is not None:
        for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) <= 3:
                angles.append(angle)
    if len(angles) >= 2:
        return float(np.clip(np.median(angles), -3, 3))

    # text lines: merge characters horizontally and measure the blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    merged = cv2.dilate(ink, kernel)
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        (_, _), (w, h), angle = cv2.minAreaRect(contour)
        if max(w, h) < width * 0.15:
            continue
        if w < h:
            angle += 90
        if angle > 45:
            angle -= 90
        if abs(angle) <= 3:
            angles.append(angle)
    if angles:
        return float(np.clip(np.median(angles), -3, 3))
    return 0.0


def deskew(image, angle):
    """Rotate an image by -angle degrees about its centre, filling with paper white."""
    if abs(angle) < 0.05:
        return image
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=255)


def detect_rotation(image_low_dpi):
    """
    Detect page orientation with tesseract's orientation and script detection.
    :param image_low_dpi: grayscale page at ~150-300 dpi
    :return: (clockwise degrees needed to make the page upright: 0/90/180/270, confidence) or (None, 0)
    """
    try:
        osd = pytesseract.image_to_osd(image_low_dpi, config="--psm 0")
    except pytesseract.TesseractError:
        return None, 0.0
    rotate = re.search(r"Rotate: (\d+)", osd)
    confidence = re.search(r"Orientation confidence: ([\d.]+)", osd)
    if not rotate:
        return None, 0.0
    return int(rotate.group(1)), float(confidence.group(1)) if confidence else 0.0


# ----------------------------------------------------------------------------
# Illustration frames
# ----------------------------------------------------------------------------


@dataclass
class Frame:
    quad: np.ndarray  # 4x2 float32 corners, ordered top-left, top-right, bottom-right, bottom-left
    bbox_px: tuple  # (x0, y0, x1, y1) axis aligned bounding box
    bbox_pct: list
    angle_deg: float  # skew of the top edge
    full_width: bool  # spans both the illustration and the text column


def order_points(points):
    """
    Order a rectangle edge coordinates as follows: top-left, top-right, bottom-right, and bottom-left.
    :param points: coordinates of the rectangle edges
    :return: ordered coordinates
    """
    rectangle = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = points.sum(axis=1)
    rectangle[0] = points[np.argmin(s)]
    rectangle[2] = points[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(points, axis=1)
    rectangle[1] = points[np.argmin(diff)]
    rectangle[3] = points[np.argmax(diff)]
    return rectangle


def compute_distance(a, b):
    """Compute the Euclidean distance between two points."""
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def level_rectangle(image, points):
    """
    Apply a perspective transformation to level an image.
    :param image: image to transform
    :param points: rectangle edge coordinates
    :return: a transformed image
    """
    rect = order_points(points)
    (top_left, top_right, bottom_right, bottom_left) = rect
    width_a = compute_distance(bottom_right, bottom_left)
    width_b = compute_distance(top_right, top_left)
    max_width = max(int(width_a), int(width_b))
    height_a = compute_distance(top_right, bottom_right)
    height_b = compute_distance(top_left, bottom_left)
    max_height = max(int(height_a), int(height_b))
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (max_width, max_height), borderValue=255)


def find_frames(image, min_area_ratio=0.003, full_width_ratio=0.55, work_dpi=300):
    """
    Detect the black rectangular frames drawn around illustrations (and tables).
    Long horizontal and vertical ink runs are kept, small gaps in them closed, and
    the enclosed rectangles measured; this survives broken or faint frame lines.
    :param image: grayscale page bitmap
    :param min_area_ratio: minimum frame area as a fraction of the page area
    :param full_width_ratio: frames wider than this fraction of the page are flagged as full width
    :param work_dpi: resolution the detection runs at
    :return: list of Frame, sorted top to bottom then left to right
    """
    page_height, page_width = image.shape[:2]
    small = scale_to_dpi(image, work_dpi)
    scale = page_width / small.shape[1]
    ink = (small < INK_THRESHOLD).astype(np.uint8) * 255

    # straight runs of at least ~0.4 in (frames are several inches long, text strokes are not)
    run = int(0.4 * work_dpi)
    horizontal = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (run, 1)))
    vertical = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, run)))
    lines = cv2.bitwise_or(horizontal, vertical)
    gap = int(0.05 * work_dpi) | 1
    lines = cv2.morphologyEx(lines, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (gap, gap)))

    contours, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frames = []
    min_area = min_area_ratio * small.shape[0] * small.shape[1]
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h < min_area or min(w, h) < 0.02 * small.shape[1]:
            continue
        rect = cv2.minAreaRect(contour)
        quad = order_points(cv2.boxPoints(rect).astype("float32"))
        if not _sides_covered(lines, quad):
            continue
        quad = quad * scale
        x, y, w, h = (int(round(v * scale)) for v in (x, y, w, h))
        top_left, top_right = quad[0], quad[1]
        angle = float(np.degrees(np.arctan2(top_right[1] - top_left[1], top_right[0] - top_left[0])))
        frames.append(Frame(
            quad=quad,
            bbox_px=(x, y, x + w, y + h),
            bbox_pct=px_to_pct((x, y, x + w, y + h), image.shape),
            angle_deg=round(angle, 2),
            full_width=w > full_width_ratio * page_width,
        ))
    frames.sort(key=lambda f: (round(f.bbox_px[1] / 200), f.bbox_px[0]))
    return frames


def _sides_covered(lines, quad, min_coverage=0.7, samples=200, tolerance=4):
    """True when ink lines run along all four sides of the quad (i.e. it is a drawn frame, not a table grid)."""
    height, width = lines.shape
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        hits = 0
        for t in np.linspace(0, 1, samples):
            x, y = a + (b - a) * t
            x0, x1 = max(0, int(x) - tolerance), min(width, int(x) + tolerance + 1)
            y0, y1 = max(0, int(y) - tolerance), min(height, int(y) + tolerance + 1)
            if lines[y0:y1, x0:x1].any():
                hits += 1
        if hits / samples < min_coverage:
            return False
    return True


def extract_frame(image, frame):
    """Cut a frame out of the page, levelled by its own corners (re-binarized after the warp)."""
    warped = level_rectangle(image, frame.quad)
    _, binary = cv2.threshold(warped, INK_THRESHOLD, 255, cv2.THRESH_BINARY)
    return binary


def trim_border(image, max_scan=40, ink_ratio=0.5, pad=4):
    """
    Remove the residual frame line around a levelled illustration.
    Walks inward from each side while rows/columns are mostly ink, then pads a little.
    :param image: grayscale illustration including its border line
    :param max_scan: maximum pixels to scan per side
    :param ink_ratio: a row/column with more ink than this is considered part of the border
    :param pad: extra pixels removed after the border line
    :return: cropped image
    """
    ink = image < INK_THRESHOLD
    height, width = ink.shape

    def edge(profile, limit):
        offset = 0
        # skip a possible thin white gap left by the contour approximation, then the line
        while offset < limit and profile[offset] > ink_ratio:
            offset += 1
        if offset == 0:
            gap = 0
            while gap < 6 and gap < limit and profile[gap] <= ink_ratio:
                gap += 1
            if gap < 6 and gap < limit and profile[gap] > ink_ratio:
                offset = gap
                while offset < limit and profile[offset] > ink_ratio:
                    offset += 1
        return min(offset + pad, limit)

    rows = ink.mean(axis=1)
    cols = ink.mean(axis=0)
    top = edge(rows, min(max_scan, height // 4))
    bottom = edge(rows[::-1], min(max_scan, height // 4))
    left = edge(cols, min(max_scan, width // 4))
    right = edge(cols[::-1], min(max_scan, width // 4))
    return image[top:height - bottom, left:width - right]


def trim_to_ink(image, pad=20):
    """Crop an image to the bounding box of its ink plus padding."""
    ink = np.argwhere(image < INK_THRESHOLD)
    if ink.size == 0:
        return image
    y0, x0 = ink.min(axis=0)
    y1, x1 = ink.max(axis=0)
    height, width = image.shape[:2]
    return image[max(0, y0 - pad):min(height, y1 + pad + 1), max(0, x0 - pad):min(width, x1 + pad + 1)]


def read_frame_ids(image):
    """
    Read the illustration ID(s) printed in the bottom right corner of a frame.
    IDs look like "EM8548"; several may be listed for composite figures.
    :param image: grayscale illustration (border removed)
    :return: list of IDs, e.g. ["EC0003", "EC0004"]
    """
    height, width = image.shape[:2]
    # widest box first: a composite figure lists its IDs side by side and needs the room.
    # Narrower boxes are the fallback for drawings whose strokes reach into the corner and
    # drown the ID; a narrow box can only ever find fewer IDs, never different ones.
    boxes = ((160, 630), (130, 500), (110, 430), (90, 350))
    # native resolution reads the small print best; sparse mode (11) copes with drawing strokes;
    # upscaling and the whitelist are last resorts (the whitelist merges I/1 and O/0)
    attempts = (
        (1, "--psm 6"), (1, "--psm 7"), (1, "--psm 11"),
        (2, "--psm 6"), (2, "--psm 7"), (2, "--psm 11"),
        (2, "--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "),
    )
    for box_height, box_width in boxes:
        crop = image[height - min(box_height, height):, width - min(box_width, width):]
        for scale, config in attempts:
            scaled = crop if scale == 1 else cv2.resize(crop, None, fx=scale, fy=scale,
                                                        interpolation=cv2.INTER_CUBIC)
            ids = parse_frame_ids(pytesseract.image_to_string(scaled, config=config))
            if ids:
                return ids
    return []


def parse_frame_ids(text):
    """Extract illustration IDs from OCR text, correcting the usual letter/digit confusions."""
    ids = []
    # the IDs of a composite figure are sometimes read as one run ("AB0014AB0257"): split them apart
    text = re.sub(r"(?<=\d{4})(?=[A-Z]{2}\d)", " ", text.upper())
    for token in re.findall(r"(?<![A-Z0-9])[A-Z0-9]{6,9}(?![A-Z0-9])", text):
        head = "".join(MISIDENTIFIED_LETTER_MAP.get(c, c) for c in token[:2])
        tail = "".join(MISIDENTIFIED_DIGIT_MAP.get(c, c) for c in token[2:])
        # extra zeros sometimes get read between the letters and the four digits
        tail = tail.lstrip("0").rjust(4, "0") if len(tail) > 4 else tail
        candidate = head + tail
        if re.match(ID_PATTERN_SPECIFIC, candidate) and candidate not in ids:
            ids.append(candidate)
    return ids


# ----------------------------------------------------------------------------
# Output images
# ----------------------------------------------------------------------------


def white_to_alpha(image):
    """
    Convert a grayscale illustration to black ink on a transparent background.
    :param image: grayscale image
    :return: BGRA image (ink = opaque black, paper = transparent)
    """
    _, binary = cv2.threshold(image, INK_THRESHOLD, 255, cv2.THRESH_BINARY)
    result = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    result[:, :, 3] = np.where(binary == 255, 0, 255).astype(np.uint8)
    return result


def save_webp(path, bgra):
    """Write a BGRA image as lossless WebP (OpenCV switches to lossless above quality 100)."""
    cv2.imwrite(str(path), bgra, [cv2.IMWRITE_WEBP_QUALITY, 101])


def save_preview_png(path, image, max_width=600):
    """Write a downscaled grayscale PNG for viewing (Claude's Read tool cannot open WebP)."""
    if image.ndim == 3:
        # composite BGRA on white
        alpha = image[:, :, 3:4].astype(np.float32) / 255
        image = (255 * (1 - alpha)).astype(np.uint8)[:, :, 0]
    height, width = image.shape[:2]
    if width > max_width:
        image = cv2.resize(image, (max_width, max(1, round(height * max_width / width))), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), image)


# ----------------------------------------------------------------------------
# OCR
# ----------------------------------------------------------------------------


def read_header(image_low_dpi, expect_section=None, strip_ratio=0.06):
    """
    Read the page code and header text from the top strip of a page.
    :param image_low_dpi: grayscale page at ~300 dpi
    :param expect_section: section code expected on this page, e.g. "EM"; used to fix OCR confusions
    :return: (code like "EM-11" or None, header line text)
    """
    height = image_low_dpi.shape[0]
    strip = image_low_dpi[: int(height * strip_ratio), :]
    text = pytesseract.image_to_string(strip, config="--psm 6").strip()
    lines = [line.strip(" |_-—–") for line in text.splitlines() if line.strip()]
    header = lines[0] if lines else ""

    # "EM-11", but also "Cco-2", "COo-3", "1G-2" ...: normalise digits to letters,
    # collapse doubled letters, then compare with the expected section code
    candidates = []
    for section, number in re.findall(r"(?<![A-Za-z0-9])([A-Za-z0-9]{1,4})-(\d{1,3})(?!\d)", text):
        letters = "".join(MISIDENTIFIED_LETTER_MAP.get(c, c) for c in section.upper())
        letters = re.sub(r"(.)\1+", r"\1", letters)
        candidates.append((letters, int(number)))

    code = None
    if expect_section:
        expected = expect_section.upper()
        for section, number in candidates:
            if section == expected:
                code = f"{expected}-{number}"
                break
        if code is None:
            # one letter misread (e.g. "FM-11" for "EM-11") or one stray letter ("CCO")
            for section, number in candidates:
                if _letters_close(section, expected):
                    code = f"{expected}-{number}"
                    break
    elif candidates:
        section, number = candidates[0]
        code = f"{section}-{number}"
    return code, header


def _letters_close(read, expected):
    if len(read) == len(expected):
        return sum(a != b for a, b in zip(read, expected)) == 1
    if len(read) == len(expected) + 1:
        return any(read[:i] + read[i + 1:] == expected for i in range(len(read)))
    return False


def ocr_page(image_low_dpi, mask_boxes_pct=(), psm=4):
    """
    OCR a page with the illustration frames masked out.
    :param image_low_dpi: grayscale page at ~300 dpi
    :param mask_boxes_pct: boxes (page percent) to white out before OCR
    :param psm: tesseract page segmentation mode
    :return: (cleaned text, TSV data as returned by tesseract)
    """
    image = image_low_dpi.copy()
    for box in mask_boxes_pct:
        x0, y0, x1, y1 = pct_to_px(box, image.shape)
        image[y0:y1, x0:x1] = 255
    config = f"--psm {psm} -c preserve_interword_spaces=1"
    text = pytesseract.image_to_string(image, config=config)
    tsv = pytesseract.image_to_data(image, config=config)
    return postprocess_ocr(text), tsv


def postprocess_ocr(text):
    """Apply the simple clean-ups that were previously done by ocr-fixup.sh."""
    text = text.replace("\f", "")
    text = re.sub(r"[“”]", '"', text)
    text = re.sub(r"[‘’]", "'", text)
    # dashes between numbers are ranges in the manual: "0.15 — 0.25" -> "0.15 – 0.25"
    text = re.sub(r"(?<=\d)\s*[—–\-]{1,2}\s*(?=\d)", " – ", text)
    # OCR renders bullet points as "e", "@" or "«"
    text = re.sub(r"^(\s*)[e@«•]\s+(?=[A-Z(])", r"\1* ", text, flags=re.MULTILINE)
    # remove hyphenated word breaks
    text = re.sub(r"(\w)-\n(?=[a-z])", r"\1", text)
    # collapse runs of blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def ocr_region(image, psm=6):
    """OCR an arbitrary grayscale crop."""
    return postprocess_ocr(pytesseract.image_to_string(image, config=f"--psm {psm} -c preserve_interword_spaces=1"))
