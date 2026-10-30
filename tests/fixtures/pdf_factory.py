"""Tiny dependency-free PDF writer for SYNTHETIC test fixtures.

Produces valid PDF 1.4 files (correct xref offsets) with Helvetica text (WinAnsi), ruled
lines for tables, optional image-only pages and an optional ``/PageLabels`` tree. The
clinical content written into fixtures is invented and must never be cited.
"""

from __future__ import annotations

from dataclasses import dataclass, field

PAGE_W, PAGE_H = 595, 842


@dataclass
class Page:
    texts: list[tuple[float, float, str]] = field(default_factory=list)  # x, y(top-down), text
    lines: list[tuple[float, float, float, float]] = field(default_factory=list)
    image_only: bool = False
    font_size: float = 10.0

    def text(self, x: float, y: float, s: str) -> Page:
        self.texts.append((x, y, s))
        return self

    def table(
        self, x: float, y: float, col_widths: list[float], rows: list[list[str]], row_h: float = 18
    ) -> Page:
        """Draw a fully ruled grid and write ``rows`` into its cells."""
        width = sum(col_widths)
        for r in range(len(rows) + 1):
            self.lines.append((x, y + r * row_h, x + width, y + r * row_h))
        cx = x
        for w in [0.0, *col_widths]:
            cx += w
            self.lines.append((cx, y, cx, y + len(rows) * row_h))
        for r, row in enumerate(rows):
            cx = x
            for c, cell in enumerate(row):
                if cell:
                    self.text(cx + 3, y + r * row_h + row_h - 5, cell)
                cx += col_widths[c]
        return self


def _esc(s: str) -> bytes:
    raw = s.encode("cp1252")
    return raw.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")


def _content(page: Page) -> bytes:
    out = bytearray()
    if page.image_only:
        out += b"q 400 0 0 500 100 200 cm /Im1 Do Q\n"
        return bytes(out)
    for x1, y1, x2, y2 in page.lines:
        out += f"{x1:.1f} {PAGE_H - y1:.1f} m {x2:.1f} {PAGE_H - y2:.1f} l S\n".encode()
    for x, y, s in page.texts:
        out += f"BT /F1 {page.font_size:g} Tf {x:.1f} {PAGE_H - y:.1f} Td (".encode()
        out += _esc(s) + b") Tj ET\n"
    return bytes(out)


def build_pdf(pages: list[Page], *, page_labels: list[tuple[int, str, int]] | None = None) -> bytes:
    """``page_labels``: ``[(start_index, style, first_number)]``, e.g. ``[(0, "D", 101)]``."""
    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    font = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
               b"/Encoding /WinAnsiEncoding >>")  # fmt: skip
    image = add(
        b"<< /Type /XObject /Subtype /Image /Width 2 /Height 2 /ColorSpace /DeviceGray "
        b"/BitsPerComponent 8 /Length 4 >>\nstream\n\x00\xff\xff\x00\nendstream"
    )
    pages_id = len(objects) + 2 * len(pages) + 1  # allocated right after the page objects
    page_ids: list[int] = []
    for page in pages:
        data = _content(page)
        content_id = add(b"<< /Length %d >>\nstream\n" % len(data) + data + b"\nendstream")
        res = b"<< /Font << /F1 %d 0 R >> /XObject << /Im1 %d 0 R >> >>" % (font, image)
        page_ids.append(
            add(
                b"<< /Type /Page /Parent %d 0 R /MediaBox [0 0 %d %d] /Resources %s "
                b"/Contents %d 0 R >>" % (pages_id, PAGE_W, PAGE_H, res, content_id)
            )
        )
    kids = b" ".join(b"%d 0 R" % i for i in page_ids)
    assert add(b"<< /Type /Pages /Kids [%s] /Count %d >>" % (kids, len(pages))) == pages_id
    labels = b""
    if page_labels:
        nums = b" ".join(
            b"%d << /S /%s /St %d >>" % (start, style.encode(), first)
            for start, style, first in page_labels
        )
        labels = b" /PageLabels << /Nums [%s] >>" % nums
    catalog = add(b"<< /Type /Catalog /Pages %d 0 R%s >>" % (pages_id, labels))

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root %d 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        catalog,
        xref,
    )
    return bytes(out)


# --------------------------------------------------------------------------- canned fixtures

SYNTHETIC_TITLE = "Synthetic radiosurgery cohort for testing obliteration reporting"
SYNTHETIC_DOI = "10.5555/synthetic.0001"


def synthetic_article(*, title: str = SYNTHETIC_TITLE, doi: str = SYNTHETIC_DOI) -> bytes:
    """A 4-page SYNTHETIC article: text, a ruled results table, an image-only page."""
    p1 = (
        Page()
        .text(50, 60, "SYNTHETIC TEST DOCUMENT - NOT A REAL STUDY")
        .text(50, 90, title)
        .text(50, 110, "Doe A, Roe B")
        .text(50, 125, f"doi: {doi}")
        .text(50, 160, "Methods")
        .text(50, 185, "We retrospectively reviewed 57 patients treated between 2005 and 2015")
        .text(50, 199, "at the Synthetic Institute. Median follow-up was 48 months (IQR 30-72).")
        .text(290, 820, "101")
    )
    p2 = (
        Page()
        .text(50, 60, "Results")
        .text(50, 85, "Complete obliteration was achieved in 45 of 57 patients (78.9%).")
        .text(50, 99, "The actuarial obliteration rate was 62.1% at 3 years and 81.4% at 5 years")
        .text(50, 113, "(95% CI 70.2-89.0) by the Kaplan-Meier method.")
        .text(50, 150, "Table 2. Outcomes by timepoint")
        .table(
            50,
            165,
            [170, 90, 90],
            [
                ["Outcome", "3 years", "5 years"],
                ["Obliteration (KM, %)", "62.1", "81.4"],
                ["Number at risk", "40", "22"],
                ["Haemorrhage events", "2", "3"],
            ],
        )
        .text(290, 820, "102")
    )
    p3 = Page(image_only=True)
    p4 = (
        Page()
        .text(50, 60, "Figure 1. Synthetic Kaplan-Meier curve (not real data).")
        .text(50, 90, "Supplementary Table S1 lists all adverse events.")
        .text(290, 820, "104")
    )
    return build_pdf([p1, p2, p3, p4], page_labels=[(0, "D", 101)])
