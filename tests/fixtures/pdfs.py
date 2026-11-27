"""Small, real PDFs built in-test, with no PDF library.

The extractor's own PDF engine (pypdfium2) is what the tests exercise, so the
documents they feed it are written by hand here: a text page is a Helvetica
``Tj`` and a scan page is a JPEG drawn over the whole page, which is what a
scanned book is.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

PAGE_SIZE = 300


def _escape(text: str) -> bytes:
    return (
        text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)").encode("latin-1")
    )


def build_pdf(pages: list[str | bytes], size: int = PAGE_SIZE) -> bytes:
    """A PDF with one *size* x *size* point page per entry of *pages*.

    A ``str`` entry is a page holding that text near its top-left corner; a
    ``bytes`` entry is a page filled by that grayscale JPEG.
    """
    objects: dict[int, bytes] = {
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    page_ids = []
    next_id = 4
    for page in pages:
        page_id, contents_id = next_id, next_id + 1
        next_id += 2
        if isinstance(page, str):
            resources = b"<< /Font << /F1 3 0 R >> >>"
            stream = b"BT /F1 12 Tf 20 %d Td (%s) Tj ET" % (size - 40, _escape(page))
        else:
            image_id = next_id
            next_id += 1
            with Image.open(io.BytesIO(page)) as img:
                width, height = img.size
            objects[image_id] = (
                b"<< /Type /XObject /Subtype /Image /Width %d /Height %d "
                b"/ColorSpace /DeviceGray /BitsPerComponent 8 /Filter /DCTDecode "
                b"/Length %d >>\nstream\n%s\nendstream" % (width, height, len(page), page)
            )
            resources = b"<< /XObject << /Im0 %d 0 R >> >>" % image_id
            stream = b"q %d 0 0 %d 0 0 cm /Im0 Do Q" % (size, size)
        objects[page_id] = (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %d %d] "
            b"/Resources %s /Contents %d 0 R >>" % (size, size, resources, contents_id)
        )
        objects[contents_id] = b"<< /Length %d >>\nstream\n%s\nendstream" % (
            len(stream),
            stream,
        )
        page_ids.append(page_id)

    kids = b" ".join(b"%d 0 R" % i for i in page_ids)
    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects[2] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (kids, len(page_ids))
    return _write(objects)


def _write(objects: dict[int, bytes]) -> bytes:
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = {}
    for obj_id in sorted(objects):
        offsets[obj_id] = out.tell()
        out.write(b"%d 0 obj\n%s\nendobj\n" % (obj_id, objects[obj_id]))
    xref = out.tell()
    count = max(objects) + 1
    out.write(b"xref\n0 %d\n0000000000 65535 f \n" % count)
    for obj_id in range(1, count):
        out.write(b"%010d 00000 n \n" % offsets[obj_id])
    out.write(b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (count, xref))
    return out.getvalue()


def form_pdf(label: str, value: str, size: int = PAGE_SIZE) -> bytes:
    """One page reading *label* in its content, with a text field filled in
    with *value*. The value exists only in the field's appearance stream, as
    in a filled-in form, not in the page content."""
    content = b"BT /F1 12 Tf 20 %d Td (%s) Tj ET" % (size - 40, _escape(label))
    appearance = b"/Tx BMC BT /F1 12 Tf 2 10 Td (%s) Tj ET EMC" % _escape(value)
    return _write({
        1: b"<< /Type /Catalog /Pages 2 0 R /AcroForm << /Fields [6 0 R] >> >>",
        2: b"<< /Type /Pages /Kids [4 0 R] /Count 1 >>",
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        4: b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %d %d] "
        b"/Resources << /Font << /F1 3 0 R >> >> /Contents 5 0 R /Annots [6 0 R] >>"
        % (size, size),
        5: b"<< /Length %d >>\nstream\n%s\nendstream" % (len(content), content),
        6: b"<< /Type /Annot /Subtype /Widget /FT /Tx /T (field) /V (%s) "
        b"/Rect [20 150 280 180] /F 4 /P 4 0 R /AP << /N 7 0 R >> >>" % _escape(value),
        7: b"<< /Type /XObject /Subtype /Form /BBox [0 0 260 30] "
        b"/Resources << /Font << /F1 3 0 R >> >> /Length %d >>\nstream\n%s\nendstream"
        % (len(appearance), appearance),
    })


def noise_jpeg(side: int = 400, seed: int = 0) -> bytes:
    """A grayscale JPEG of random noise: large, and incompressible like a scan."""
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 255, (side, side), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(pixels, "L").save(buf, "JPEG", quality=90)
    return buf.getvalue()


def text_pdf(pages: int) -> bytes:
    """*pages* pages reading ``page N body text``."""
    return build_pdf([f"page {i + 1} body text" for i in range(pages)])


def scan_pdf(pages: int, side: int = 400) -> bytes:
    """*pages* pages that are nothing but a noisy image, like a scanned book."""
    return build_pdf([noise_jpeg(side, seed=i) for i in range(pages)])


def page_count(data: bytes) -> int:
    """Pages in *data*, read by pypdfium2."""
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(data)
    try:
        return len(doc)
    finally:
        doc.close()
