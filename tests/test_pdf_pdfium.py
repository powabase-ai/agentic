"""PDFium (pypdfium2) behind every PDF primitive the extractor uses.

PyMuPDF (AGPL-3.0) was replaced by pypdfium2. These tests run the real
engine over small PDFs built in-test: page counting, page-batch splitting,
page rendering to PNG and text-layer extraction, plus the legacy method names
("fitz", "PyMuPDF") that stored configurations may still carry.
"""

import ast
import io
import pathlib
import sys

import pypdfium2 as pdfium
import pytest
from PIL import Image

from agentic.ingest import PDFExtractor, RawContent
from agentic.ingest.extractor import pdf as pdf_module
from agentic.knowledge.indexing._pageindex_lib.utils import get_page_tokens
from tests.fixtures.pdfs import (
    build_pdf,
    form_pdf,
    noise_jpeg,
    page_count,
    scan_pdf,
    text_pdf,
)

SRC = pathlib.Path(__file__).resolve().parent.parent / "src"


def _raw(data: bytes, **metadata) -> RawContent:
    return RawContent(
        content=data,
        mime_type="application/pdf",
        source_uri="upload://doc.pdf",
        filename="doc.pdf",
        metadata=metadata,
    )


class TestPageCount:
    def test_counts_pages(self):
        assert pdf_module._count_pages(text_pdf(1)) == 1
        assert pdf_module._count_pages(text_pdf(7)) == 7

    def test_regex_fallback_without_pypdfium2(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pypdfium2", None)

        assert pdf_module._count_pages(text_pdf(4)) == 4

    def test_unreadable_pdf_raises(self):
        with pytest.raises(pdfium.PdfiumError):
            pdf_module._count_pages(b"not a pdf at all")


class TestBatchSplit:
    def test_batches_cover_every_page_in_order(self):
        data = text_pdf(10)

        ranges = pdf_module._plan_pdf_batches(data, max_pages=4, max_bytes=10**9)
        batches = list(pdf_module._iter_pdf_batches(data, ranges))

        assert ranges == [(0, 3), (4, 7), (8, 9)]
        assert [(start, end) for _, start, end in batches] == [(1, 4), (5, 8), (9, 10)]
        assert [page_count(pdf) for pdf, _, _ in batches] == [4, 4, 2]

    def test_a_batch_holds_the_pages_it_names(self):
        data = text_pdf(5)
        ranges = [(1, 2)]

        (batch, start, end), = pdf_module._iter_pdf_batches(data, ranges)
        result = PDFExtractor()._extract_pdfium(_raw(batch))

        pages = [d.content for d in result.derivatives if d.type == "page_text"]
        assert (start, end) == (2, 3)
        assert pages == ["page 2 body text", "page 3 body text"]

    def test_byte_limit_splits_large_pages(self):
        data = scan_pdf(4)
        limit = len(data) // 2

        ranges = pdf_module._plan_pdf_batches(data, max_pages=1000, max_bytes=limit)

        assert len(ranges) > 1
        for pdf, _, _ in pdf_module._iter_pdf_batches(data, ranges):
            assert len(pdf) <= limit

    def test_a_single_page_over_the_limit_is_refused(self):
        data = scan_pdf(1)

        with pytest.raises(pdf_module._BatchTooLarge):
            pdf_module._plan_pdf_batches(data, max_pages=10, max_bytes=1000)


class TestRenderToPng:
    @pytest.mark.parametrize(("dpi", "side"), [(72, 300), (150, 625)])
    def test_pages_render_to_png_at_the_requested_dpi(self, dpi, side):
        images = PDFExtractor()._render_page_images(_raw(text_pdf(2)), dpi=dpi)

        assert [d.page for d in images] == [1, 2]
        for d in images:
            assert d.type == "image"
            assert d.format == "png"
            assert d.metadata == {"width": side, "height": side, "dpi": dpi}
            with Image.open(io.BytesIO(d.content)) as img:
                assert img.format == "PNG"
                assert img.mode == "RGB"  # opaque, like alpha=False before
                assert img.size == (side, side)

    def test_letter_height_is_not_rounded_up_a_pixel(self):
        """792 pt x 150/72 is 1650.0000000000002 in floating point. PDFium
        rounds bitmap sizes up, which would make every US Letter page 1651 px
        tall where MuPDF made it 1650."""
        (d,) = PDFExtractor()._render_page_images(_raw(build_pdf(["x"], size=792)))

        assert (d.metadata["width"], d.metadata["height"]) == (1650, 1650)

    def test_text_page_renders_white_with_dark_text(self):
        (d,) = PDFExtractor()._render_page_images(_raw(text_pdf(1)))

        with Image.open(io.BytesIO(d.content)) as img:
            lo, hi = img.convert("L").getextrema()
        assert hi == 255  # white background, not transparent black
        assert lo < 128  # the text was drawn

    def test_filled_form_fields_are_drawn(self):
        """Without a form environment PDFium leaves form fields blank."""
        def ink(data):
            (d,) = PDFExtractor()._render_page_images(_raw(data), dpi=72)
            with Image.open(io.BytesIO(d.content)) as img:
                # The field's area only (PDF y 150-180 is image y 120-150).
                region = img.convert("L").crop((20, 120, 280, 150))
                return sum(1 for v in region.getdata() if v < 128)

        assert ink(form_pdf("Name:", "")) == 0
        assert ink(form_pdf("Name:", "Ada Lovelace")) > 50

    def test_scan_page_renders(self):
        (d,) = PDFExtractor()._render_page_images(_raw(build_pdf([noise_jpeg(200)])))

        with Image.open(io.BytesIO(d.content)) as img:
            assert img.size == (625, 625)

    def test_unreadable_pdf_degrades_to_no_images(self):
        assert PDFExtractor()._render_page_images(_raw(b"%PDF-broken")) == []


class TestTextExtraction:
    @pytest.mark.parametrize("model", ["pdfium", "fitz", "PyMuPDF", "pymupdf"])
    @pytest.mark.asyncio
    async def test_method_and_legacy_aliases_extract_the_text_layer(self, model):
        result = await PDFExtractor().extract(_raw(text_pdf(3), extraction_model=model))

        assert result.extraction_method == "pdfium"
        pages = [d for d in result.derivatives if d.type == "page_text"]
        assert [d.page for d in pages] == [1, 2, 3]
        assert [d.content for d in pages] == [f"page {n} body text" for n in (1, 2, 3)]
        (text,) = [d for d in result.derivatives if d.type == "text"]
        assert text.content == "\n\n".join(d.content for d in pages)
        assert result.auto_metadata["page_count"] == 3
        assert result.auto_metadata["char_count"] == len(text.content)
        assert [d.page for d in result.derivatives if d.type == "image"] == [1, 2, 3]

    def test_filled_form_field_values_are_part_of_the_text(self):
        """MuPDF read the values of filled form fields; PDFium's text page
        alone does not, so the page is flattened first."""
        result = PDFExtractor()._extract_pdfium(_raw(form_pdf("Name:", "Ada Lovelace")))

        (page,) = [d.content for d in result.derivatives if d.type == "page_text"]
        assert "Name:" in page
        assert "Ada Lovelace" in page

    def test_scan_page_has_empty_text(self):
        result = PDFExtractor()._extract_pdfium(_raw(scan_pdf(2)))

        assert [d.content for d in result.derivatives if d.type == "page_text"] == ["", ""]

    def test_page_text_is_stripped_non_blank_lines(self):
        assert pdf_module._clean_page_text("  one \r\n\r\ntwo\x00\r\n  \r\n") == "one\ntwo"

    @pytest.mark.asyncio
    async def test_legacy_name_in_a_host_fallback_chain(self, monkeypatch):
        monkeypatch.setattr(
            "agentic.knowledge.model_config.EXTRACTION_FALLBACK_CHAIN", ["fitz"]
        )

        result = await PDFExtractor().extract(_raw(text_pdf(1), extraction_model="auto"))

        assert result.extraction_method == "pdfium"


class TestPageIndexParser:
    def test_page_tokens_include_form_field_values(self):
        pages = get_page_tokens(io.BytesIO(form_pdf("Name:", "Ada")), pdf_parser="pdfium")

        assert "Ada" in pages[0][0]

    @pytest.mark.parametrize("parser", ["pdfium", "PyMuPDF"])
    def test_page_tokens_from_a_stream(self, parser):
        pages = get_page_tokens(io.BytesIO(text_pdf(2)), pdf_parser=parser)

        assert [text for text, _ in pages] == ["page 1 body text", "page 2 body text"]
        assert all(tokens > 0 for _, tokens in pages)

    def test_page_tokens_from_a_path(self, tmp_path):
        path = tmp_path / "doc.pdf"
        path.write_bytes(text_pdf(3))

        pages = get_page_tokens(str(path), pdf_parser="pdfium")

        assert len(pages) == 3

    def test_unsupported_input_is_an_error(self):
        with pytest.raises(ValueError):
            get_page_tokens("not-a-file.txt", pdf_parser="pdfium")


def test_no_source_module_imports_pymupdf():
    """PyMuPDF is AGPL-3.0; the library must not depend on it again."""
    offenders = []
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if name.split(".")[0].lower() in ("fitz", "pymupdf"):
                    offenders.append(f"{path.relative_to(SRC)}:{node.lineno}")
    assert offenders == []
