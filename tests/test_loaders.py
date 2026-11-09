from sheba_rag.loaders import broken_ratio, needs_ocr

CLEAN = "সেবা প্রদান প্রতিশ্রুতি। ভূমি মন্ত্রণালয় দক্ষ, আধুনিক ও টেকসই ভূমি ব্যবস্থাপনা নিশ্চিত করে।"
# Real PyMuPDF output from the Nikosh-font land ministry citizen charter
BROKEN = "সসবা প্রোন প্রমতশ্রুমত। ভূমি িন্ত্রণালয় েক্ষ, আধুমনক ও সর্কসই ভূমি ব্যবস্থাপনা বাাংলাদেশ"


def test_clean_bangla_scores_zero():
    assert broken_ratio(CLEAN) == 0.0


def test_broken_nikosh_text_scores_high():
    assert broken_ratio(BROKEN) > 0.05


def test_english_text_is_not_broken():
    assert broken_ratio("The Bangladesh Passport Order, 1973.") == 0.0


def test_needs_ocr_for_empty_or_broken_pages():
    assert needs_ocr("   ")
    assert needs_ocr(BROKEN * 3)
    assert not needs_ocr(CLEAN * 3)


def test_pdf_page_without_usable_text_is_ocrd_or_skipped(monkeypatch, tmp_path):
    import pymupdf
    from sheba_rag import loaders
    path = tmp_path / "scan.pdf"
    with pymupdf.open() as doc:  # one page with almost no text layer, like a scan
        doc.new_page().insert_text((72, 72), "x")
        doc.new_page().insert_text((72, 72), "x")
        doc.save(path)
    replies = iter([CLEAN * 2, BROKEN * 3])  # page 1: OCR works; page 2: OCR is garbage too
    monkeypatch.setattr(loaders, "ocr_page", lambda page: next(replies))
    pages = loaders.pdf_pages(path, log=lambda msg: None)
    assert [p["page"] for p in pages] == [1]
    assert pages[0]["text"].startswith("সেবা প্রদান প্রতিশ্রুতি")


def test_html_keeps_bold_faq_questions():
    from sheba_rag.loaders import html_pages
    html = """<html><body><article><h1>FAQ</h1><div class="faq-items">
    <div class="faq-item"><strong>১। প্রশ্নঃ কার্ডের তথ্য কিভাবে সংশোধন করা যায়?</strong>
    <p>উত্তরঃ অনলাইন বা নির্বাচন অফিসে ভুল তথ্য সংশোধনের জন্য আবেদন করতে হবে। পর্যাপ্ত দলিলাদি সংযুক্ত করতে হবে।</p></div>
    <div class="faq-item"><strong>২। প্রশ্নঃ সংশোধনের রেকর্ড রাখা হয় কি?</strong>
    <p>উত্তরঃ সকল সংশোধনের রেকর্ড সেন্ট্রাল ডাটাবেজে সংরক্ষিত থাকে এবং তা যেকোনো সময় যাচাই করা যায়।</p></div>
    </div></article></body></html>"""
    text = html_pages(html.encode("utf-8"))[0]["text"]
    assert "প্রশ্নঃ কার্ডের তথ্য কিভাবে সংশোধন করা যায়?" in text
    assert "উত্তরঃ সকল সংশোধনের রেকর্ড" in text
