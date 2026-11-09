from sheba_rag.chunking import split_sections, split_sentences, sentence_chunks, fixed_chunks

LAW = """প্রথম অধ্যায়
১। (১) এই আইন জাতীয় পরিচয় নিবন্ধন আইন, ২০২৩ নামে অভিহিত হইবে। (২) সরকার তারিখ নির্ধারণ করিবে।
২। “জাতীয় পরিচয়পত্র” অর্থ নিবন্ধক কর্তৃক প্রদত্ত পরিচয়পত্র।
১০। কোনো নাগরিক আবেদন করিতে পারিবেন।"""


def test_sections_split_on_bangla_numbers():
    secs = split_sections(LAW)
    assert [s for s, _ in secs] == [None, "১", "২", "১০"]
    assert secs[2][1].startswith("২। “জাতীয়")


def test_sections_split_on_english_numbers():
    secs = split_sections("Preamble.\n1. (1) This Order may be called X.\n2. Definitions here.")
    assert [s for s, _ in secs] == [None, "1", "2"]


def test_decimal_numbers_are_not_sections():
    secs = split_sections("২.১ নাগরিক সেবা: হটলাইন ১৬১২২\n৮.২৫ একর পর্যন্ত মওকুফ")
    assert [s for s, _ in secs] == [None]


def test_sentences_split_on_danda_and_english_period():
    s = split_sentences("প্রথম বাক্য। দ্বিতীয় বাক্য? Third one. Fourth")
    assert s == ["প্রথম বাক্য।", "দ্বিতীয় বাক্য?", "Third one.", "Fourth"]


def test_sentence_chunks_respect_sections_and_size():
    pages = [{"page": None, "text": LAW}]
    chunks = sentence_chunks(pages, max_chars=60)
    assert all(len(c["text"]) <= 60 for c in chunks)
    # no chunk mixes text from two sections
    assert not any("১। (১)" in c["text"] and "২। “" in c["text"] for c in chunks)
    assert {c["section"] for c in chunks} >= {"১", "২", "১০"}


def test_sentence_chunks_overlap_one_sentence():
    text = "১। ক খ গ। ঘ ঙ চ। ছ জ ঝ। ঞ ট ঠ।"
    chunks = sentence_chunks([{"page": 3, "text": text}], max_chars=20)
    assert len(chunks) >= 2
    last_of_first = split_sentences(chunks[0]["text"])[-1]
    assert chunks[1]["text"].startswith(last_of_first)
    assert all(c["page"] == 3 for c in chunks)


def test_fixed_chunks_window_and_section_label():
    text = "ক" * 50 + "\n৫। " + "খ" * 2000
    chunks = fixed_chunks([{"page": 1, "text": text}], size=1000, overlap=150)
    assert all(len(c["text"]) <= 1000 for c in chunks)
    assert chunks[1]["text"][:150] == chunks[0]["text"][-150:]
    assert chunks[0]["section"] is None and chunks[1]["section"] == "৫"
