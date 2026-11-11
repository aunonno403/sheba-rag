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


def test_short_numbered_list_is_packed_not_one_chunk_per_line():
    offices = "Currently following e-Passport Offices are functional:\n" + "\n".join(
        f"{i}. Office number {i}" for i in range(1, 41))
    chunks = sentence_chunks([{"page": None, "text": offices}], max_chars=1000)
    assert len(chunks) == 3  # was 41: one tiny chunk per office
    assert all(len(c["text"]) >= 300 for c in chunks[:-1])  # each closes at the first section after 300 chars
    assert [c["section"] for c in chunks] == ["1–13", "14–28", "29–40"]


def test_table_rows_keep_their_headings_across_chunks():
    def group(name, currency):
        return f"e-Passport fees for {name}\n" + "\n".join(
            f"e-Passport with {pages} pages and {years} years validity\n"
            f"- Regular delivery: {currency} {pages}{years}1\n- Express delivery: {currency} {pages}{years}2"
            for pages in (48, 64) for years in (5, 10))
    text = "Fees can be paid online or at a bank.\n" + group("inside Bangladesh", "TK") + "\n" + group(
        "Bangladesh Missions", "USD")
    chunks = sentence_chunks([{"page": None, "text": text}], max_chars=330, min_chars=0)
    assert len(chunks) >= 3
    for c in chunks[1:]:
        assert c["text"].startswith("e-Passport fees for ")  # the group heading comes along
        assert not c["text"].split(" e-Passport with")[0].count("delivery")  # no orphan price row before a heading
    for c in chunks:
        assert not c["text"].endswith("validity")  # no heading left behind without its rows
    # every price sits in a chunk that also names its group and its passport type
    for name, cur in (("inside Bangladesh", "TK"), ("Bangladesh Missions", "USD")):
        for pages in (48, 64):
            for years in (5, 10):
                row = f"- Express delivery: {cur} {pages}{years}2"
                home = next(c["text"] for c in chunks if row in c["text"])
                assert f"fees for {name}" in home
                before = home[:home.index(row)]
                assert before.rstrip().rsplit("e-Passport with ", 1)[1].startswith(f"{pages} pages and {years} years")


def test_sentence_chunks_respect_sections_and_size():
    pages = [{"page": None, "text": LAW}]
    chunks = sentence_chunks(pages, max_chars=60, min_chars=0)
    assert all(len(c["text"]) <= 60 for c in chunks)
    # no chunk mixes text from two sections
    assert not any("১। (১)" in c["text"] and "২। “" in c["text"] for c in chunks)
    assert {c["section"] for c in chunks} >= {"১", "২", "১০"}


def test_sentence_chunks_overlap_one_sentence():
    text = "১। ক খ গ। ঘ ঙ চ। ছ জ ঝ। ঞ ট ঠ।"
    chunks = sentence_chunks([{"page": 3, "text": text}], max_chars=20, min_chars=0)
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
