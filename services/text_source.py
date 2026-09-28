from __future__ import annotations

import io
import re
import zipfile

import httpx
from bs4 import BeautifulSoup

# Section headings to remove (normalized: lowercase, letters only).
REMOVE_SECTIONS = {
    "contents",
    "tableofcontents",
    "listofcontents",
    "illustrations",
    "listofillustrations",
    "preface",
}

# Contents-list entries: "CHAPTER I. Title" style lines (number + title on one line).
LIST_ENTRY_RE = re.compile(
    r"^(?:CHAPTER|PART|BOOK|SECTION)\s+(?:[IVXLCDM]+|\d+)\s*\.?\s+\S",
    re.IGNORECASE,
)
# Contents-list entries without a keyword: "I.—A SCANDAL IN BOHEMIA 3", "1. Title".
NUMBERED_ENTRY_RE = re.compile(r"^(?:[IVXLCDM]+|\d+)\s*[.)]\s*[—–-]*\s*\S")

# Table-of-contents headings (the section whose first entry names the first chapter).
CONTENTS_SECTIONS = {"contents", "tableofcontents", "listofcontents"}
# Column headers that may precede the first real contents entry.
_CONTENTS_COLUMN_HEADERS = {"page", "pages", "chapter", "chap", "no", "pageno", "chapterpage"}
# Leading chapter numbering of a contents entry / body heading: "I.—", "CHAPTER I.", "1.".
_ENTRY_PREFIX_RE = re.compile(
    r"^(?:(?:chapter|part|book|section|adventure|story)\s+)?(?P<num>[ivxlcdm]+|\d+)\b\s*[.:)—–-]*\s*",
    re.IGNORECASE,
)
# Trailing page number of a contents entry: "A SCANDAL IN BOHEMIA     3".
_PAGE_SUFFIX_RE = re.compile(r"\s+\d+\s*$")
# A bare chapter number line ("I", "I.", "12") under a chapter title.
_BARE_NUMBER_RE = re.compile(r"^(?:[IVXLCDM]+|\d+)\.?$")
# A keyword + number line with no title ("CHAPTER I.", "Adventure 2").
_KEYWORD_NUMBER_RE = re.compile(
    r"^(?:chapter|part|book|section|adventure|story)\s+(?:[ivxlcdm]+|\d+)\.?$",
    re.IGNORECASE,
)

START_MARKER_RE = re.compile(
    r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\n",
    re.IGNORECASE,
)
END_MARKER_RE = re.compile(
    r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK",
    re.IGNORECASE,
)
CAPTION_RE = re.compile(r"(?m)^\s*\[[^\]]*\]\s*$")

# Chapter-style headings ("CHAPTER I.", "Chapter 1", "PART ONE", "Adventure II", ...).
# The keyword must be followed by a chapter number (roman, arabic or a number
# word) so ordinary prose lines starting with "part of ..." / "book that ..."
# are not mistaken for headings.
_NUMBER_WORDS = (
    "one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|"
    "fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|"
    "first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last"
)
CHAPTER_HEADING_RE = re.compile(
    rf"^(?:chapter|part|book|section|adventure)\s+(?:[ivxlcdm]+|\d+|(?:the\s+)?(?:{_NUMBER_WORDS}))\b",
    re.IGNORECASE,
)
# If front-matter removal keeps less than this fraction of the text AND
# discards more text than any plausible contents/preface section, the
# heuristics misfired (e.g. a book with no CHAPTER headings); fall back
# rather than narrate an empty book. Small all-front-matter slices (as
# produced by the story narrator) are still legitimately emptied.
_MIN_KEEP_RATIO = 0.05
_MAX_FRONT_MATTER_CHARS = 20_000

# Gutenberg marks italics/emphasis with underscores; in fiction this
# convention denotes quoted speech: _Mr. Fox, are you coming_ -> "Mr. Fox, are you coming".
UNDERSCORE_QUOTE_RE = re.compile(r"_([^_\n]+)_")

# A paragraph ending with one of these continues onto the next paragraph
# (Gutenberg list formatting: "were--\n\n Flopsy,\n Mopsy,").
_CONTINUATION_END_RE = re.compile(r"(?:--|—|–|,|:|;)\s*$")


def sanitize_text(text: str) -> str:
    """Sanitize Gutenberg text for TTS narration.

    Converts underscore-wrapped text (the Gutenberg italics convention, used
    for quoted speech in fiction) to double quotes so the narrator reads it
    as dialogue.
    """
    return UNDERSCORE_QUOTE_RE.sub(r'"\1"', text)


def strip_gutenberg_boilerplate(text: str) -> str:
    """Remove the Gutenberg license header and footer."""
    start = START_MARKER_RE.search(text)
    if start:
        text = text[start.end():]
    end = END_MARKER_RE.search(text)
    if end:
        text = text[:end.start()]
    return text


def _is_heading(line: str) -> bool:
    """True if the line looks like a section heading (short, all-caps or title-case)."""
    s = line.strip()
    if not s or len(s) > 80:
        return False
    cleaned_s = s.rstrip(".:?!")
    normalized_cleaned = re.sub(r"[^a-z]", "", cleaned_s.lower())
    if normalized_cleaned in REMOVE_SECTIONS or normalized_cleaned.startswith("preface"):
        return True
    # Reject sentence-ending punctuation, except "CHAPTER I." style headings.
    if s.endswith((".", "!", "?")) and not (
        s.endswith(".") and re.fullmatch(r"\S+\s+\S+\.", s) and CHAPTER_HEADING_RE.match(s)
    ):
        return False
    if s.isupper():
        return True
    words = s.split()
    capitalized = sum(1 for word in words if word[:1].isupper())
    return capitalized >= max(1, len(words) - 1)


def _section_name(line: str) -> str | None:
    """Normalized name of a removable section heading, or None."""
    s = line.strip()
    if not s or len(s) > 80:
        return None
    normalized = re.sub(r"[^a-z]", "", s.lower().rstrip(".:?!"))
    if normalized in REMOVE_SECTIONS or normalized.startswith("preface"):
        if _is_heading(line):
            return normalized
        # Period-ending headings like "PREFACE." / "List of Illustrations."
        # (common in Gutenberg HTML) are rejected by _is_heading; accept short
        # heading-like lines ending in punctuation instead.
        if len(s) <= 30 and s.endswith((".", ":")):
            return normalized
    return None


def _is_list_entry(line: str) -> bool:
    """True if the line looks like a table-of-contents entry."""
    s = line.strip()
    return bool(LIST_ENTRY_RE.match(s) or NUMBERED_ENTRY_RE.match(s))


def _next_nonblank(lines: list[str], j: int) -> int:
    while j < len(lines) and not lines[j].strip():
        j += 1
    return j


def _is_followed_by_content(lines: list[str], index: int) -> bool:
    """True if the heading at index is followed by real prose (the story body)."""
    j = _next_nonblank(lines, index + 1)
    if j >= len(lines):
        return False

    line_after = lines[j].strip()
    if _is_list_entry(line_after) or _is_chapter_heading(line_after):
        return False

    # Skip up to two subtitle lines (e.g. "Down the Rabbit-Hole" under
    # "CHAPTER I.", or "A SCANDAL IN BOHEMIA" + "I" under "Adventure I").
    for _ in range(2):
        if not (_is_heading(lines[j]) and len(lines[j].strip()) < 60):
            break
        j = _next_nonblank(lines, j + 1)
        if j >= len(lines):
            return False
        sub_after = lines[j].strip()
        if _is_list_entry(sub_after) or _is_chapter_heading(sub_after):
            return False

    upcoming_lines_checked = 0
    scan_idx = j
    while scan_idx < len(lines) and upcoming_lines_checked < 15:
        curr = lines[scan_idx].strip()
        if curr:
            upcoming_lines_checked += 1
            if _is_chapter_heading(curr):
                return False
        scan_idx += 1

    return len(lines[j].strip()) >= 40


def _is_chapter_heading(line: str) -> bool:
    """True if the line looks like a chapter/part/book/section heading."""
    s = line.strip()
    return bool(CHAPTER_HEADING_RE.match(s)) and _is_heading(s)


def _entry_key(line: str, strip_prefix: bool = True) -> str:
    """Normalize a contents entry or heading so both spellings compare equal.

    "I.—A SCANDAL IN BOHEMIA     3" and "A SCANDAL IN BOHEMIA" both become
    "ascandalinbohemia".
    """
    s = line.strip()
    # Drop a trailing page number, unless the number is the chapter number
    # itself ("Chapter 1").
    if not _KEYWORD_NUMBER_RE.match(s):
        s = _PAGE_SUFFIX_RE.sub("", s)
    if strip_prefix:
        s = _ENTRY_PREFIX_RE.sub("", s)
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _entry_number(line: str) -> str | None:
    """Chapter number at the start of a contents entry or heading ("i", "12"), or None."""
    m = _ENTRY_PREFIX_RE.match(line.strip())
    return m.group("num").lower() if m else None


def _include_preceding_headings(lines: list[str], index: int, lower_bound: int, number: str | None) -> int:
    """Walk back over number-only heading lines directly above a chapter title.

    "Adventure I" above "A SCANDAL IN BOHEMIA", or "CHAPTER I." above "Down
    the Rabbit-Hole", belong to the chapter. Lines carrying a title of their
    own (e.g. a contents entry) are never crossed, and a matched line that
    already starts with a number is taken as the heading itself.
    """
    if _ENTRY_PREFIX_RE.match(lines[index].strip()):
        return index
    start = index
    steps = 0
    k = index - 1
    while k >= lower_bound and steps < 2:
        s = lines[k].strip()
        if not s:
            k -= 1
            continue
        if (_KEYWORD_NUMBER_RE.match(s) or _BARE_NUMBER_RE.match(s)) and (
            number is None or _entry_number(s) == number
        ):
            start = k
            steps += 1
            k -= 1
            continue
        break
    return start


def _find_body_start_from_contents(lines: list[str], contents_index: int) -> int | None:
    """Locate the story start from the table of contents.

    The first contents entry names the first chapter; its next occurrence in
    the text is the chapter heading where reading starts. The occurrence is
    a line whose only content is that title, compared ignoring case,
    punctuation and any trailing page number. Only when no such line exists
    is a numbered line accepted, and then only with the same chapter number
    (so a later contents entry sharing the title is never chosen). Returns
    that line index (extended to include a chapter-number line directly
    above it), or None when no second occurrence exists.
    """
    j = _next_nonblank(lines, contents_index + 1)
    while j < len(lines):
        key = _entry_key(lines[j], strip_prefix=False)
        if key and key not in _CONTENTS_COLUMN_HEADERS:
            break
        j = _next_nonblank(lines, j + 1)
    if j >= len(lines):
        return None
    number = _entry_number(lines[j])
    title_key = _entry_key(lines[j], strip_prefix=True)

    # 1. A line holding nothing but the title (case-insensitive).
    if len(title_key) >= 2:
        for k in range(j + 1, len(lines)):
            if lines[k].strip() and _entry_key(lines[k], strip_prefix=False) == title_key:
                return _include_preceding_headings(lines, k, j + 1, number)

    # 2. A numbered heading repeating the entry ("1. Title", "Chapter 1"),
    #    with the same number: several chapters may share a title, and a
    #    later contents entry must never be taken for the story start.
    for strip_prefix in (True, False):
        key = _entry_key(lines[j], strip_prefix)
        if len(key) < 2:
            continue
        for k in range(j + 1, len(lines)):
            if not lines[k].strip() or _entry_key(lines[k], strip_prefix) != key:
                continue
            candidate_number = _entry_number(lines[k])
            if candidate_number is not None and number is not None and candidate_number != number:
                continue
            return _include_preceding_headings(lines, k, j + 1, number)
    return None


def _is_body_start(lines: list[str], index: int, require_chapter: bool = False) -> bool:
    """True if the line at index begins the story body."""
    if not _is_heading(lines[index]):
        return False
    if _is_list_entry(lines[index]):
        return False
    if require_chapter and not _is_chapter_heading(lines[index]):
        return False
    return _is_followed_by_content(lines, index)


def remove_front_matter(text: str) -> str:
    """Remove CONTENTS / ILLUSTRATIONS / PREFACE sections, keeping the story body.

    After a CONTENTS heading the first entry names the first chapter, and its
    next occurrence in the text is where reading starts; everything in
    between (contents list, illustrations, preface) is dropped. When no such
    anchor is found, heading heuristics locate the body instead: when the
    text contains chapter headings, front matter is skipped until a chapter
    heading followed by prose (so preface bodies with heading-like
    attribution lines are not mistaken for the story start).
    """
    lines = text.splitlines()
    has_chapter = any(
        _is_chapter_heading(line) and _is_followed_by_content(lines, i)
        for i, line in enumerate(lines)
    )
    total = sum(len(line.strip()) for line in lines)
    for require_chapter in ([True, False] if has_chapter else [False]):
        out = _strip_sections(lines, require_chapter)
        kept = sum(len(line.strip()) for line in out)
        dropped = total - kept
        if kept >= total * _MIN_KEEP_RATIO or dropped <= _MAX_FRONT_MATTER_CHARS:
            return "\n".join(out)
    # Heuristics would have discarded (almost) the whole book: keep it as is.
    return text


def _strip_sections(lines: list[str], require_chapter: bool) -> list[str]:
    """Drop removable sections, resuming at the first detected body start."""
    out: list[str] = []
    skipping = False
    skip_until: int | None = None
    for index, line in enumerate(lines):
        if skip_until is not None:
            if index < skip_until:
                continue
            skip_until = None
            skipping = False
        if not skipping:
            name = _section_name(line)
            if name:
                skipping = True
                if name in CONTENTS_SECTIONS:
                    skip_until = _find_body_start_from_contents(lines, index)
                continue
        if skipping:
            if _is_body_start(lines, index, require_chapter=require_chapter):
                skipping = False
            else:
                continue
        out.append(line)
    return out


def clean_whitespace(text: str) -> str:
    """Drop bracketed captions and collapse runs of blank lines."""
    text = CAPTION_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def join_wrapped_lines(text: str) -> str:
    """Join lines wrapped at the Gutenberg ~70-char column limit.

    Plain-text Gutenberg books wrap every line at ~70 characters, breaking
    sentences mid-flow ("...the root of a\\nvery big fir-tree."). Lines within
    a paragraph (no blank line between them) are joined with a single space.
    A paragraph ending with a continuation signal (comma, colon, em-dash)
    is also joined with the next one, covering Gutenberg list formatting
    ("were--\\n\\n Flopsy,\\n Mopsy,"). Blank lines between complete
    paragraphs are preserved.
    """
    paragraphs = re.split(r"\n\s*\n", text.strip())
    joined: list[str] = []
    for para in paragraphs:
        line = " ".join(para.split())
        if joined and _CONTINUATION_END_RE.search(joined[-1]):
            joined[-1] = f"{joined[-1]} {line}"
        else:
            joined.append(line)
    return "\n\n".join(joined) + "\n"


def curate_text(text: str) -> str:
    """Curate a fetched URL text for audiobook narration.

    Sanitizes Gutenberg conventions (underscore-wrapped speech -> quotes),
    strips the Gutenberg license boilerplate, removes front-matter sections
    (CONTENTS / ILLUSTRATIONS / PREFACE), joins lines wrapped at the
    ~70-char column limit, and cleans whitespace so only the story body
    remains.
    """
    text = sanitize_text(text)
    text = strip_gutenberg_boilerplate(text)
    text = remove_front_matter(text)
    text = clean_whitespace(text)
    return join_wrapped_lines(text)


# Block-level elements that start a new paragraph in the extracted text.
_BLOCK_TAGS = [
    "p", "div", "h1", "h2", "h3", "h4", "h5", "h6",
    "li", "blockquote", "pre", "table", "ul", "ol", "dl",
]


def _extract_html_text(html: str) -> tuple[str, str | None]:
    """Extract title and body text from an HTML document."""
    soup = BeautifulSoup(html, "lxml")
    title = soup.title.string.strip() if soup.title and soup.title.string else None
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "noscript"]):
        tag.decompose()
    # Separate block elements with a blank line so paragraph boundaries
    # survive the flat text extraction (join_wrapped_lines relies on them).
    for tag in soup.find_all(_BLOCK_TAGS):
        tag.insert_after("\n")
    text = soup.get_text(separator="\n")
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text, title


# Gutenberg HTML zips ship the main document as a .htm/.html entry plus images.
HTML_EXTENSIONS = (".htm", ".html")
MAX_ZIP_HTML_SIZE = 20 * 1024 * 1024  # 20 MB uncompressed cap (zip-bomb guard)


def _zip_html_content(data: bytes) -> str:
    """Read the main HTML entry from a Gutenberg-style zip archive."""
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        html_entries = [
            i for i in zf.infolist()
            if i.filename.lower().endswith(HTML_EXTENSIONS) and not i.is_dir()
        ]
        if not html_entries:
            raise ValueError("No HTML file found in the zip archive")
        entry = max(html_entries, key=lambda i: i.file_size)
        if entry.file_size > MAX_ZIP_HTML_SIZE:
            raise ValueError("HTML entry too large")
        raw = zf.read(entry)
    return raw.decode("utf-8", errors="replace")


def fetch_text(url: str, curate: bool = True) -> tuple[str, str | None]:
    """Fetch a URL and extract its text.

    Supports plain text/HTML URLs and Gutenberg-style HTML zip archives
    (e.g. https://www.gutenberg.org/cache/epub/1342/pg1342-h.zip): the zip
    is read in memory and its main HTML entry is parsed.

    With curate=True (default) the extracted text is filtered for audiobook
    narration (Gutenberg boilerplate, front matter, whitespace). With
    curate=False the raw extracted text is returned unchanged, e.g. so the
    LLM story planner can see character lists / Dramatis Personae.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Linux; Android 10) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36"
    }
    is_zip = url.lower().endswith(".zip")
    with httpx.Client(timeout=60.0 if is_zip else 30.0, follow_redirects=True, headers=headers) as client:
        resp = client.get(url)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "").lower()
        if is_zip or "zip" in content_type:
            html = _zip_html_content(resp.content)
        else:
            html = resp.text
    text, title = _extract_html_text(html)
    if curate:
        text = curate_text(text)
    return text, title
