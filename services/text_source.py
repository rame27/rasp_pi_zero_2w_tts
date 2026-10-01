from __future__ import annotations

import io
import re
import zipfile
from collections import Counter

import httpx
from bs4 import BeautifulSoup

START_MARKER_RE = re.compile(
    r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\n",
    re.IGNORECASE,
)
END_MARKER_RE = re.compile(
    r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK",
    re.IGNORECASE,
)
CAPTION_RE = re.compile(r"(?m)^\s*\[[^\]]*\]\s*$")

# If front-matter removal keeps less than this fraction of the text AND
# discards more text than any plausible front matter, the structural cut
# misfired (e.g. the body was never located); fall back rather than narrate
# an empty book. Small all-front-matter slices (as produced by the story
# narrator) are still legitimately emptied.
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


# --- Structural front-matter detection (ported from preaudio.py) ------------
# Detection is structural, not name based: the body starts at the first
# heading belonging to the document's dominant repeating heading pattern --
# the chapter/part/section skeleton -- which is what a real book structure
# looks like in every Gutenberg edition regardless of how its headings are
# spelled. Ported from preaudio.py (kept standalone and stdlib-only there).

# A line of asterisks, dashes, underscores, equals signs or dots is an
# ornament (section break, rule), never a heading.
DECORATION_RE = re.compile(r"^[\s*+=~_.\-]{3,}$")
# A single trailing dot after a lone letter or digit is numbering punctuation
# ("IV.", "12."); the same dot after a word is sentence punctuation.
TRAILING_NUMERAL_DOT_RE = re.compile(r"(?:\b[A-Za-z]|\d)\.$")
# Punctuation that means the line runs on into a sentence.
TRAILING_CONTINUATION_RE = re.compile(r"[,;:!?—’”]+$")
ALPHA_RE = re.compile(r"[A-Za-z]")
TOKEN_RE = re.compile(r"[a-z0-9]+")
NUMERIC_TOKEN_RE = re.compile(r"^(?:[ivxlcdm]+|\d+)$")
# The closing line of a story, alone on its line. Books that end this way
# often carry publisher advertising after it, which is not narration.
# Matched optionally decorated ("*** THE END ***", "THE END.").
ENDING_RE = re.compile(
    r"(?im)^[ \t]*(?:[*_~=\s]*)(?:the\s+end|fin)\b[^\n]*$",
)

# A numbered heading line: a single space then UPPERCASE roman numeral or
# digits, followed by end/period/blank or a title ("CHAPTER I. Y-o-u-u
# Tom...", "ACT IV", "Letter 1"). The strict shape rejects prose such as
# "contains 2,000...", "the 18th Brumaire" or index lines padded with
# spaces ("Ruskin       156. Charing").
NUMBERED_HEADING_RE = re.compile(
    r"^\s*([A-Za-z][A-Za-z'\u2019\-]*)\s([IVXLCDM]+|\d+)(?=$|\.|\s*$|\s+[A-Z0-9])",
)
# When the second numbered heading appears this close after the first, the
# first one belongs to a contents listing, not to the body.
CONTENTS_GAP_LINES = 15

MAX_HEADING_CHARS = 80
MAX_HEADING_BLOCK_LINES = 3
MIN_PROSE_CHARS = 40
# How far past a heading to look for the prose it introduces. A chapter may
# open with short dialogue ("Tom!" / "No answer.") before the first long
# line, so the first non-blank line alone is not proof of narration.
PROSE_SCAN_LINES = 15
MIN_REPEATS = 2
# Only the last few blocks of a heading run can be a real section opener.
MAX_RUN_BLOCKS = 8
MIN_LISTING_CHAIN = 8
LISTING_SHARE = 0.5
# A closing line this far into the text is the end of the story, not prose.
ENDING_SEARCH_SHARE = 0.9


def is_ornament(line: str) -> bool:
    """True for decorative rules and lines carrying too little text."""
    stripped = line.strip()
    return bool(DECORATION_RE.match(stripped)) or len(ALPHA_RE.findall(stripped)) < 3


def is_heading_line(line: str) -> bool:
    """True if a single line could be a heading rather than running prose."""
    stripped = line.strip()
    if not stripped or len(stripped) > MAX_HEADING_CHARS:
        return False
    if is_ornament(stripped):
        return False
    if stripped.endswith("."):
        # A period is only tolerated as numbering punctuation ("IV.", "12.").
        # Any other full stop means the line runs on as prose.
        if not TRAILING_NUMERAL_DOT_RE.search(stripped):
            return False
        stripped = stripped[:-1].rstrip()
    return not TRAILING_CONTINUATION_RE.search(stripped)


def is_heading_block(block: list[str]) -> bool:
    """True if a blank-line delimited block is short enough to be headings."""
    if not block or len(block) > MAX_HEADING_BLOCK_LINES:
        return False
    return all(is_heading_line(line) for line in block)


def is_prose(lines: list[str], start: int) -> bool:
    """True if real prose appears within the window at/after start.

    The check scans a bounded window rather than only the first non-blank
    line: dialogue-driven chapters (pg74 opens with "Tom!" / "No answer.")
    lead with short lines before any line long enough to be narration.
    """
    for line in lines[start:start + PROSE_SCAN_LINES]:
        if len(line.strip()) >= MIN_PROSE_CHARS:
            return True
    return False


def shape_key(line: str) -> str:
    """Normalised shape of a heading: numbers and numerals collapse to '#'."""
    tokens = TOKEN_RE.findall(line.lower())
    return " ".join("#" if NUMERIC_TOKEN_RE.match(t) else t for t in tokens)


def split_blocks(lines: list[str]) -> list[tuple[int, list[str]]]:
    """Group lines into blank-line delimited (start_index, block) pairs."""
    blocks: list[tuple[int, list[str]]] = []
    current: list[str] = []
    start = 0
    for index, line in enumerate(lines):
        if line.strip():
            if not current:
                start = index
            current.append(line)
        elif current:
            blocks.append((start, current))
            current = []
    if current:
        blocks.append((start, current))
    return blocks


def heading_blocks(lines: list[str]) -> tuple[list[tuple[int, list[str]]], list[str | None]]:
    """Classify every blank-line delimited block as heading or not."""
    blocks = split_blocks(lines)
    keys: list[str | None] = [
        shape_key(block[0]) if is_heading_block(block) else None
        for _, block in blocks
    ]
    return blocks, keys


def block_chains(qualifying: list[bool]) -> list[list[int]]:
    """Group block indices into maximal runs of consecutive heading blocks."""
    chains: list[list[int]] = []
    current: list[int] = []
    for index, qualifies in enumerate(qualifying):
        if qualifies:
            current.append(index)
            continue
        if current:
            chains.append(current)
            current = []
    if current:
        chains.append(current)
    return chains


def listing_prefix(key: str) -> str:
    """Shape prefix that groups contents entries of one numbered series.

    Contents entries carry chapter/act titles ("chapter # y o u u tom ...")
    while the body heading is bare ("chapter #"); collapsing numbered keys
    to their first two tokens lets both count as the same repeating shape.
    Non-numbered keys stay whole so prose-like chains are unaffected.
    """
    tokens = key.split()
    if len(tokens) >= 2 and tokens[1] == "#":
        return " ".join(tokens[:2])
    return key


def is_contents_listing(chain: list[int], keys: list[str | None]) -> bool:
    """True if a heading chain looks like a table of contents / illustration list.

    Such a chain is long and its entries share one repeating shape, e.g. every
    entry collapsing to the key "chapter #" -- with or without trailing
    chapter titles. Prose front matter is too short to form a chain, and a
    numbered footnote list mixes shapes, so neither is mistaken for a listing.
    """
    shapes = [keys[index] for index in chain if keys[index]]
    if not shapes:
        return False
    prefixes = [listing_prefix(key) for key in shapes]
    dominant = Counter(prefixes).most_common(1)[0][1]
    return dominant / len(prefixes) >= LISTING_SHARE


def find_contents_body_start(lines: list[str]) -> int | None:
    """Anchor the body via the contents listing (second-occurrence rule).

    Books print a contents listing before the body: the first numbered
    heading line belongs to it, and the body is the second occurrence of
    that same heading ("CHAPTER I" in the contents -> the next "CHAPTER I"
    is the body). A same-series heading within CONTENTS_GAP_LINES marks the
    listing; scenes following a body "ACT I" are a different series and do
    not count, so contents-less books fall through to the structural scan.
    """
    numbered: list[tuple[int, str, str]] = []
    for index, line in enumerate(lines):
        match = NUMBERED_HEADING_RE.match(line)
        if not match:
            continue
        numbered.append((index, match.group(1).lower(), match.group(2).lower()))
    if not numbered:
        return None

    first_index, first_word, first_numeral = numbered[0]
    in_listing = any(
        word == first_word and index - first_index <= CONTENTS_GAP_LINES
        for index, word, _ in numbered[1:]
    )
    if not in_listing:
        # First heading is the body's own ("ACT I" + "Scene I" + prose);
        # require narration to follow so a dangling heading is not anchored.
        return first_index if is_prose(lines, first_index + 1) else None
    for index, word, numeral in numbered[1:]:
        if word == first_word and numeral == first_numeral:
            return index
    return None


def find_body_start(lines: list[str]) -> int:
    """Index of the first line of the story body.

    The body begins at the second occurrence of the contents' first numbered
    heading when a contents listing is present; otherwise at the first
    structural heading whose shape recurs in the document -- the
    chapter/part/book skeleton. Nothing here inspects what a heading is
    spelled beyond its numbered shape, so it holds across editions.
    """
    anchored = find_contents_body_start(lines)
    if anchored is not None:
        return anchored
    blocks, keys = heading_blocks(lines)
    qualifying = [key is not None for key in keys]
    counts = Counter(key for key in keys if key)
    chains = block_chains(qualifying)
    owner = {index: chain for chain in chains for index in chain}

    def leads_to_prose(chain: list[int]) -> bool:
        """True if prose follows the last block of this heading run."""
        tail = chain[-1] + 1
        return tail < len(blocks) and is_prose(lines, blocks[tail][0])

    def run_of(index: int) -> list[int]:
        return owner.get(index, [index])

    def tail_of(chain: list[int]) -> list[int]:
        """Trim a run to its last few blocks.

        A table of contents and the headings that follow it form one
        unbroken run, so only its tail is a real section opener.
        """
        return chain[-MAX_RUN_BLOCKS:]

    # Front-matter listings end where their last chain ends.
    cutoff = -1
    for chain in chains:
        if len(chain) >= MIN_LISTING_CHAIN and is_contents_listing(chain, keys):
            cutoff = max(cutoff, chain[-1])

    def numbered(key: str | None) -> bool:
        return bool(key) and "#" in key.split()

    for index in range(max(cutoff, 0), len(blocks)):
        key = keys[index]
        if not numbered(key) or counts[key] < MIN_REPEATS:
            continue
        chain = tail_of(run_of(index))
        if not leads_to_prose(chain):
            continue
        return blocks[index][0]

    for chain in chains:
        run = tail_of(chain)
        if not leads_to_prose(run):
            continue
        for index in run:
            if keys[index] and counts[keys[index]] >= MIN_REPEATS:
                return blocks[index][0]

    return 0


def find_ending(lines: list[str], start: int) -> int | None:
    """Index of the story's closing line at or after start, or None.

    Only a line in the last few percent of the body counts, so that a
    chapter or sentence reading "the end of it" is not mistaken for the
    end of the book. (preaudio.py adds `start + len * share` instead of
    measuring the share of the body itself, which disables the search
    whenever the body starts past ~10% of the lines -- this version is
    identical on full-length books and also works on small texts.)
    """
    tail_start = start + int((len(lines) - start) * ENDING_SEARCH_SHARE)
    for index in range(len(lines) - 1, tail_start - 1, -1):
        if ENDING_RE.match(lines[index]):
            return index
    return None


def extract_story_body(text: str) -> str:
    """Cut everything before the story body and after the closing line."""
    lines = text.splitlines()
    start = find_body_start(lines)
    ending = find_ending(lines, start)
    stop = len(lines) if ending is None else ending + 1
    return "\n".join(lines[start:stop])


def remove_front_matter(text: str) -> str:
    """Remove the front matter (and anything after THE END) from a text.

    The body start is found structurally -- the first heading whose shape
    recurs through the document -- so title pages, dedications, contents
    listings and prefaces are dropped no matter how they are titled. The
    keep-ratio guard returns the input unchanged when the cut would discard
    nearly everything (and more than _MAX_FRONT_MATTER_CHARS), so
    structureless text (e.g. story-narrator slices) is never emptied by a
    misfired cut.
    """
    out = extract_story_body(text)
    kept = len(out)
    total = len(text)
    if kept >= total * _MIN_KEEP_RATIO or (total - kept) <= _MAX_FRONT_MATTER_CHARS:
        return out
    return text


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
    strips the Gutenberg license boilerplate, cuts to the story body using
    structural heading detection (front matter before the first recurring
    heading pattern, and publisher text after THE END, are dropped), joins
    lines wrapped at the ~70-char column limit, and cleans whitespace so
    only the narration text remains.
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
