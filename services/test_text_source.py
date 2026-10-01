"""Tests for structural front-matter removal and text curation.

The structural detection (ported from preaudio.py) must start the body at
the first recurring heading pattern regardless of section names, cut
publisher text after THE END, and never empty structureless text.
"""

from services.text_source import curate_text, extract_story_body, remove_front_matter

_PROSE = (
    "The road was long and the night was dark, but they walked on bravely together."
)


def _book_with_front_matter() -> str:
    """A small book shaped like pg345: title page/dedication, no CONTENTS."""
    return (
        "DRACULA\n\n"
        "by\n\n"
        "Bram Stoker\n\n"
        "NEW YORK\n\n"
        "GROSSET & DUNLAP\n\n"
        "TO\nMY DEAR FRIEND\nHOMMY-BEG\n\n"
        "CHAPTER I\n\n"
        "JONATHAN HARKER'S JOURNAL\n\n"
        f"{_PROSE}\n\n"
        "CHAPTER II\n\n"
        "THE SECOND CHAPTER TITLE\n\n"
        f"{_PROSE}\n\n"
        "CHAPTER III\n\n"
        "THE THIRD CHAPTER TITLE\n\n"
        f"{_PROSE}\n"
    )


def test_remove_front_matter_starts_body_at_first_recurring_chapter():
    # Arrange: title page and dedication, no named sections (pg345 shape)
    text = _book_with_front_matter()

    # Act
    out = remove_front_matter(text)

    # Assert: narration starts at the first chapter; front matter gone
    assert out.startswith("CHAPTER I")
    assert "GROSSET" not in out
    assert "HOMMY-BEG" not in out
    assert "JONATHAN HARKER'S JOURNAL" in out


def test_remove_front_matter_cuts_content_after_the_end():
    # Arrange: story plus a trailing publisher ad after THE END
    text = _book_with_front_matter() + "\nTHE END\n\nPUBLISHERS OF THIS EDITION WANT YOU TO BUY MORE BOOKS\n"

    # Act
    out = remove_front_matter(text)

    # Assert: closing line kept, advertising dropped
    assert out.rstrip().endswith("THE END")
    assert "WANT YOU TO BUY MORE BOOKS" not in out


def test_remove_front_matter_keeps_structureless_text():
    # Arrange: plain paragraphs, no headings at all
    text = "\n\n".join([_PROSE] * 5) + "\n"

    # Act
    out = remove_front_matter(text)

    # Assert: nothing was cut (modulo the dropped trailing newline)
    assert out.rstrip("\n") == text.rstrip("\n")


def test_remove_front_matter_falls_back_when_cut_would_emptify_text():
    # Arrange: >20k chars of headingless front matter, body only in the
    # last few percent -- the structural cut would discard >95% of the book
    front = "\n\n".join(
        [f"Paragraph {i} of introductory material that carries no heading structure at all." for i in range(300)]
    )
    text = f"{front}\n\nCHAPTER I\n\n{_PROSE}\n\nCHAPTER II\n\n{_PROSE}\n"
    assert len(front) > 20_000

    # Act
    out = remove_front_matter(text)

    # Assert: guard fired, input returned unchanged
    assert out == text


def test_extract_story_body_matches_pre_audio_pipeline():
    # Arrange/Act: the ported pipeline equals preaudio's contract
    text = _book_with_front_matter()
    out = extract_story_body(text)

    # Assert: extraction equals the slice from CHAPTER I (modulo newline)
    assert out.startswith("CHAPTER I")
    assert out.rstrip("\n") == text[text.index("CHAPTER I"):].rstrip("\n")


def test_curate_text_strips_boilerplate_and_sanitizes_underscores():
    # Arrange: license wrapper, underscored speech, a Gutenberg-wrapped line
    raw = (
        "*** START OF THE PROJECT GUTENBERG EBOOK DEMO ***\n"
        "_Hello world_, he said to his friend across the narrow hall and they\n"
        "laughed together about the old times.\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK DEMO ***\n"
    )

    # Act
    out = curate_text(raw)

    # Assert: markers gone, speech quoted, wrapped line joined
    assert "PROJECT GUTENBERG" not in out
    assert '"Hello world"' in out
    assert "narrow hall and they laughed together about the old times." in out


def test_remove_front_matter_starts_body_at_dialogue_opening_chapter():
    # Arrange: pg74 (Tom Sawyer) shape -- Chapter I opens with short
    # dialogue ("Tom!", "No answer.") before the first long prose line
    text = (
        "THE ADVENTURES OF TOM SAWYER\n\n"
        "by\n\n"
        "Mark Twain\n\n"
        "CHAPTER I\n\n"
        '"Tom!"\n\n'
        "No answer.\n\n"
        '"TOM!"\n\n'
        f"{_PROSE}\n\n"
        "CHAPTER II\n\n"
        f"{_PROSE}\n"
    )

    # Act
    out = remove_front_matter(text)

    # Assert: body starts at the dialogue-opening first chapter, not the second
    assert out.splitlines()[0] == "CHAPTER I"
    assert '"Tom!"' in out


def test_remove_front_matter_skips_contents_listing_to_body_act():
    # Arrange: pg1524 (Hamlet) shape -- a contents listing whose entries mix
    # bare act numbers with scene titles, followed by a cast list containing
    # long (>40 char) name lines, then the body starting at ACT I
    contents = "\n\n".join(
        [
            "ACT I\n\n Scene I. A platform before the Castle\n\n Scene II. Another room in the Castle",
            "ACT II\n\n Scene I. A room in the Castle\n\n Scene II. Another room",
            "ACT III\n\n Scene I. A room in state\n\n Scene II. Another room",
            "ACT IV\n\n Scene I. A castle hall\n\n Scene II. Another room",
            "ACT V\n\n Scene I. A churchyard\n\n Scene II. A hall in the Castle",
        ]
    )
    cast = "\n\n".join(
        [
            "Dramatis Personae",
            "HAMLET, Prince of Denmark",
            "CLAUDIUS, King of Denmark, Hamlet's uncle",
            "GERTRUDE, the Queen, Hamlet's mother, now wife of Claudius",
            "POLONIUS, Lord Chamberlain",
            "OPHELIA, Daughter to Polonius",
        ]
    )
    body = (
        "ACT I\n\n"
        "SCENE I. Elsinore. A platform before the Castle.\n\n"
        f"{_PROSE}\n\n"
        "ACT II\n\n"
        f"{_PROSE}\n"
    )
    text = f"{contents}\n\n{cast}\n\n{body}"

    # Act
    out = remove_front_matter(text)

    # Assert: the contents act headings are skipped; body starts at ACT I
    assert out.startswith("ACT I\n\nSCENE I. Elsinore")


def test_remove_front_matter_body_act_followed_by_scene_starts_at_act():
    # Arrange: contents-less play whose body ACT I is followed by Scene I --
    # a different heading series, so it must not be read as a contents entry
    text = (
        "THE TRAGEDY OF HAMLET\n\n"
        "ACT I\n\n"
        "Scene I. A platform before the Castle.\n\n"
        f"{_PROSE}\n\n"
        "ACT II\n\n"
        f"{_PROSE}\n"
    )

    # Act
    out = remove_front_matter(text)

    # Assert: body starts at the first act, not past it
    assert out.startswith("ACT I\n\nScene I. A platform")
