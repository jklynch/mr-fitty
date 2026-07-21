#!/usr/bin/env python3
"""Regenerate the table of contents in a Markdown file between TOC markers.

The TOC is built from the level-2 (``## ``) headings and written between a pair of
marker comments::

    <!-- toc -->
    ... generated list, do not edit by hand ...
    <!-- /toc -->

Files without the markers are left untouched. Anchors follow GitHub's slug rules so the
links resolve both on GitHub and in most Markdown viewers.

Used as a pre-commit hook (see .pre-commit-config.yaml): it rewrites the file in place
and exits non-zero when it changed anything, so the commit stops and the refreshed file
can be re-staged -- the same behavior as black and the other formatters here.

Run directly on one or more files:

    python3 scripts/gen_toc.py dev-journal.md
"""

import re
import sys

TOC_OPEN = "<!-- toc -->"
TOC_CLOSE = "<!-- /toc -->"

# The heading that introduces the TOC itself, excluded so it never lists itself.
SELF_HEADING = "contents"


def slugify(heading_text):
    """GitHub-style anchor slug for a heading.

    Lowercase, drop every character that is not a word character, hyphen or space
    (so backticks, colons, commas and em/en dashes all vanish), then turn spaces into
    hyphens. A ' - ' between words therefore collapses to '--', matching GitHub.
    """
    slug = heading_text.strip().lower()
    slug = re.sub(r"[^\w\- ]", "", slug)
    return slug.replace(" ", "-")


def collect_headings(lines):
    """Level-2 headings, as (display_text, anchor), skipping the TOC's own heading.

    Fenced code blocks are skipped so a '## ' inside a ``` block is not mistaken for a
    heading. Duplicate slugs get GitHub's -1, -2, ... suffixes.
    """
    headings = []
    seen_slugs = {}
    in_fence = False
    for line in lines:
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = re.match(r"^##[ \t]+(.*\S)\s*$", line)
        if not match:
            continue
        text = match.group(1)
        if text.strip().lower() == SELF_HEADING:
            continue
        slug = slugify(text)
        count = seen_slugs.get(slug, 0)
        seen_slugs[slug] = count + 1
        if count:
            slug = f"{slug}-{count}"
        headings.append((text, slug))
    return headings


def render_toc(headings):
    return "\n".join(f"- [{text}](#{anchor})" for text, anchor in headings)


def update_file(path):
    """Rewrite path's TOC in place. Returns True if the file changed."""
    with open(path, encoding="utf-8") as handle:
        original = handle.read()

    if TOC_OPEN not in original:
        print(f"{path}: no {TOC_OPEN} marker, skipping", file=sys.stderr)
        return False
    if TOC_CLOSE not in original:
        raise SystemExit(f"{path}: found {TOC_OPEN} but no matching {TOC_CLOSE}")

    headings = collect_headings(original.splitlines())
    toc_block = f"{TOC_OPEN}\n{render_toc(headings)}\n{TOC_CLOSE}"

    updated = re.sub(
        re.escape(TOC_OPEN) + r".*?" + re.escape(TOC_CLOSE),
        lambda _: toc_block,  # avoid re backreference interpretation in the replacement
        original,
        count=1,
        flags=re.DOTALL,
    )

    if updated == original:
        return False
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(updated)
    return True


def main(argv):
    paths = argv or ["dev-journal.md"]
    changed = [path for path in paths if update_file(path)]
    for path in changed:
        print(f"{path}: table of contents updated")
    return 1 if changed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
