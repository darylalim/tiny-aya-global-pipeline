# SPDX-License-Identifier: Apache-2.0
import os
import re
from collections.abc import Iterator
from typing import Any

# Mute transformers alias-warning spam triggered by Streamlit's module watcher.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# -- Config ------------------------------------------------------------------

MODEL_ID: str = "mlx-community/tiny-aya-global-8bit-mlx"
DEFAULT_TEMPERATURE: float = 0.1
DEFAULT_MAX_TOKENS: int = 8192
MAX_INPUT_TOKENS: int = 8192

# Chunk budget for document translation: well below MAX_INPUT_TOKENS to leave
# room for the per-chunk prompt wrapper (instruction + chat template).
MAX_CHUNK_TOKENS: int = 7000
# LiteParse handles PDFs and images natively. Office formats additionally need
# LibreOffice on PATH, so they are left out rather than failing at parse time.
DOCUMENT_TYPES: list[str] = ["pdf", "png", "jpg", "jpeg", "tiff", "webp"]

# Shared UI constants reused across the Text and Document tabs.
PANEL_HEIGHT: int = 450
SAME_LANGUAGE_WARNING: str = "Please pick two different languages."
NO_OUTPUT_WARNING: str = "Model produced no output."

# -- Languages ---------------------------------------------------------------
# 67 languages across Europe, West Asia, South Asia, Asia Pacific, and Africa.

LANGUAGES: list[str] = [
    # Europe (31)
    "English",
    "Dutch",
    "French",
    "Italian",
    "Portuguese",
    "Romanian",
    "Spanish",
    "Czech",
    "Polish",
    "Ukrainian",
    "Russian",
    "Greek",
    "German",
    "Danish",
    "Swedish",
    "Bokmål",
    "Catalan",
    "Galician",
    "Welsh",
    "Irish",
    "Basque",
    "Croatian",
    "Latvian",
    "Lithuanian",
    "Slovak",
    "Slovenian",
    "Estonian",
    "Finnish",
    "Hungarian",
    "Serbian",
    "Bulgarian",
    # West Asia (5)
    "Arabic",
    "Persian",
    "Turkish",
    "Maltese",
    "Hebrew",
    # South Asia (9)
    "Hindi",
    "Marathi",
    "Bengali",
    "Gujarati",
    "Punjabi",
    "Tamil",
    "Telugu",
    "Nepali",
    "Urdu",
    # Asia Pacific (12)
    "Tagalog",
    "Malay",
    "Indonesian",
    "Vietnamese",
    "Javanese",
    "Khmer",
    "Thai",
    "Lao",
    "Chinese",
    "Burmese",
    "Japanese",
    "Korean",
    # African (10)
    "Amharic",
    "Hausa",
    "Igbo",
    "Malagasy",
    "Shona",
    "Swahili",
    "Wolof",
    "Xhosa",
    "Yoruba",
    "Zulu",
]


# Tesseract language codes, keyed by the LANGUAGES entry the user picks as the
# document's source. OCR defaults to English, so a Cyrillic, CJK or Arabic scan
# came back as plausible-looking Latin garbage with words silently dropped --
# non-empty, so it sailed past the "no translatable text" guard.
#
# Only languages Tesseract publishes traineddata for are listed. Hausa, Igbo,
# Malagasy, Shona, Wolof, Xhosa and Zulu have none, so they fall back to
# English rather than asking for a file that does not exist. Each language's
# data is a separate ~12-15 MB download on first use.
OCR_LANGUAGES: dict[str, str] = {
    "English": "eng",
    "Dutch": "nld",
    "French": "fra",
    "Italian": "ita",
    "Portuguese": "por",
    "Romanian": "ron",
    "Spanish": "spa",
    "Czech": "ces",
    "Polish": "pol",
    "Ukrainian": "ukr",
    "Russian": "rus",
    "Greek": "ell",
    "German": "deu",
    "Danish": "dan",
    "Swedish": "swe",
    "Bokmål": "nor",
    "Catalan": "cat",
    "Galician": "glg",
    "Welsh": "cym",
    "Irish": "gle",
    "Basque": "eus",
    "Croatian": "hrv",
    "Latvian": "lav",
    "Lithuanian": "lit",
    "Slovak": "slk",
    "Slovenian": "slv",
    "Estonian": "est",
    "Finnish": "fin",
    "Hungarian": "hun",
    "Serbian": "srp",
    "Bulgarian": "bul",
    "Arabic": "ara",
    "Persian": "fas",
    "Turkish": "tur",
    "Maltese": "mlt",
    "Hebrew": "heb",
    "Hindi": "hin",
    "Marathi": "mar",
    "Bengali": "ben",
    "Gujarati": "guj",
    "Punjabi": "pan",
    "Tamil": "tam",
    "Telugu": "tel",
    "Nepali": "nep",
    "Urdu": "urd",
    "Tagalog": "fil",
    "Malay": "msa",
    "Indonesian": "ind",
    "Vietnamese": "vie",
    "Javanese": "jav",
    "Khmer": "khm",
    "Thai": "tha",
    "Lao": "lao",
    "Chinese": "chi_sim",
    "Burmese": "mya",
    "Japanese": "jpn",
    "Korean": "kor",
    "Amharic": "amh",
    "Yoruba": "yor",
    "Swahili": "swa",
}


# -- Pure functions -----------------------------------------------------------


def build_translation_prompt(
    text: str, source_lang: str, target_lang: str
) -> list[dict[str, str]]:
    """Build the chat messages list for a translation request."""
    return [
        {
            "role": "user",
            "content": (
                f"Translate the following text from {source_lang} to {target_lang}. "
                f"Output only the translation, nothing else.\n\n{text}"
            ),
        }
    ]


def tokenize_prompt(
    text: str, source_lang: str, target_lang: str, tokenizer: Any
) -> list[int]:
    """Apply the chat template and return the prompt token ids."""
    messages = build_translation_prompt(text, source_lang, target_lang)
    return tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )


def clean_model_output(decoded_text: str) -> str:
    """Strip the ``<|END_RESPONSE|>`` end-of-turn marker and surrounding whitespace."""
    return decoded_text.replace("<|END_RESPONSE|>", "").strip()


def stream_translate(
    prompt_ids: list[int],
    model: Any,
    tokenizer: Any,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Iterator[str]:
    """Stream cleaned translation chunks from a pre-tokenized prompt."""
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=temperature)
    accumulated = ""
    for response in stream_generate(
        model,
        tokenizer,
        prompt=prompt_ids,
        max_tokens=max_tokens,
        sampler=sampler,
    ):
        accumulated += response.text
        yield clean_model_output(accumulated)


# -- Document functions -------------------------------------------------------
# LiteParse parses to markdown but ships no chunker, so chunk_text below is the
# token-aware packer that splits a document into translatable pieces.

_BLANK_LINE_RE = re.compile(r"\n\s*\n")
# Sentence boundaries for Latin punctuation plus CJK full stops, since the app
# translates across all 67 languages. The two need different whitespace rules:
# Latin prose separates sentences with a space (and requiring one is what keeps
# "3.14" intact), while CJK writes 。！？ flush against the next sentence, so
# demanding whitespace there means CJK never splits on a sentence at all.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|(?<=[。！？])\s*")
# Fence markers, stripped when deciding whether a parse found any real text.
_FENCE_RE = re.compile(r"```[a-zA-Z]*")
# Most tokens a single character can cost under byte fallback: UTF-8 runs to
# four bytes, and a byte-fallback tokenizer spends one token per byte.
_MAX_CHAR_TOKENS = 4


def is_blank_markdown(text: str) -> bool:
    """True when a parse produced no readable content.

    A file LiteParse could not read comes back as an empty ```` ```text``` ````
    fence. That is not blank, so it would sail past the Document tab's "no
    translatable text" guard and be handed to the model as a prompt.
    """
    return not _FENCE_RE.sub("", text).strip()


def load_document_markdown(file_bytes: bytes, source_lang: str = "English") -> str:
    """Parse uploaded file bytes into a single markdown string via LiteParse.

    ``source_lang`` selects the OCR language for image uploads; it is ignored
    for PDFs, which never run OCR. Returns ``""`` when the file yielded no
    readable text.
    """
    from liteparse import LiteParse

    # OCR is enabled only where it is the sole extraction path. An image has no
    # text layer, so reading it *requires* OCR -- forcing it off returned an
    # empty '```text```' fence for every image type in DOCUMENT_TYPES. A PDF
    # does carry a text layer, and LiteParse's auto mode would still reach for
    # OCR on sparse pages, which downloads ~15 MB of Tesseract training data
    # from GitHub on first use. This app's whole premise is that nothing leaves
    # the machine, so PDFs -- the common case -- stay strictly offline.
    # ocr_failure_fatal=False so an OCR attempt that cannot fetch its data
    # never kills a parse whose text layer was readable all along.
    # quiet=True keeps LiteParse's timing lines out of Streamlit's stdout.
    parser = LiteParse(
        output_format="markdown",
        quiet=True,
        ocr_enabled=not file_bytes.lstrip()[:4].startswith(b"%PDF"),
        ocr_language=OCR_LANGUAGES.get(source_lang, "eng"),
        ocr_failure_fatal=False,
    )
    text = parser.parse(file_bytes).text
    return "" if is_blank_markdown(text) else text


def heading_line(block: str) -> str:
    """Return ``block``'s first line.

    Blocks are split on blank lines, so a markdown heading followed immediately
    by its opening body line arrives as one block. The heading trail keeps this
    line alone: storing the whole block would prepend that body text to every
    later chunk and charge its tokens against the budget the trail reserves.
    """
    return block.lstrip().split("\n", 1)[0].rstrip()


def heading_level(block: str) -> int:
    """Return the ATX markdown heading level of ``block``, or 0 if not a heading."""
    marker = heading_line(block).split(" ", 1)[0]
    return len(marker) if marker and set(marker) == {"#"} and len(marker) <= 6 else 0


def token_overhead(tokenizer: Any) -> int:
    """Count the special tokens ``encode`` prepends to every string.

    The model's tokenizer adds a BOS token to each ``encode`` call, so summing
    per-block counts would charge one BOS per block for a chunk that only ever
    carries one. Measuring the constant lets the packer work in net tokens.
    """
    return len(tokenizer.encode(""))


def count_tokens(text: str, tokenizer: Any) -> int:
    """Return the token length of ``text``, excluding the special-token prefix."""
    return len(tokenizer.encode(text)) - token_overhead(tokenizer)


def sentence_units(text: str) -> list[str]:
    """Split ``text`` after sentence enders, keeping each unit's trailing gap.

    Units carry the whitespace that followed them, so concatenating them back
    reproduces the source exactly -- CJK included, where the gap is empty and
    rejoining on a space would insert one the original never had.
    """
    units: list[str] = []
    start = 0
    for match in _SENTENCE_RE.finditer(text):
        if match.end() > start:
            units.append(text[start : match.end()])
            start = match.end()
    if start < len(text):
        units.append(text[start:])
    return units


def hard_windows(text: str, tokenizer: Any, max_tokens: int) -> list[str]:
    """Cut ``text`` into token windows -- the last resort when nothing else fits.

    Strips the BOS prefix before slicing, or ``decode`` would write the special
    token back into the chunk as visible text.

    Boundaries are pulled back until the window decodes cleanly. A byte-fallback
    tokenizer spends several tokens on one character in scripts outside its
    vocabulary -- Thai, Lao, Khmer, Burmese, Amharic and rarer CJK, all of them
    in LANGUAGES -- so a fixed stride can cut a character in half. The halves
    decode to U+FFFD and the character is lost outright, and the corrupted text
    is what would then be sent to the model as source.
    """
    # A window of zero tokens would never advance past its own start.
    max_tokens = max(max_tokens, 1)
    ids = tokenizer.encode(text)[token_overhead(tokenizer) :]
    windows: list[str] = []
    start = 0
    while start < len(ids):
        end = min(start + max_tokens, len(ids))
        piece = tokenizer.decode(ids[start:end])
        # Give back at most a character's worth of tokens: past that the
        # replacement character is in the source itself, not an artefact of
        # the cut, and shrinking further would only strand tokens.
        floor = max(start + 1, end - _MAX_CHAR_TOKENS)
        while end > floor and "�" in piece:
            end -= 1
            piece = tokenizer.decode(ids[start:end])
        windows.append(piece)
        start = end
    return windows


def pack_by_estimate(
    units: list[str], tokenizer: Any, budget: int, join_cost: int = 0
) -> list[list[str]]:
    """Group ``units`` greedily using one token count per unit.

    Measuring every growing candidate instead is quadratic: on a single
    chunk-sized block it re-encodes the whole accumulating piece per unit and
    costs tens of seconds. Callers verify each finished group once, which keeps
    the guarantee while staying linear in the document length.

    ``join_cost`` is charged for every join after the first. Callers that glue
    units together with a separator must pass it: a thousand blocks joined by a
    blank line cost a thousand tokens the per-unit counts cannot see, and
    ignoring that overshot the budget by a quarter.
    """
    groups: list[list[str]] = []
    current: list[str] = []
    running = 0
    for unit in units:
        size = count_tokens(unit, tokenizer)
        if current and running + size + join_cost > budget:
            groups.append(current)
            current, running = [], 0
        elif current:
            size += join_cost
        current.append(unit)
        running += size
    if current:
        groups.append(current)
    return groups


def split_oversized(block: str, tokenizer: Any, max_tokens: int) -> list[str]:
    """Split a single over-budget block into pieces that each fit ``max_tokens``.

    Sentences are re-packed up to the budget rather than emitted one per piece.
    Without that, a chunk overflowing by a single token exploded into one piece
    per sentence -- hundreds of separate translation calls, each stripped of the
    neighbours that gave it context.
    """
    units: list[str] = []
    for unit in sentence_units(block):
        if not unit.strip():
            continue
        if count_tokens(unit, tokenizer) <= max_tokens:
            units.append(unit)
            continue
        # A single sentence over budget: an unpunctuated wall of text, or a
        # script whose sentence enders this pattern does not know.
        windows = hard_windows(unit, tokenizer, max_tokens)
        # Windows are cut mid-text, so they carry no trailing gap of their own.
        units.extend(w + " " for w in windows[:-1])
        units.extend(windows[-1:])

    pieces: list[str] = []
    for group in pack_by_estimate(units, tokenizer, max_tokens):
        text = "".join(group).strip()
        if not text:
            continue
        measured = count_tokens(text, tokenizer)
        if measured <= max_tokens:
            pieces.append(text)
            continue
        # The estimate drifted (a tokenizer merges across unit boundaries).
        # Re-pack this group alone against a budget cut by the overshoot.
        for sub in pack_by_estimate(
            group, tokenizer, max(2 * max_tokens - measured, 1)
        ):
            sub_text = "".join(sub).strip()
            if count_tokens(sub_text, tokenizer) <= max_tokens:
                pieces.append(sub_text)
            else:
                pieces.extend(
                    w.strip()
                    for w in hard_windows(sub_text, tokenizer, max_tokens)
                    if w.strip()
                )
    return pieces


def leading_headings(blocks: list[str]) -> list[str]:
    """Return the headings ``blocks`` opens with, before its first body block.

    Only these make prepended context redundant. A heading further in is the
    *next* section starting inside the chunk, so the body ahead of it still
    needs its own section's heading prepended.
    """
    found: list[str] = []
    for block in blocks:
        if not heading_level(block):
            break
        found.append(heading_line(block))
    return found


def repack_blocks(
    headings: list[str], blocks: list[str], tokenizer: Any, max_tokens: int
) -> list[str]:
    """Re-pack ``blocks`` into chunks that each carry their own heading context.

    The greedy packer estimates a chunk by summing its blocks, but a BPE
    tokenizer merges across block boundaries, so the finished chunk can measure
    a token or two over. Re-packing whole blocks keeps the blank lines between
    them -- splitting the assembled body on sentences instead would dissolve
    every paragraph, table and list in it.

    ``headings`` is the full trail, not the caller's already-deduplicated
    context: a piece cut from the middle of a chunk contains none of the
    headings the first piece opened with, so it has to have them prepended or
    it reaches the model as an unlabelled fragment.
    """

    # A chunk that opens with its own headings carries them as blocks, not in
    # the trail -- the trail was still empty when it opened. Pieces cut from
    # its middle contain neither, so those headings have to join the context or
    # every piece after the first reaches the model unlabelled.
    context = [*headings, *(h for h in leading_headings(blocks) if h not in headings)]

    def assemble(body: list[str]) -> str:
        own = leading_headings(body)
        return "\n\n".join([*[h for h in context if h not in own], *body])

    separator = count_tokens("\n\n", tokenizer)
    overhead = sum(count_tokens(h, tokenizer) + separator for h in context)
    # Each join costs the blank line plus about a token of merge drift the
    # per-block counts cannot see.
    join_cost = separator + 1
    budget = max(max_tokens - overhead, 1)

    packed: list[str] = []
    for group in pack_by_estimate(blocks, tokenizer, budget, join_cost):
        text = assemble(group)
        measured = count_tokens(text, tokenizer)
        if measured <= max_tokens:
            packed.append(text)
            continue
        # The estimate still drifted over. Re-pack this group alone against a
        # budget cut by the overshoot, assembling each sub-group so it keeps
        # its heading context -- falling straight through to sentence
        # splitting here is what stripped the context off middle pieces.
        tighter = max(budget - (measured - max_tokens), 1)
        for sub in pack_by_estimate(group, tokenizer, tighter, join_cost):
            sub_text = assemble(sub)
            if count_tokens(sub_text, tokenizer) <= max_tokens:
                packed.append(sub_text)
            else:
                packed.extend(split_oversized(sub_text, tokenizer, max_tokens))
    return packed


def chunk_text(
    text: str, tokenizer: Any, max_tokens: int = MAX_CHUNK_TOKENS
) -> list[str]:
    """Pack markdown paragraphs into chunks that stay under ``max_tokens``.

    LiteParse returns one markdown string and no chunker, so this is where a
    document becomes translatable pieces. Blocks are packed greedily so each
    chunk carries as much as it can hold, paragraphs are never split unless one
    alone exceeds the budget, and the enclosing markdown headings are prepended
    to chunks that do not already open with them -- context that matters because
    every chunk is translated as an independent prompt with no memory of its
    neighbours.
    """
    # A non-positive budget would divide the document by zero-width windows.
    max_tokens = max(max_tokens, 1)
    blocks = [b.strip() for b in _BLANK_LINE_RE.split(text) if b.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    # Blocks are joined with a blank line, so every join after the first costs
    # a separator. Charging it keeps the running total honest instead of
    # drifting over budget by one token per block.
    separator = count_tokens("\n\n", tokenizer)
    # ``trail`` advances as blocks are read; ``pending`` is the snapshot taken
    # when the open chunk started. They differ once a heading is read into a
    # chunk that is still filling, and prepending ``trail`` at that point would
    # label a chunk with the heading of the section that comes *after* it.
    trail: dict[int, str] = {}
    trail_tokens = 0
    pending: dict[int, str] = {}
    pending_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens, pending, pending_tokens
        if not current:
            return
        # Skip only the headings the chunk *opens* with -- those alone make the
        # prepended context redundant. A heading deeper in the chunk starts the
        # next section, and the body ahead of it still needs its own label.
        headings = [pending[lv] for lv in sorted(pending)]
        opening = leading_headings(current)
        context = [h for h in headings if h not in opening]
        body = "\n\n".join([*context, *current])
        # Packing sums per-block counts, but a tokenizer merges across block
        # boundaries, so the assembled chunk can run a little over. Re-pack the
        # blocks against measured sizes rather than splitting the body, which
        # would shatter a full chunk into one prompt per sentence.
        if count_tokens(body, tokenizer) > max_tokens:
            chunks.extend(repack_blocks(headings, current, tokenizer, max_tokens))
        else:
            chunks.append(body)
        current = []
        current_tokens = 0
        pending = {}
        pending_tokens = 0

    in_fence = False
    for block in blocks:
        # Reserve room for whichever context is larger: the one already
        # committed to the open chunk, or the one a chunk opening now would
        # take. Either is possible depending on where this block lands.
        room = max_tokens - max(trail_tokens, pending_tokens)
        if room < max_tokens // 4:
            # The heading trail is eating the budget. Clamping to a token
            # instead shreds the document: a trail of 7,002 tokens against a
            # 7,000 budget turned 13,121 tokens of markdown into 12,082 chunks
            # and 292 seconds of packing. A chunk carrying less context still
            # translates; a chunk of one token does not.
            trail, trail_tokens = {}, 0
            pending, pending_tokens = {}, 0
            room = max_tokens
        budget = max(room, 1)
        n = count_tokens(block, tokenizer)
        pieces = [block] if n <= budget else split_oversized(block, tokenizer, budget)
        for piece in pieces:
            size = n if piece is block else count_tokens(piece, tokenizer)
            if current:
                size += separator
            if current and current_tokens + size > budget:
                flush()
                size -= separator
            if not current:
                pending, pending_tokens = dict(trail), trail_tokens
            current.append(piece)
            current_tokens += size

        # A '#' line inside a fenced code block is a shell comment, not a
        # heading. Blocks split on blank lines, so a fence containing one
        # arrives as several blocks and the marker count has to be carried
        # across them -- otherwise the comment evicts the real heading and
        # every chunk below it is labelled with a line of shell.
        if not in_fence and (level := heading_level(block)):
            # A deeper heading replaces its siblings; shallower ones survive.
            trail = {lv: h for lv, h in trail.items() if lv < level}
            trail[level] = heading_line(block)
            # Each context heading is followed by a separator when rendered.
            trail_tokens = sum(
                count_tokens(h, tokenizer) + separator for h in trail.values()
            )
        in_fence ^= block.count("```") % 2 == 1

    flush()
    return chunks


def translate_document(
    chunks: list[str],
    source_lang: str,
    target_lang: str,
    model: Any,
    tokenizer: Any,
) -> Iterator[tuple[int, str]]:
    """Translate each chunk; yield ``(index, cumulative_text)`` per token."""
    done: list[str] = []
    for i, chunk in enumerate(chunks):
        prompt_ids = tokenize_prompt(chunk, source_lang, target_lang, tokenizer)
        if len(prompt_ids) > MAX_INPUT_TOKENS:
            # Skip rather than abort the whole document on one oversized chunk.
            done.append("[Section skipped: too long to translate.]")
            yield i, "\n\n".join(p for p in done if p)
            continue
        partial = ""
        for partial in stream_translate(prompt_ids, model, tokenizer):
            yield i, "\n\n".join(p for p in [*done, partial] if p)
        done.append(partial)


import streamlit as st  # noqa: E402

st.set_page_config(
    page_title="Tiny Aya Translate",
    page_icon=":material/translate:",
    layout="wide",
)


@st.cache_resource
def load_model() -> tuple[Any, Any]:
    """Load model and tokenizer once, cached for the session lifetime."""
    from mlx_lm import load

    loaded = load(MODEL_ID)
    return loaded[0], loaded[1]


@st.cache_data(max_entries=8, show_spinner=False)
def cached_document_chunks(
    file_bytes: bytes,
    source_lang: str = "English",
    max_tokens: int = MAX_CHUNK_TOKENS,
) -> list[str]:
    """Parse + chunk an uploaded document, cached by file bytes and source
    language.

    Re-translating the same upload into another *target* language then skips
    the parse + chunk. The source language is part of the key because it picks
    the OCR language, so it changes the parsed text. LiteParse sniffs the
    format itself, so the filename never reaches the parser. The tokenizer is
    fetched from the cached ``load_model()`` rather than taken as an argument,
    since it is unhashable and would defeat ``@st.cache_data``'s argument
    hashing.
    """
    _, tokenizer = load_model()
    markdown = load_document_markdown(file_bytes, source_lang)
    return chunk_text(markdown, tokenizer, max_tokens)


def render_output(placeholder: Any, text: str) -> None:
    """Render streamed translation output into ``placeholder``.

    Uses ``st.code`` rather than ``st.text_area`` so the placeholder can be
    replaced mid-script without colliding with a widget's auto-generated
    element id.
    """
    placeholder.code(text, language=None, wrap_lines=True, height=PANEL_HEIGHT)


# -- Main page ----------------------------------------------------------------

st.title("Tiny Aya Translate")


# -- Model loading ------------------------------------------------------------

try:
    with st.spinner("Loading model..."):
        model, tokenizer = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Failed to load model: {e}")
    model, tokenizer = None, None
    model_loaded = False

# -- Session state defaults ---------------------------------------------------

st.session_state.setdefault("source_lang", "English")
st.session_state.setdefault("target_lang", "French")
st.session_state.setdefault("translate_input", "")
st.session_state.setdefault("translate_output", "")
st.session_state.setdefault("_do_translate", False)
st.session_state.setdefault("doc_source_lang", "English")
st.session_state.setdefault("doc_target_lang", "French")
st.session_state.setdefault("doc_output", "")


def request_translate() -> None:
    """Flag that a translation was requested (processed after controls row)."""
    st.session_state._do_translate = True


def swap_languages() -> None:
    """Swap source/target languages and move output into input."""
    st.session_state.source_lang, st.session_state.target_lang = (
        st.session_state.target_lang,
        st.session_state.source_lang,
    )
    st.session_state.translate_input = st.session_state.translate_output
    st.session_state.translate_output = ""


text_tab, doc_tab = st.tabs(
    [":material/text_fields: Text", ":material/description: Document"]
)

with text_tab:
    # -- Language bar ---------------------------------------------------------

    with st.container(border=True):
        col_from, col_swap, col_to = st.columns(
            [10, 1, 10], vertical_alignment="center"
        )
        with col_from:
            st.selectbox(
                "From",
                LANGUAGES,
                key="source_lang",
                label_visibility="collapsed",
            )
        with col_swap:
            st.button(
                "",
                key="swap",
                icon=":material/swap_horiz:",
                on_click=swap_languages,
                width="stretch",
                type="tertiary",
                help="Swap languages",
            )
        with col_to:
            st.selectbox(
                "To",
                LANGUAGES,
                key="target_lang",
                label_visibility="collapsed",
            )

    # -- Warning slot (above panels) ------------------------------------------

    warning_slot = st.container()

    # -- Side-by-side text panels ---------------------------------------------

    col_input, col_output = st.columns(2)
    with col_input:
        st.text_area(
            "Input",
            height=PANEL_HEIGHT,
            max_chars=30000,
            key="translate_input",
            label_visibility="collapsed",
        )
    with col_output:
        output_placeholder = st.empty()
        output_placeholder.text_area(
            "Output",
            height=PANEL_HEIGHT,
            placeholder="Translation",
            disabled=True,
            value=st.session_state.translate_output,
            label_visibility="collapsed",
        )

    # -- Controls row ---------------------------------------------------------

    sub_translate, sub_download = st.columns(
        2, vertical_alignment="center", gap="small"
    )
    with sub_translate:
        st.button(
            "Translate",
            key="translate",
            on_click=request_translate,
            disabled=not model_loaded,
            type="primary",
            width="stretch",
        )
    with sub_download:
        st.download_button(
            "Download",
            key="download",
            data=st.session_state.translate_output,
            file_name="translation.txt",
            mime="text/plain",
            disabled=not st.session_state.translate_output.strip(),
            type="secondary",
            width="stretch",
        )

    # -- Process translation request (below controls) -------------------------

    if st.session_state._do_translate:
        st.session_state._do_translate = False
        current_input = st.session_state.translate_input
        if not current_input.strip():
            warning_slot.warning("Please enter some text first.")
        elif st.session_state.source_lang == st.session_state.target_lang:
            warning_slot.warning(SAME_LANGUAGE_WARNING)
        elif (
            n_tok := len(
                prompt_ids := tokenize_prompt(
                    current_input,
                    st.session_state.source_lang,
                    st.session_state.target_lang,
                    tokenizer,
                )
            )
        ) > MAX_INPUT_TOKENS:
            warning_slot.warning(
                f"Input is {n_tok} tokens — please keep it under {MAX_INPUT_TOKENS}."
            )
        else:
            partial = ""
            try:
                with st.spinner("Translating..."):
                    for partial in stream_translate(prompt_ids, model, tokenizer):
                        render_output(output_placeholder, partial)
            except Exception as e:
                warning_slot.error(f"Translation failed: {e}")
            else:
                if not partial.strip():
                    warning_slot.warning(NO_OUTPUT_WARNING)
                else:
                    st.session_state.translate_output = partial
                    # Rerun so the disabled output picks up the final value.
                    st.rerun()

with doc_tab:
    # -- Language bar ---------------------------------------------------------

    with st.container(border=True):
        doc_col_from, doc_col_to = st.columns(2)
        with doc_col_from:
            st.selectbox(
                "From",
                LANGUAGES,
                key="doc_source_lang",
                label_visibility="collapsed",
            )
        with doc_col_to:
            st.selectbox(
                "To",
                LANGUAGES,
                key="doc_target_lang",
                label_visibility="collapsed",
            )

    # -- Upload + controls ----------------------------------------------------

    uploaded = st.file_uploader(
        "Upload a document",
        type=DOCUMENT_TYPES,
        label_visibility="collapsed",
    )
    translate_doc_clicked = st.button(
        "Translate document",
        key="translate_doc",
        disabled=not (model_loaded and uploaded is not None),
        type="primary",
        width="stretch",
    )

    # -- Warning slot + streamed output ---------------------------------------

    doc_warning_slot = st.container()
    doc_output_placeholder = st.empty()
    if st.session_state.doc_output:
        render_output(doc_output_placeholder, st.session_state.doc_output)

    # -- Process document translation -----------------------------------------

    if translate_doc_clicked and uploaded is not None:
        if st.session_state.doc_source_lang == st.session_state.doc_target_lang:
            doc_warning_slot.warning(SAME_LANGUAGE_WARNING)
        else:
            result = ""
            try:
                with st.spinner("Reading document..."):
                    chunks = cached_document_chunks(
                        uploaded.getvalue(), st.session_state.doc_source_lang
                    )
                if not chunks:
                    doc_warning_slot.warning(
                        "No translatable text found in the document."
                    )
                else:
                    progress = st.progress(0.0)
                    status = st.empty()
                    last_rendered = -1
                    for idx, cumulative in translate_document(
                        chunks,
                        st.session_state.doc_source_lang,
                        st.session_state.doc_target_lang,
                        model,
                        tokenizer,
                    ):
                        result = cumulative
                        progress.progress(idx / len(chunks))
                        status.write(f"Translating section {idx + 1} of {len(chunks)}")
                        # Re-render only on chunk boundaries; re-sending the
                        # whole growing document every token is O(n²).
                        if idx != last_rendered:
                            render_output(doc_output_placeholder, result)
                            last_rendered = idx
                    progress.progress(1.0)
                    status.empty()
                    render_output(doc_output_placeholder, result)
                    if result.strip():
                        st.session_state.doc_output = result
                    else:
                        doc_warning_slot.warning(NO_OUTPUT_WARNING)
            except Exception as e:
                if result.strip():
                    st.session_state.doc_output = result
                    doc_warning_slot.error(
                        f"Translation failed after partial output: {e}"
                    )
                else:
                    doc_warning_slot.error(f"Translation failed: {e}")

    # -- Download -------------------------------------------------------------

    st.download_button(
        "Download",
        key="download_doc",
        data=st.session_state.doc_output,
        file_name="translation.md",
        mime="text/markdown",
        disabled=not st.session_state.doc_output.strip(),
        type="secondary",
        width="stretch",
    )
