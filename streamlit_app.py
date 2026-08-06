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
DOCUMENT_TYPES: list[str] = ["pdf", "docx", "pptx", "xlsx", "html"]

# Document parser backends. Docling is model-based (accurate on hard layouts,
# ~500 MB of weights, pulls in PyTorch); LiteParse is heuristic (one 11 MB
# wheel, no weights, far faster) but ships no chunker -- see chunk_text.
PARSER_DOCLING: str = "Docling"
PARSER_LITEPARSE: str = "LiteParse"
# LiteParse handles PDFs and images natively; Office formats need LibreOffice
# on PATH and HTML is unsupported, so the uploader narrows to what always works.
LITE_DOCUMENT_TYPES: list[str] = ["pdf", "png", "jpg", "jpeg", "tiff", "webp"]

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


# -- Document functions (optional docling dependency) -------------------------


def docling_available() -> bool:
    """Return True if the optional ``docling`` dependency is importable."""
    import importlib.util

    return importlib.util.find_spec("docling") is not None


def load_document(file_bytes: bytes, filename: str) -> Any:
    """Parse uploaded file bytes into a ``DoclingDocument``."""
    import io

    from docling.datamodel.base_models import DocumentStream
    from docling.document_converter import DocumentConverter

    source = DocumentStream(name=filename, stream=io.BytesIO(file_bytes))
    return DocumentConverter().convert(source).document


def chunk_document(
    doc: Any, tokenizer: Any, max_tokens: int = MAX_CHUNK_TOKENS
) -> list[str]:
    """Split a ``DoclingDocument`` into structure-aware text chunks."""
    from docling.chunking import HybridChunker
    from docling_core.transforms.chunker.tokenizer.huggingface import (
        HuggingFaceTokenizer,
    )

    # HybridChunker's token budget lives on the tokenizer; mlx-lm wraps the
    # real Hugging Face tokenizer, so unwrap it via ._tokenizer.
    dl_tokenizer = HuggingFaceTokenizer(
        tokenizer=tokenizer._tokenizer, max_tokens=max_tokens
    )
    chunker = HybridChunker(tokenizer=dl_tokenizer)
    return [chunker.contextualize(chunk=c) for c in chunker.chunk(doc)]


# -- LiteParse document path (optional lite dependency) -----------------------
# LiteParse parses to markdown but ships no chunker, so chunk_text below is the
# token-aware packer that HybridChunker provides on the docling path.

_BLANK_LINE_RE = re.compile(r"\n\s*\n")
# Sentence boundaries for Latin punctuation plus CJK full stops, since the app
# translates across all 67 languages.
_SENTENCE_RE = re.compile(r"(?<=[.!?。！？])\s+")


def liteparse_available() -> bool:
    """Return True if the optional ``liteparse`` dependency is importable."""
    import importlib.util

    return importlib.util.find_spec("liteparse") is not None


def load_document_markdown(file_bytes: bytes) -> str:
    """Parse uploaded file bytes into a single markdown string via LiteParse."""
    from liteparse import LiteParse

    # ocr_enabled=False: born-digital PDFs carry a real text layer, and OCR on
    # top of it is slower without changing the result. quiet=True keeps
    # LiteParse's timing lines out of Streamlit's stdout.
    parser = LiteParse(output_format="markdown", ocr_enabled=False, quiet=True)
    return parser.parse(file_bytes).text


def heading_level(block: str) -> int:
    """Return the ATX markdown heading level of ``block``, or 0 if not a heading."""
    marker = block.lstrip().split(" ", 1)[0]
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


def split_oversized(block: str, tokenizer: Any, max_tokens: int) -> list[str]:
    """Split a single over-budget block into pieces that each fit ``max_tokens``."""
    pieces: list[str] = []
    overhead = token_overhead(tokenizer)
    for sentence in _SENTENCE_RE.split(block):
        if not sentence:
            continue
        if count_tokens(sentence, tokenizer) <= max_tokens:
            pieces.append(sentence)
            continue
        # A single sentence over budget (unpunctuated wall of text, or a table
        # dumped as one block): fall back to hard token windows. Drop the BOS
        # prefix first, or it would be decoded back into the chunk as text.
        ids = tokenizer.encode(sentence)[overhead:]
        pieces.extend(
            tokenizer.decode(ids[i : i + max_tokens])
            for i in range(0, len(ids), max_tokens)
        )
    return pieces


def chunk_text(
    text: str, tokenizer: Any, max_tokens: int = MAX_CHUNK_TOKENS
) -> list[str]:
    """Pack markdown paragraphs into chunks that stay under ``max_tokens``.

    The token-aware counterpart to ``chunk_document`` for the LiteParse path.
    Blocks are packed greedily so each chunk carries as much as it can hold,
    paragraphs are never split unless one alone exceeds the budget, and the
    enclosing markdown headings are prepended to chunks that do not already
    open with them -- the same context that ``HybridChunker.contextualize``
    supplies, and which matters here because every chunk is translated as an
    independent prompt with no memory of its neighbours.
    """
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
        # Skip any heading the chunk already opens with, so context is not
        # duplicated at the top of the chunk.
        context = [pending[lv] for lv in sorted(pending) if pending[lv] != current[0]]
        body = "\n\n".join([*context, *current])
        # Packing sums per-block counts, but a tokenizer merges across block
        # boundaries, so the assembled chunk can run a little over. Measure the
        # real thing and split it if the estimate drifted past the budget.
        if count_tokens(body, tokenizer) > max_tokens:
            chunks.extend(split_oversized(body, tokenizer, max_tokens))
        else:
            chunks.append(body)
        current = []
        current_tokens = 0
        pending = {}
        pending_tokens = 0

    for block in blocks:
        # Reserve room for whichever context is larger: the one already
        # committed to the open chunk, or the one a chunk opening now would
        # take. Either is possible depending on where this block lands.
        budget = max(max_tokens - max(trail_tokens, pending_tokens), 1)
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

        if level := heading_level(block):
            # A deeper heading replaces its siblings; shallower ones survive.
            trail = {lv: h for lv, h in trail.items() if lv < level}
            trail[level] = block
            # Each context heading is followed by a separator when rendered.
            trail_tokens = sum(
                count_tokens(h, tokenizer) + separator for h in trail.values()
            )

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
    filename: str,
    backend: str = PARSER_DOCLING,
    max_tokens: int = MAX_CHUNK_TOKENS,
) -> list[str]:
    """Parse + chunk an uploaded document, cached by file bytes, name and backend.

    Re-translating the same upload (e.g. into another language) then skips the
    expensive parse + chunk. ``backend`` is part of the cache key so switching
    parsers re-parses instead of returning the other backend's chunks. The
    tokenizer is fetched from the cached ``load_model()`` rather than taken as an
    argument, since it is unhashable and would defeat ``@st.cache_data``'s
    argument hashing.
    """
    _, tokenizer = load_model()
    if backend == PARSER_LITEPARSE:
        return chunk_text(load_document_markdown(file_bytes), tokenizer, max_tokens)
    doc = load_document(file_bytes, filename)
    return chunk_document(doc, tokenizer, max_tokens)


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
    parsers = [
        name
        for name, installed in (
            (PARSER_DOCLING, docling_available()),
            (PARSER_LITEPARSE, liteparse_available()),
        )
        if installed
    ]
    if not parsers:
        st.info(
            "Document translation needs a parser: `uv sync --extra docs` for "
            "Docling, or `uv sync --extra lite` for LiteParse.",
            icon=":material/download:",
        )
    else:
        # -- Parser backend ---------------------------------------------------

        # Seeded here rather than with the other setdefault() calls because the
        # valid options depend on which extras are installed; this also resets a
        # stored choice whose backend has since been uninstalled.
        if st.session_state.get("doc_parser") not in parsers:
            st.session_state.doc_parser = parsers[0]
        st.radio(
            "Parser",
            parsers,
            key="doc_parser",
            horizontal=True,
            help=(
                "Docling: model-based, better on dense tables and scans. "
                "LiteParse: heuristic, no model weights, much faster to start."
            ),
        )
        parser = st.session_state.doc_parser

        # -- Language bar -----------------------------------------------------

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

        # -- Upload + controls ------------------------------------------------

        uploaded = st.file_uploader(
            "Upload a document",
            type=(
                LITE_DOCUMENT_TYPES if parser == PARSER_LITEPARSE else DOCUMENT_TYPES
            ),
            label_visibility="collapsed",
        )
        translate_doc_clicked = st.button(
            "Translate document",
            key="translate_doc",
            disabled=not (model_loaded and uploaded is not None),
            type="primary",
            width="stretch",
        )

        # -- Warning slot + streamed output -----------------------------------

        doc_warning_slot = st.container()
        doc_output_placeholder = st.empty()
        if st.session_state.doc_output:
            render_output(doc_output_placeholder, st.session_state.doc_output)

        # -- Process document translation -------------------------------------

        if translate_doc_clicked and uploaded is not None:
            if st.session_state.doc_source_lang == st.session_state.doc_target_lang:
                doc_warning_slot.warning(SAME_LANGUAGE_WARNING)
            else:
                result = ""
                try:
                    with st.spinner(f"Reading document with {parser}..."):
                        chunks = cached_document_chunks(
                            uploaded.getvalue(), uploaded.name, parser
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
                            status.write(
                                f"Translating section {idx + 1} of {len(chunks)}"
                            )
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

        # -- Download ---------------------------------------------------------

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
