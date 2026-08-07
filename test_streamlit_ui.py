# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock, patch

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest


@pytest.fixture(autouse=True)
def clear_st_cache() -> None:
    """Clear Streamlit's @st.cache_resource between tests."""
    st.cache_resource.clear()


@pytest.fixture
def app() -> AppTest:
    """Create an AppTest instance that has completed its initial render.

    The ``mlx_lm.load`` patch is belt-and-braces: the model loads lazily from
    the translate handlers, so an initial render never reaches it (see
    ``test_page_renders_without_loading_the_model``). Any test that goes on to
    click Translate must re-patch -- ``_rerun_with_mocks`` or
    ``_run_inference_test`` -- or the real loader runs and pulls 3.6 GB.
    """
    with patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)
    return at


def _rerun_with_mocks(app: AppTest) -> MagicMock:
    """Re-run the app with mocked model loading, returning the loader mock.

    Callers that expect a rerun to short-circuit before the model is needed can
    assert on the returned mock.
    """
    with patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())) as load:
        app.run(timeout=60)
    return load


def _make_stream_chunk(text: str) -> MagicMock:
    chunk = MagicMock()
    chunk.text = text
    return chunk


def _run_inference_test(input_text: str, chunk_text: str) -> AppTest:
    """Build a fresh AppTest, enter text, click Translate, and return it."""
    with (
        patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())),
        patch(
            "mlx_lm.stream_generate",
            return_value=iter([_make_stream_chunk(chunk_text)]),
        ),
    ):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)
        at.text_area[0].set_value(input_text)
        at.button("translate").click()
        at.run(timeout=60)
    return at


# -- Title ---------------------------------------------------------------------


def test_title_is_app_name(app: AppTest) -> None:
    assert str(app.title[0].value) == "Tiny Aya Translate"


# -- Tabs ----------------------------------------------------------------------


def test_tabs_labelled_text_and_document_with_icons(app: AppTest) -> None:
    assert [t.label for t in app.tabs] == [
        ":material/text_fields: Text",
        ":material/description: Document",
    ]


# -- Language defaults ---------------------------------------------------------


def test_source_language_default(app: AppTest) -> None:
    assert app.selectbox[0].value == "English"


def test_target_language_default(app: AppTest) -> None:
    assert app.selectbox[1].value == "French"


# -- Swap button ---------------------------------------------------------------


def test_swap_button_exists(app: AppTest) -> None:
    assert app.button("swap") is not None


def test_swap_flips_languages(app: AppTest) -> None:
    app.button("swap").click()
    _rerun_with_mocks(app)

    assert app.selectbox[0].value == "French"
    assert app.selectbox[1].value == "English"


def test_swap_moves_output_to_input() -> None:
    """After translating, swap should move the output into the input field."""
    with (
        patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())),
        patch(
            "mlx_lm.stream_generate",
            return_value=iter([_make_stream_chunk("Bonjour")]),
        ),
    ):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)

        # Translate "Hello" -> "Bonjour"
        at.text_area[0].set_value("Hello")
        at.button("translate").click()
        at.run(timeout=60)

        # Swap
        at.button("swap").click()
        at.run(timeout=60)

    # Input should now contain the previous output
    assert at.text_area[0].value == "Bonjour"
    # Output should be cleared
    assert at.text_area[1].value == ""


# -- Text panels ---------------------------------------------------------------


def test_input_text_area_has_no_placeholder(app: AppTest) -> None:
    assert app.text_area[0].placeholder == ""


def test_output_uses_text_area(app: AppTest) -> None:
    assert len(app.text_area) == 2


def test_output_text_area_placeholder(app: AppTest) -> None:
    assert app.text_area[1].placeholder == "Translation"


# -- Translate flow ------------------------------------------------------------


def test_translate_button_exists(app: AppTest) -> None:
    assert app.button("translate") is not None


def test_translate_button_enabled_when_model_loaded(app: AppTest) -> None:
    assert not app.button("translate").disabled


def test_translate_success_shows_result() -> None:
    at = _run_inference_test(input_text="Hello", chunk_text="Bonjour")
    assert at.text_area[1].value == "Bonjour"


def test_translate_empty_text_shows_warning(app: AppTest) -> None:
    app.button("translate").click()
    load = _rerun_with_mocks(app)

    warning_values = [w.value for w in app.warning]
    assert any("Please enter some text first" in str(v) for v in warning_values)
    # The check is free, so it must short-circuit before the weights load.
    load.assert_not_called()


def test_translate_same_language_shows_warning(app: AppTest) -> None:
    app.selectbox[1].set_value("English")
    app.text_area[0].set_value("Hello")
    app.button("translate").click()
    load = _rerun_with_mocks(app)

    warning_values = [w.value for w in app.warning]
    assert any("two different languages" in str(v) for v in warning_values)
    # Likewise free -- rejecting the pair must not cost a 3.6 GB load.
    load.assert_not_called()


# -- Language switching --------------------------------------------------------


def test_change_source_language(app: AppTest) -> None:
    app.selectbox[0].set_value("Spanish")
    _rerun_with_mocks(app)

    assert app.selectbox[0].value == "Spanish"


def test_change_target_language(app: AppTest) -> None:
    app.selectbox[1].set_value("Spanish")
    _rerun_with_mocks(app)

    assert app.selectbox[1].value == "Spanish"


# -- Input constraints ---------------------------------------------------------


def test_input_max_chars_enforced(app: AppTest) -> None:
    app.text_area[0].set_value("x" * 30001)
    _rerun_with_mocks(app)

    value = app.text_area[0].value
    assert value is not None
    assert len(value) <= 30000


def test_translate_too_many_tokens_shows_warning() -> None:
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = list(range(8193))

    with patch("mlx_lm.load", return_value=(MagicMock(), mock_tokenizer)):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)
        at.text_area[0].set_value("Hello world")
        at.button("translate").click()
        at.run(timeout=60)

    warning_values = [str(w.value) for w in at.warning]
    assert any("8193" in v and "8192" in v for v in warning_values)


def test_translate_at_input_token_limit_succeeds() -> None:
    """Input at exactly MAX_INPUT_TOKENS should translate without warning."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = list(range(8192))

    with (
        patch("mlx_lm.load", return_value=(MagicMock(), mock_tokenizer)),
        patch(
            "mlx_lm.stream_generate",
            return_value=iter([_make_stream_chunk("OK")]),
        ),
    ):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)
        at.text_area[0].set_value("Hello world")
        at.button("translate").click()
        at.run(timeout=60)

    assert at.text_area[1].value == "OK"
    assert not at.warning


def test_translation_error_shows_message() -> None:
    with (
        patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())),
        patch("mlx_lm.stream_generate", side_effect=RuntimeError("OOM")),
    ):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)
        at.text_area[0].set_value("Hello")
        at.button("translate").click()
        at.run(timeout=60)

    error_values = [str(e.value) for e in at.error]
    assert any("Translation failed" in v and "OOM" in v for v in error_values)


def test_tokenizer_failure_shows_message_not_traceback() -> None:
    # apply_chat_template raises on a model whose chat template rejects this
    # message shape -- reachable by editing MODEL_ID as the README invites.
    # tokenize_prompt sits inside the translate try/except so this gets the same
    # warning-slot treatment as a streaming failure, not a red traceback.
    bad_tokenizer = MagicMock()
    bad_tokenizer.apply_chat_template.side_effect = RuntimeError("no chat template")

    with patch("mlx_lm.load", return_value=(MagicMock(), bad_tokenizer)):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)
        at.text_area[0].set_value("Hello")
        at.button("translate").click()
        at.run(timeout=60)

    assert not at.exception
    error_values = [str(e.value) for e in at.error]
    assert any(
        "Translation failed" in v and "no chat template" in v for v in error_values
    )


def test_empty_stream_shows_warning() -> None:
    with (
        patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())),
        patch("mlx_lm.stream_generate", return_value=iter([])),
    ):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)
        at.text_area[0].set_value("Hello")
        at.button("translate").click()
        at.run(timeout=60)

    warning_values = [str(w.value) for w in at.warning]
    assert any("no output" in v for v in warning_values)


def test_end_response_only_stream_shows_warning() -> None:
    with (
        patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())),
        patch(
            "mlx_lm.stream_generate",
            return_value=iter([_make_stream_chunk("<|END_RESPONSE|>")]),
        ),
    ):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)
        at.text_area[0].set_value("Hello")
        at.button("translate").click()
        at.run(timeout=60)

    warning_values = [str(w.value) for w in at.warning]
    assert any("no output" in v for v in warning_values)


# -- Download button -----------------------------------------------------------


def test_download_button_exists(app: AppTest) -> None:
    # One in the Text tab, one in the Document tab.
    assert len(app.get("download_button")) == 2


def test_download_button_label(app: AppTest) -> None:
    assert app.get("download_button")[0].label == "Download"  # ty: ignore[unresolved-attribute]


def test_download_button_disabled_when_output_empty(app: AppTest) -> None:
    assert app.get("download_button")[0].disabled  # ty: ignore[unresolved-attribute]


def test_download_button_enabled_when_output_present() -> None:
    at = _run_inference_test(input_text="Hello", chunk_text="Bonjour")
    assert not at.get("download_button")[0].disabled  # ty: ignore[unresolved-attribute]


# -- Output text area ----------------------------------------------------------


def test_output_text_area_disabled(app: AppTest) -> None:
    assert app.text_area[1].disabled


# -- Lazy model loading --------------------------------------------------------


def test_page_renders_without_loading_the_model() -> None:
    """The UI paints before the weights are touched.

    The model is loaded from the translate handlers, not at page load, so a
    broken install still gets a full page -- tabs, language pickers, and a
    usable input -- with no error until the user actually asks to translate.
    """
    with patch("mlx_lm.load", side_effect=RuntimeError("download failed")) as load:
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)

    load.assert_not_called()
    assert len(at.tabs) == 2
    assert len(at.text_area) == 2
    assert not at.error


def test_model_load_failure_shows_error_on_translate() -> None:
    with patch("mlx_lm.load", side_effect=RuntimeError("download failed")):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)
        at.text_area[0].set_value("Hello")
        at.button("translate").click()
        at.run(timeout=60)

    error_values = [e.value for e in at.error]
    assert any("Failed to load model" in str(v) for v in error_values)


def test_translate_button_stays_enabled_after_a_failed_load() -> None:
    # Click first, so the load actually fails before the assertion -- asserting
    # on a fresh render would only re-test the initial state, since nothing
    # loads the model there.
    with patch("mlx_lm.load", side_effect=RuntimeError("download failed")):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)
        at.text_area[0].set_value("Hello")
        at.button("translate").click()
        at.run(timeout=60)

    # A failed load must not disable the button: @st.cache_resource does not
    # memoize the exception, so a retry works without reloading the page.
    assert not at.button("translate").disabled


# -- Document tab --------------------------------------------------------------


def test_document_translate_button_exists(app: AppTest) -> None:
    assert app.button("translate_doc") is not None


def test_document_translate_button_disabled_without_upload(app: AppTest) -> None:
    assert app.button("translate_doc").disabled


def test_document_download_button_disabled_when_no_output(app: AppTest) -> None:
    # The second download button belongs to the Document tab.
    assert app.get("download_button")[1].disabled  # ty: ignore[unresolved-attribute]


def test_document_tab_needs_no_install_hint(app: AppTest) -> None:
    # liteparse is a required dependency now, so the tab is never gated behind
    # an extra and there is nothing for the user to install.
    assert not any("uv sync" in str(i.value) for i in app.info)


def test_document_uploader_offers_no_parser_choice(app: AppTest) -> None:
    # One parser, so no backend radio -- the Text tab's swap button and the two
    # language selectboxes per tab are the only widgets of their kind.
    assert not [r for r in app.radio]
