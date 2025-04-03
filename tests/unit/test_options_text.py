import textwrap
from pathlib import Path
from typing import Generator

import pytest

from mcio_remote import util

##
# OptionsTxt


@pytest.fixture
def test_path(tmp_path: Path) -> Generator[Path, None, None]:
    yield tmp_path


@pytest.fixture
def test_options() -> str:
    return textwrap.dedent(
        """
        # Test comment
        foo:bar
        test:value
    """
    )


def test_options_txt_basic(test_path: Path, test_options: str) -> None:
    # Create a test file
    test_file: Path = test_path / "options.txt"
    test_file.write_text(test_options)
    # Test basic loading
    with util.OptionsTxt(test_file) as opts:
        assert opts["foo"] == "bar"


def test_options_txt_save(test_path: Path, test_options: str) -> None:
    test_file: Path = test_path / "options.txt"
    test_file.write_text(test_options)

    # Test save_on_exit=True
    with util.OptionsTxt(test_file, save=True) as opts:
        opts["new"] = "value"
    with util.OptionsTxt(test_file) as opts:
        assert opts["new"] == "value"


def test_options_txt_separator(test_path: Path) -> None:
    test_file: Path = test_path / "server.properties"
    test_file.write_text("key=value\n")

    with util.OptionsTxt(test_file, separator="=") as opts:
        assert opts["key"] == "value"


def test_options_txt_empty_file(test_path: Path) -> None:
    test_file: Path = test_path / "empty.txt"
    test_file.touch()

    with util.OptionsTxt(test_file) as opts:
        assert opts.options == {}


def test_options_txt_missing_file(test_path: Path) -> None:
    test_file: Path = test_path / "nonexistent.txt"

    with util.OptionsTxt(test_file) as opts:
        assert opts.options == {}
        opts["foo"] = "bar"


def test_options_txt_clear(test_path: Path) -> None:
    test_file: Path = test_path / "options.txt"
    test_file.write_text("foo:bar\n")

    with util.OptionsTxt(test_file) as opts:
        opts.clear()
        assert opts.options == {}


def test_options_txt_manual_save(test_path: Path) -> None:
    test_file: Path = test_path / "options.txt"

    opts: util.OptionsTxt = util.OptionsTxt(test_file)
    opts.load()
    opts["foo"] = "bar"
    opts.save()

    with util.OptionsTxt(test_file) as new_opts:
        assert new_opts["foo"] == "bar"


def test_options_txt_exception_handling(test_path: Path) -> None:
    test_file: Path = test_path / "options.txt"
    test_file.write_text("foo:bar\n")

    try:
        with util.OptionsTxt(test_file, save=True) as opts:
            opts["new"] = "value"
            raise ValueError("Test exception")
    except ValueError:
        pass

    # Verify it didn't save due to the exception
    with util.OptionsTxt(test_file) as opts:
        assert opts.options is not None
        assert "new" not in opts.options
