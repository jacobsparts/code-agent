"""
CLI helpers for Code Agent.
"""

from .mixin import CLIMixin, SQLiteHistory, InputSession
from .sessions import select_session_ui
from .terminal import (
    RESET, BOLD, DIM, ITALIC, UNDERLINE, STRIKE,
    RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, GRAY,
    parse_markup, strip_ansi, get_terminal_width, render_markdown,
    highlight_python,
    Panel, Markdown, Console,
    DEFAULT_THEME,
)

__all__ = [
    "CLIMixin",
    "SQLiteHistory",
    "InputSession",
    "select_session_ui",
    "RESET", "BOLD", "DIM", "ITALIC", "UNDERLINE", "STRIKE",
    "RED", "GREEN", "YELLOW", "BLUE", "MAGENTA", "CYAN", "WHITE", "GRAY",
    "parse_markup", "strip_ansi", "get_terminal_width", "render_markdown",
    "highlight_python",
    "Panel", "Markdown", "Console",
    "DEFAULT_THEME",
]