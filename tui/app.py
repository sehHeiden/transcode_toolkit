"""
TranscodeApp is the main entry point for the TUI application, responsible for setting up
layouts and managing user interactions.
"""

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Placeholder
from textual.containers import Horizontal, Vertical


class TranscodeApp(App):
    """Main application class for the transcode toolkit TUI."""

    TITLE = "Transcode Toolkit"
    BINDINGS = [
        ("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        """Compose the UI layout."""
        yield Header()
        with Horizontal():
            yield Placeholder("Directory/File Selection", id="file_selector")
            with Vertical():
                yield Placeholder("Processing Monitor", id="monitor")
                yield Placeholder("Status Dashboard", id="status")
            yield Placeholder("Preset Selection", id="preset_selector")
        yield Footer()
