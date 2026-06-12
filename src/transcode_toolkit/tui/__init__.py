from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.reactive import reactive
from textual.widgets import Button, Footer, Header, Input, Label, ProgressBar, Select, Static

from transcode_toolkit.config import ToolkitConfig
from transcode_toolkit.types import ProcessingResult, ProcessingStatus


class TranscodeApp(App):
    CSS = """
    Screen {
        layout: vertical;
    }
    #main {
        layout: horizontal;
        height: 1fr;
    }
    #sidebar {
        width: 32;
        dock: left;
        border-right: solid green;
        padding: 1;
    }
    #output {
        padding: 1;
        height: 1fr;
        overflow-y: auto;
    }
    .label {
        margin-bottom: 1;
    }
    Button {
        margin-top: 1;
    }
    ProgressBar {
        margin-top: 1;
    }
    #status_bar {
        dock: bottom;
        height: 3;
        background: $boost;
        padding: 0 1;
    }
    """

    TITLE = "Transcode Toolkit"
    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("q", "quit", "Quit"),
        Binding("r", "run_transcode", "Run"),
    ]

    mode: reactive[str] = reactive("audio")
    path_text: reactive[str] = reactive("")
    preset: reactive[str] = reactive("music")
    status_message: reactive[str] = reactive("Ready")

    def __init__(self, config_path: Path | None = None) -> None:
        super().__init__()
        self._config_path = config_path or Path("config.yaml")
        self._config: ToolkitConfig | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main"):
            with VerticalScroll(id="sidebar"):
                yield Label("Mode:", classes="label")
                yield Select(
                    [
                        ("Audio", "audio"),
                        ("Video", "video"),
                        ("Audio Estimate", "audio_estimate"),
                        ("Video Estimate", "video_estimate"),
                    ],
                    value="audio",
                    id="mode_select",
                )
                yield Label("Directory:", classes="label")
                yield Input(placeholder="/path/to/media", id="path_input")
                yield Label("Preset:", classes="label")
                yield Select(
                    [
                        ("music", "music"),
                        ("audiobook", "audiobook"),
                        ("audiobook_stereo", "audiobook_stereo"),
                        ("high", "high"),
                        ("low", "low"),
                    ],
                    value="music",
                    id="preset_select",
                )
                yield Button("Run", variant="primary", id="run_btn")
                yield Button("Estimate", variant="default", id="estimate_btn")
                yield ProgressBar(id="progress")
            with VerticalScroll(id="output"):
                yield Static("Select a directory and press Run.", id="output_text")
        yield Static(self.status_message, id="status_bar")
        yield Footer()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "mode_select":
            self.mode = str(event.value) if event.value != Select.BLANK else "audio"
        elif event.select.id == "preset_select":
            self.preset = str(event.value) if event.value != Select.BLANK else "music"

    def on_input_changed(self, event: Input.Changed) -> None:
        self.path_text = event.value

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "run_btn":
            self.action_run_transcode()
        elif event.button.id == "estimate_btn":
            self.action_run_estimate()

    def _load_config(self) -> ToolkitConfig:
        if self._config is None:
            self._config = ToolkitConfig.from_yaml(self._config_path)
        return self._config

    def _update_output(self, text: str) -> None:
        output = self.query_one("#output_text", Static)
        output.update(text)

    def _set_status(self, msg: str) -> None:
        self.status_message = msg
        bar = self.query_one("#status_bar", Static)
        bar.update(msg)

    def action_run_transcode(self) -> None:
        path = Path(self.path_text)
        if not path.exists():
            self._update_output(f"[red]Path not found: {path}[/red]")
            return

        if self.mode in ("audio", "audio_estimate"):
            self._run_audio(path)
        else:
            self._run_video(path)

    def action_run_estimate(self) -> None:
        path = Path(self.path_text)
        if not path.exists():
            self._update_output(f"[red]Path not found: {path}[/red]")
            return

        if self.mode in ("audio", "audio_estimate"):
            self._run_audio_estimate(path)
        else:
            self._run_video_estimate(path)

    def _run_audio(self, path: Path) -> None:
        from transcode_toolkit.audio import transcode_audio_directory

        config = self._load_config()
        self._set_status("Transcoding audio...")
        self._update_output("[yellow]Starting audio transcode...[/yellow]")

        results = transcode_audio_directory(path, self.preset, config)
        lines = _format_results(results)
        self._update_output("\n".join(lines))

        success = sum(1 for r in results if r.status == ProcessingStatus.SUCCESS)
        skipped = sum(1 for r in results if r.status == ProcessingStatus.SKIPPED)
        errors = sum(1 for r in results if r.status == ProcessingStatus.ERROR)
        self._set_status(f"Done: {success} ok, {skipped} skipped, {errors} errors")

    def _run_audio_estimate(self, path: Path) -> None:
        from transcode_toolkit.audio import estimate_audio

        config = self._load_config()
        preset = config.audio.presets[self.preset]
        files = [f for f in path.rglob("*") if f.suffix.lower() in config.audio.extensions and f.is_file()]

        self._set_status(f"Estimating {len(files)} files...")
        lines = [f"{'File':40s} {'Src MB':>10s} {'Est MB':>10s} {'Saved':>10s}"]
        lines.append("-" * 72)
        total_src = 0
        total_est = 0
        for f in files:
            try:
                r = estimate_audio(f, preset)
                total_src += r.original_size
                total_est += r.new_size or 0
                saved = (1 - (r.new_size or 0) / r.original_size) * 100 if r.original_size else 0
                lines.append(
                    f"{f.name:40s} {r.original_size / 1e6:10.1f} {(r.new_size or 0) / 1e6:10.1f} {saved:9.1f}%"
                )
            except Exception as e:
                lines.append(f"{f.name:40s} ERROR: {e}")

        pct = (1 - total_est / total_src) * 100 if total_src else 0
        lines.append(f"\nTotal: {total_src / 1e6:.1f} MB -> {total_est / 1e6:.1f} MB ({pct:.1f}% saved)")
        self._update_output("\n".join(lines))
        self._set_status("Estimate complete")

    def _run_video(self, path: Path) -> None:
        from transcode_toolkit.video import transcode_video_directory

        config = self._load_config()
        self._set_status("Transcoding video...")
        self._update_output("[yellow]Starting video transcode...[/yellow]")

        results = transcode_video_directory(path, codec="libx265", crf=28, speed="fast", config=config)
        lines = _format_results(results)
        self._update_output("\n".join(lines))

        success = sum(1 for r in results if r.status == ProcessingStatus.SUCCESS)
        skipped = sum(1 for r in results if r.status == ProcessingStatus.SKIPPED)
        errors = sum(1 for r in results if r.status == ProcessingStatus.ERROR)
        self._set_status(f"Done: {success} ok, {skipped} skipped, {errors} errors")

    def _run_video_estimate(self, path: Path) -> None:
        from transcode_toolkit.video import estimate_video

        config = self._load_config()
        files = [f for f in path.rglob("*") if f.suffix.lower() in config.video.extensions and f.is_file()]
        self._set_status(f"Estimating {len(files)} video files...")

        lines: list[str] = []
        for f in files:
            if f.stat().st_size <= 1024 * 1024:
                continue
            try:
                results = estimate_video(f, config)
                lines.append(f"\n{f.name} ({f.stat().st_size / 1e6:.1f} MB):")
                lines.append(f"  {'Preset':25s} {'VMAF':>6s} {'Ratio':>6s}")
                lines.extend(f"  {r['label']:25s} {r['vmaf']:6.1f} {r['size_ratio']:6.2f}" for r in results)
            except Exception as e:
                lines.append(f"\n{f.name}: ERROR: {e}")

        self._update_output("\n".join(lines) if lines else "No video files found.")
        self._set_status("Video estimate complete")


def _format_results(results: list[ProcessingResult]) -> list[str]:
    lines: list[str] = []
    by_status: dict[ProcessingStatus, list[ProcessingResult]] = {}
    for r in results:
        by_status.setdefault(r.status, []).append(r)

    for status in ProcessingStatus:
        items = by_status.get(status, [])
        if not items:
            continue
        lines.append(f"\n[bold]{status.value.upper()} ({len(items)}):[/bold]")
        for r in items:
            if r.status == ProcessingStatus.SUCCESS and r.new_size is not None:
                saved = (1 - r.new_size / r.original_size) * 100 if r.original_size else 0
                lines.append(
                    f"  {r.source.name}: {r.original_size / 1e6:.1f} -> {r.new_size / 1e6:.1f} MB ({saved:.0f}% saved)"
                )
            else:
                lines.append(f"  {r.source.name}: {r.original_size / 1e6:.1f} MB")

    total_saved = sum(
        r.original_size - (r.new_size or r.original_size) for r in results if r.status == ProcessingStatus.SUCCESS
    )
    if total_saved:
        lines.append(f"\nTotal saved: {total_saved / 1e6:.1f} MB")
    return lines


def run_tui(config_path: Path | None = None) -> None:
    app = TranscodeApp(config_path)
    app.run()
