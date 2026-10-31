"""Until the app and the plugin are out, every way into them on the landing
says "Coming soon" in the button's place, and nothing there starts a chat,
copies an install line or opens the install sheet."""
from __future__ import annotations

from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "frontend" / "src"
LANDING = SRC / "components" / "landing"
ENTRIES = ["Nav.tsx", "Hero.tsx", "Close.tsx", "Footer.tsx", "PluginStrip.tsx"]


def test_every_way_into_the_app_or_the_plugin_says_coming_soon():
    for name in ENTRIES:
        text = (LANDING / name).read_text()
        assert "oming soon" in text, name
        for gone in ("Get started", "Get the plugin", "Get the Claude Code plugin", "onStart", "<CopyButton"):
            assert gone not in text, f"{name}: {gone}"


def test_nothing_on_the_page_starts_a_chat_or_opens_the_install_sheet():
    page = (SRC / "app" / "(landing)" / "page.tsx").read_text()
    for gone in ("startChat", "InstallSheet", "isHosted", "onStart"):
        assert gone not in page, gone
    assert not (LANDING / "InstallSheet.tsx").exists()
