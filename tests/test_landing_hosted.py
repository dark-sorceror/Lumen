"""The public site at illumen.lol is the landing alone: no chat to forward
to, and a workflow that publishes only the landing."""
from __future__ import annotations

from pathlib import Path

from tests.node_ts import needs_node, run_ts

ROOT = Path(__file__).resolve().parent.parent
HOSTS = ["illumen.lol", "www.illumen.lol", "ILLUMEN.LOL", "dark-sorceror.github.io",
         "localhost", "127.0.0.1", "notillumen.lol", ""]

# Runs the inline script against a stand-in page and returns where it sent it.
FORWARD = """
return INPUT.map((hostname) => {
  let went = null;
  const doc = { documentElement: { style: {} } };
  const loc = { hostname, hash: "#c=abc", search: "", replace: (u) => { went = u; } };
  new Function("document", "location", "addEventListener", "setTimeout", m.FORWARD_SCRIPT)(
    doc, loc, () => {}, () => {});
  return went;
});
"""


@needs_node
def test_an_old_chat_link_forwards_only_where_the_app_is():
    hosted = run_ts("components/landing/served.ts", f"return {HOSTS!r}.map(m.isHosted);")
    went = run_ts("lib/landing.ts", FORWARD.replace("INPUT", repr(HOSTS)))
    assert went == [None if h else "/chat/#c=abc" for h in hosted]
    assert hosted[:4] == [True, True, True, True] and not any(hosted[4:])


def test_the_site_is_published_at_illumen_lol():
    assert (ROOT / "frontend/public/CNAME").read_text().strip() == "illumen.lol"


def test_the_workflow_publishes_the_landing_alone_from_main():
    flow = (ROOT / ".github/workflows/deploy.yaml").read_text()
    assert 'branches: ["main"]' in flow
    build, keep, upload = (flow.index(s) for s in (
        "run: npm run build", "rm -rf out/chat", "uses: actions/upload-pages-artifact"))
    assert build < keep < upload
    # The step fails the job unless the landing is there and the chat is gone.
    assert "test -f out/index.html" in flow and "test ! -e out/chat" in flow
    assert 'test "$(cat out/CNAME)" = illumen.lol' in flow
