"""Phase 3 - inject the payload into the template to produce one standalone file."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import OUT, ROOT

tpl = (ROOT / "dashboard" / "template.html").read_text()
payload = (OUT / "dashboard_payload.json").read_text()
assert "__PAYLOAD__" in tpl
html = tpl.replace("__PAYLOAD__", payload)
out = ROOT / "dashboard" / "lifesnaps_triple_audit.html"
out.write_text(html)
print(f"wrote {out}  ({out.stat().st_size/1e6:.2f} MB)")
