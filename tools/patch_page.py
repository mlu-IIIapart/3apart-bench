"""Add tester name, machine name and a "Download result" button to the benchmark page.

Usage: python tools/patch_page.py <page.html> <out.html>
Used to build web/index.html from the published page, and to update the published page itself.
Idempotent: running it on an already-patched page changes nothing.
"""
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src, encoding="utf-8").read()
if 'id="who"' in s:
    open(dst, "w", encoding="utf-8").write(s)
    sys.exit(0)

CSS = """
.who-row { display: flex; flex-wrap: wrap; gap: 10px; }
.who-row input { appearance: none; border: 1px solid var(--axis); background: transparent; color: var(--ink); font: 400 14px/1 var(--font); padding: 10px 14px; border-radius: 999px; min-width: 0; flex: 1 1 180px; max-width: 260px; }
.who-row input::placeholder { color: var(--ink-3); }
"""
s = s.replace("/* runs */", CSS + "\n/* runs */", 1)

ROW = """      <div class="who-row">
        <input id="who" type="text" autocomplete="name" placeholder="Your name" aria-label="Your name" maxlength="40">
        <input id="mname" type="text" placeholder="Machine name, e.g. Lorium" aria-label="Machine name" maxlength="40">
      </div>
      <div class="runbar">"""
s = s.replace('      <div class="runbar">', ROW, 1)
s = s.replace(
    '<button class="btn" id="stop" type="button" hidden>Stop</button>',
    '<button class="btn" id="stop" type="button" hidden>Stop</button>\n'
    '        <button class="btn" id="dl" type="button" hidden>Download result</button>', 1)

# label each run with who and which machine, keep it for download
s = s.replace(
    "gpuState.mine = res; showGpu('mine');",
    "const whoV = ($('#who').value || '').trim(), mV = ($('#mname').value || '').trim();\n"
    "    if (whoV) res.who = whoV.toLowerCase().replace(/[^a-z0-9-]+/g, '-');\n"
    "    if (mV) res.label = [mV, res.env.gpu, res.env.browser].filter(Boolean).join(' · ');\n"
    "    res.kind = 'browser'; res.schema = 1; res.machine = { name: mV || null, gpu: res.env.gpu, os: res.env.os, cores: res.env.cores };\n"
    "    try { localStorage.setItem('bench-who', whoV); localStorage.setItem('bench-machine', mV); } catch (e) { /* storage blocked */ }\n"
    "    LAST = res; $('#dl').hidden = false;\n"
    "    gpuState.mine = res; showGpu('mine');", 1)

TAIL = """
// ---------------------------------------------------------------------------- tester labels and download
let LAST = null;
try { $('#who').value = localStorage.getItem('bench-who') || ''; $('#mname').value = localStorage.getItem('bench-machine') || ''; } catch (e) { /* storage blocked */ }
$('#dl').addEventListener('click', () => {
  if (!LAST) return;
  const slug = x => String(x || 'unknown').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
  const name = `${LAST.at.replace(/[:]/g, '').replace(/\\..+/, '')}-browser.json`;
  const blob = new Blob([JSON.stringify(LAST, null, 1) + '\\n'], { type: 'application/json' });
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = name;
  document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  $('#status').textContent = `Saved ${name}. In the repo it goes in results/${slug(LAST.who)}/${slug(LAST.machine && LAST.machine.name)}/.`;
});
"""
idx = s.rfind("</script>")
s = s[:idx] + TAIL + s[idx:]
open(dst, "w", encoding="utf-8").write(s)
