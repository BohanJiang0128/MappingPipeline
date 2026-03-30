#!/usr/bin/env python3
"""
Step 2 – Web-based QA selection of the best outpainted variant.

Opens a lightweight Flask web server that presents, for each clinical image:

  * the **original clinical image** on the left,
  * the **9 outpainted candidates** in a 3×3 grid on the right.

Click a candidate to select it, press **Skip** to skip an image, or
**Quit** to stop early (progress is saved incrementally).

The server shuts down automatically once every image has been reviewed.
Because this is browser-based it works over SSH — just point your browser
at ``http://<host>:<port>``.

Usage
-----
    python -m steps.qa_select                         # all patients
    python -m steps.qa_select --patient NIH-000021
    python -m steps.qa_select --port 9000             # custom port
"""

import argparse
import glob
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path

from flask import Flask, request, send_file, redirect, render_template_string

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg

app = Flask(__name__)

# ── server state ─────────────────────────────────────────────────────────────

_state = {
    "queue": [],          # [(patient_id, base_name, orig_path, candidates), …]
    "idx": 0,
    "logs": {},           # patient_id → selection-log dict
    "sel_dirs": {},       # patient_id → Path
    "log_paths": {},      # patient_id → Path
    "done": False,
}

_done_event = threading.Event()

# ── HTML ─────────────────────────────────────────────────────────────────────

_HTML = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>QA Selection{% if not done %} – {{ base_name }}{% endif %}</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:'Segoe UI',system-ui,sans-serif;background:#0f0f1a;color:#e0e0e0;min-height:100vh}
  header{background:#16162a;padding:16px 24px;border-bottom:1px solid #2a2a4a;display:flex;
         justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px}
  header h1{font-size:1.25rem;color:#7c8cf8}
  header .meta{font-size:.85rem;color:#888}
  header .meta strong{color:#bbb}
  .wrap{max-width:1400px;margin:0 auto;padding:24px}
  .layout{display:flex;gap:28px;justify-content:center;align-items:flex-start;flex-wrap:wrap}
  .original{text-align:center}
  .original h3{margin-bottom:8px;color:#aaa;font-weight:500;font-size:.9rem;text-transform:uppercase;letter-spacing:.05em}
  .original img{max-height:640px;max-width:420px;border:2px solid #7c8cf8;border-radius:10px}
  .grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
  .cell{position:relative;cursor:pointer;border-radius:10px;overflow:hidden;border:2px solid #222;transition:border-color .15s,transform .12s}
  .cell:hover{border-color:#7c8cf8;transform:scale(1.03)}
  .cell img{display:block;width:260px;height:260px;object-fit:cover}
  .cell .tag{position:absolute;top:6px;left:6px;background:#7c8cf8;color:#fff;font-weight:700;
             font-size:1rem;padding:2px 10px;border-radius:6px}
  .empty{width:260px;height:260px;background:#181830;display:flex;align-items:center;justify-content:center;border-radius:10px;border:2px dashed #2a2a4a}
  .empty span{color:#444;font-size:.85rem}
  .actions{text-align:center;margin-top:28px;display:flex;gap:14px;justify-content:center}
  .btn{display:inline-block;padding:12px 36px;border-radius:8px;text-decoration:none;font-size:.95rem;font-weight:600;transition:background .15s}
  .btn-skip{background:#2a2a4a;color:#ccc}.btn-skip:hover{background:#3a3a5a}
  .btn-quit{background:#c0392b;color:#fff}.btn-quit:hover{background:#e74c3c}
  .done-wrap{text-align:center;padding:120px 24px}
  .done-wrap h2{font-size:1.6rem;color:#7c8cf8;margin-bottom:12px}
  .done-wrap p{color:#888;font-size:1rem}
</style>
</head><body>
{% if done %}
<div class="done-wrap">
  <h2>QA Selection Complete</h2>
  <p>All images have been reviewed.  The server will shut down automatically.</p>
</div>
{% else %}
<header>
  <h1>QA Selection</h1>
  <div class="meta">
    Patient: <strong>{{ patient_id }}</strong> &nbsp;·&nbsp;
    Image: <strong>{{ base_name }}</strong> &nbsp;·&nbsp;
    {{ progress_cur }} / {{ progress_total }}
  </div>
</header>
<div class="wrap">
  <div class="layout">
    <div class="original">
      <h3>Original</h3>
      <img src="/image?p={{ orig_path }}" alt="Original">
    </div>
    <div class="grid">
    {% for pos, cpath in candidates %}
      {% if cpath %}
      <a class="cell" href="/select?pos={{ pos }}">
        <span class="tag">{{ pos }}</span>
        <img src="/image?p={{ cpath }}" alt="pos {{ pos }}">
      </a>
      {% else %}
      <div class="empty"><span>pos {{ pos }}</span></div>
      {% endif %}
    {% endfor %}
    </div>
  </div>
  <div class="actions">
    <a class="btn btn-skip" href="/select?pos=0">Skip</a>
    <a class="btn btn-quit" href="/quit">Quit</a>
  </div>
</div>
{% endif %}
</body></html>
"""

# ── routes ───────────────────────────────────────────────────────────────────

@app.route("/image")
def serve_image():
    """Serve an image file, restricted to the data directory."""
    fpath = request.args.get("p", "")
    real = os.path.realpath(fpath)
    allowed = os.path.realpath(str(cfg.DATA_DIR))
    if not real.startswith(allowed) or not os.path.isfile(real):
        return "Not found", 404
    return send_file(real, mimetype="image/jpeg")


@app.route("/")
def index():
    st = _state
    if st["done"] or st["idx"] >= len(st["queue"]):
        st["done"] = True
        _done_event.set()
        return render_template_string(_HTML, done=True)

    patient_id, base_name, orig_path, candidates = st["queue"][st["idx"]]
    return render_template_string(
        _HTML,
        done=False,
        patient_id=patient_id,
        base_name=base_name,
        orig_path=orig_path,
        candidates=candidates,
        progress_cur=st["idx"] + 1,
        progress_total=len(st["queue"]),
    )


@app.route("/select")
def select():
    st = _state
    if st["done"] or st["idx"] >= len(st["queue"]):
        return redirect("/")

    pos = int(request.args.get("pos", 0))
    patient_id, base_name, orig_path, candidates = st["queue"][st["idx"]]

    log = st["logs"][patient_id]
    sel_dir = st["sel_dirs"][patient_id]
    log_path = st["log_paths"][patient_id]

    if pos == 0:
        log[base_name] = {"position": 0, "status": "skipped"}
        print(f"  Skipped {base_name}")
    elif 1 <= pos <= 9:
        cand_path = None
        for p, cp in candidates:
            if p == pos and cp:
                cand_path = cp
                break
        if cand_path:
            dest_name = f"{base_name}_pos{pos}.jpg"
            dest = sel_dir / dest_name
            shutil.copy2(cand_path, str(dest))
            log[base_name] = {
                "position": pos,
                "file": dest_name,
                "status": "selected",
            }
            print(f"  Selected pos {pos} for {base_name}")

    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    st["idx"] += 1
    return redirect("/")


@app.route("/quit")
def quit_qa():
    _state["done"] = True
    _done_event.set()
    print("  QA interrupted by user (progress saved).")
    return render_template_string(_HTML, done=True)


# ── setup helpers ────────────────────────────────────────────────────────────

def _find_candidates(outpaint_dir: Path, base_name: str):
    candidates = []
    for pos in range(1, cfg.NUM_POSITIONS + 1):
        fname = f"{base_name}_pos{pos}.jpg"
        p = outpaint_dir / fname
        candidates.append((pos, str(p) if p.exists() else None))
    return candidates


def _build_queue(patients: list):
    """Populate the global state with all images needing QA."""
    st = _state
    for patient_id in patients:
        pdir = cfg.patient_dir(patient_id)
        src_dir = pdir / cfg.UNMARKED_IMAGES_DIR
        outpaint_dir = pdir / cfg.OUTPAINTED_DIR
        sel_dir = cfg.ensure_dir(pdir / cfg.QA_SELECTED_DIR)
        log_path = sel_dir / "selection_log.json"

        if log_path.exists():
            with open(log_path) as f:
                log = json.load(f)
        else:
            log = {}

        st["logs"][patient_id] = log
        st["sel_dirs"][patient_id] = sel_dir
        st["log_paths"][patient_id] = log_path

        originals = sorted(glob.glob(str(src_dir / "*.jpg")))

        for orig_path in originals:
            base_name = os.path.splitext(os.path.basename(orig_path))[0]
            if base_name in log:
                continue
            candidates = _find_candidates(outpaint_dir, base_name)
            if all(c[1] is None for c in candidates):
                print(f"  [skip] No outpainted variants for {base_name}")
                continue
            st["queue"].append((patient_id, base_name, orig_path, candidates))


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Step 2: Web-based QA selection")
    parser.add_argument("--patient", type=str, default=None)
    parser.add_argument("--port", type=int, default=8505,
                        help="Port for the web server (default: 8505)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind to (default: 0.0.0.0)")
    args = parser.parse_args()

    patients = [args.patient] if args.patient else cfg.get_patient_ids()
    if not patients:
        print("No patient directories found under", cfg.DATA_DIR)
        sys.exit(1)

    for pid in patients:
        print(f"[Step 2] QA selection: {pid}")

    _build_queue(patients)

    if not _state["queue"]:
        print("  All images already reviewed — nothing to do.")
        return

    print(f"\n  {len(_state['queue'])} image(s) need QA.")
    print(f"  Open  http://localhost:{args.port}  in your browser.\n")

    server = threading.Thread(
        target=lambda: app.run(
            host=args.host, port=args.port, debug=False, use_reloader=False),
        daemon=True,
    )
    server.start()

    _done_event.wait()
    time.sleep(0.5)
    print("\n  QA selection finished.")


if __name__ == "__main__":
    main()
