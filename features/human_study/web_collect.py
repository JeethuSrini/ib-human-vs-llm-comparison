"""Flask web app for collecting human baseline responses.

Two-page flow per trial:
  Page 1 (Study):   Show the fact block. Participant reads and memorises.
                    Facts are NOT shown during answering.
  Page 2 (Answer):  Facts hidden. Ask all N memory + reasoning questions.
                    Participant answers from memory only.

Usage:
    # 1. Generate manifest (once)
    python3 main.py --export-human-manifest data/human_manifest.json --manifest-only

    # 2. Run the study server
    python3 -m features.human_study.web_collect \
        --manifest data/human_manifest.json \
        --out data/human_results.csv \
        --port 5050

    # 3. Open http://localhost:5050?pid=P001 in a browser.
       Give each participant a unique ID via ?pid=PXXX in the URL.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

try:
    from flask import Flask, redirect, render_template_string, request, session, url_for
    _FLASK_AVAILABLE = True
except ImportError:
    _FLASK_AVAILABLE = False

# ── shared CSS ────────────────────────────────────────────────────────────────
_BASE_CSS = """
<style>
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    max-width: 720px; margin: 40px auto; padding: 0 24px;
    color: #1a1a1a; line-height: 1.6;
  }
  h2 { margin-bottom: 6px; }
  .meta { color: #666; font-size: 13px; margin-bottom: 20px; }
  pre {
    background: #f0f4f8; padding: 20px; border-radius: 8px;
    font-size: 14.5px; line-height: 1.8; white-space: pre-wrap;
    border-left: 4px solid #4285F4;
  }
  .card {
    background: #fafafa; border: 1px solid #e0e0e0;
    border-radius: 8px; padding: 20px; margin-bottom: 16px;
  }
  label { font-weight: 600; display: block; margin-bottom: 6px; }
  input[type=text] {
    width: 100%; padding: 10px 12px; font-size: 15px;
    border: 1px solid #ccc; border-radius: 6px; margin-bottom: 12px;
  }
  input[type=text]:focus { outline: none; border-color: #4285F4; }
  .conf-row { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
  .conf-row span { font-weight: 600; margin-right: 4px; }
  .conf-row label { font-weight: normal; display: flex; align-items: center; gap: 4px; }
  .btn {
    display: inline-block; margin-top: 24px; padding: 12px 32px;
    font-size: 15px; font-weight: 600;
    background: #1a73e8; color: white; border: none;
    border-radius: 6px; cursor: pointer; text-decoration: none;
  }
  .btn:hover { background: #1557b0; }
  .btn-secondary { background: #5f6368; }
  .btn-secondary:hover { background: #444; }
  .progress-bar-wrap {
    background: #e8eaed; border-radius: 100px; height: 8px;
    margin-bottom: 6px;
  }
  .progress-bar-fill {
    background: #1a73e8; border-radius: 100px; height: 8px;
    transition: width 0.3s;
  }
  .done { text-align: center; padding: 80px 0; }
  .done h2 { font-size: 28px; }
  .warning {
    background: #fff3cd; border: 1px solid #ffc107;
    border-radius: 6px; padding: 12px 16px; margin-bottom: 16px;
    font-size: 13px;
  }
  .q-label { color: #5f6368; font-size: 12px; font-weight: 600;
             text-transform: uppercase; letter-spacing: 0.5px; }
</style>
"""

# ── Page 1: Study the facts ───────────────────────────────────────────────────
_STUDY_TEMPLATE = """
<!doctype html><html><head><meta charset="utf-8">
<title>Memory Study — Read Facts</title>""" + _BASE_CSS + """
</head><body>
<div class="meta">
  Trial {{ trial_num }} of {{ total_trials }} &nbsp;·&nbsp; Participant <strong>{{ pid }}</strong>
</div>
<div class="progress-bar-wrap">
  <div class="progress-bar-fill" style="width:{{ progress_pct }}%"></div>
</div>

<h2>Study these facts carefully</h2>
<p style="color:#444; margin-bottom:16px;">
  Read all the facts below. You will be asked questions about them
  <strong>without</strong> being able to refer back to this page.
</p>

<pre>{{ fact_block }}</pre>

<div class="warning">
  ⚠️ Once you click <strong>Start Questions</strong>, the facts will be hidden.
  Answer entirely from memory.
</div>

<form method="POST" action="{{ url_for('start_questions') }}">
  <input type="hidden" name="pid"        value="{{ pid }}">
  <input type="hidden" name="trial_id"   value="{{ trial_id }}">
  <input type="hidden" name="trial"      value="{{ trial }}">
  <input type="hidden" name="n_examples" value="{{ n_examples }}">
  <input type="hidden" name="fact_block" value="{{ fact_block }}">
  <input type="hidden" name="questions"  value="{{ questions_json }}">
  <button class="btn" type="submit">Start Questions →</button>
</form>
</body></html>
"""

# ── Page 2: Answer questions (facts hidden) ───────────────────────────────────
_ANSWER_TEMPLATE = """
<!doctype html><html><head><meta charset="utf-8">
<title>Memory Study — Answer Questions</title>""" + _BASE_CSS + """
</head><body>
<div class="meta">
  Trial {{ trial_num }} of {{ total_trials }} &nbsp;·&nbsp;
  Participant <strong>{{ pid }}</strong> &nbsp;·&nbsp;
  {{ n_examples }} facts &nbsp;·&nbsp;
  <span style="color:#e53935; font-weight:600;">Facts are now hidden</span>
</div>
<div class="progress-bar-wrap">
  <div class="progress-bar-fill" style="width:{{ progress_pct }}%"></div>
</div>

<h2>Answer from memory</h2>
<p style="color:#444; margin-bottom:20px;">
  Answer all questions below based on the facts you just studied.
  The facts are no longer visible.
</p>

<form method="POST" action="{{ url_for('submit_answers') }}">
  <input type="hidden" name="pid"        value="{{ pid }}">
  <input type="hidden" name="trial_id"   value="{{ trial_id }}">
  <input type="hidden" name="trial"      value="{{ trial }}">
  <input type="hidden" name="n_examples" value="{{ n_examples }}">
  <input type="hidden" name="fact_block" value="{{ fact_block }}">
  <input type="hidden" name="questions"  value="{{ questions_json }}">

  {% for q in questions %}
  <div class="card">
    <div class="q-label">Person {{ loop.index }} of {{ questions|length }}</div>

    <label style="margin-top:8px">
      🧠 Memory — {{ q.memory_question }}
    </label>
    <input type="text" name="memory_{{ loop.index0 }}"
           placeholder="Type the name…" required
           autocomplete="off" spellcheck="false">

    <label>
      🔗 Reasoning — {{ q.reasoning_question }}
    </label>
    <input type="text" name="reasoning_{{ loop.index0 }}"
           placeholder="Type the city name…" required
           autocomplete="off" spellcheck="false">

    <div class="conf-row">
      <span>Confidence:</span>
      {% for i in range(1,6) %}
      <label>
        <input type="radio" name="conf_{{ loop.index0 }}" value="{{ i }}" required>
        {{ i }}
      </label>
      {% endfor %}
      <span style="color:#888; font-size:12px">(1 = guessing, 5 = certain)</span>
    </div>
  </div>
  {% endfor %}

  <button class="btn" type="submit">Submit & Next Trial →</button>
</form>
</body></html>
"""

# ── Done page ─────────────────────────────────────────────────────────────────
_DONE_TEMPLATE = """
<!doctype html><html><head><meta charset="utf-8">
<title>Memory Study — Complete</title>""" + _BASE_CSS + """
</head><body>
<div class="done">
  <h2>🎉 You're done!</h2>
  <p>Thank you, <strong>{{ pid }}</strong>. Your responses have been saved.</p>
  <p style="color:#666;">You may close this window.</p>
</div>
</body></html>
"""

_CSV_FIELDS = [
    "participant_id", "trial_id", "trial", "n_examples",
    "group_id", "target_position",
    "memory_prediction", "reasoning_prediction",
    "memory_truth", "reasoning_truth",
    "confidence", "elapsed_seconds", "timestamp",
]


def make_app(manifest_path: Path, out_path: Path) -> "Flask":
    manifest = json.loads(manifest_path.read_text())
    app = Flask(__name__)
    app.secret_key = os.urandom(24)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_path.exists():
        with out_path.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=_CSV_FIELDS).writeheader()

    def _next_trial(pid: str, answered: set[str]) -> dict | None:
        for trial in manifest:
            if f"{pid}:{trial['trial_id']}" not in answered:
                return trial
        return None

    def _progress(answered_count: int) -> float:
        return round(100 * answered_count / max(len(manifest), 1), 1)

    # ── Route 1: show facts ───────────────────────────────────────────────────
    @app.route("/")
    def index():
        pid = request.args.get("pid", "anonymous")
        session.setdefault("pid", pid)
        session.setdefault("answered", [])
        answered = set(session["answered"])
        trial = _next_trial(pid, answered)
        if trial is None:
            return render_template_string(_DONE_TEMPLATE, pid=pid)
        return render_template_string(
            _STUDY_TEMPLATE,
            pid=pid,
            trial_id=trial["trial_id"],
            trial=trial["trial"],
            n_examples=trial["n_examples"],
            fact_block=trial["fact_block"],
            questions_json=json.dumps(trial["questions"]),
            trial_num=len(answered) + 1,
            total_trials=len(manifest),
            progress_pct=_progress(len(answered)),
        )

    # ── Route 2: hide facts, show questions ───────────────────────────────────
    @app.route("/questions", methods=["POST"])
    def start_questions():
        pid       = request.form["pid"]
        questions = json.loads(request.form["questions"])
        session["study_start"] = time.time()
        return render_template_string(
            _ANSWER_TEMPLATE,
            pid=pid,
            trial_id=request.form["trial_id"],
            trial=request.form["trial"],
            n_examples=int(request.form["n_examples"]),
            fact_block=request.form["fact_block"],
            questions_json=request.form["questions"],
            questions=questions,
            trial_num=len(set(session.get("answered", []))) + 1,
            total_trials=len(manifest),
            progress_pct=_progress(len(set(session.get("answered", [])))),
        )

    # ── Route 3: save answers, advance ───────────────────────────────────────
    @app.route("/submit", methods=["POST"])
    def submit_answers():
        pid       = request.form["pid"]
        questions = json.loads(request.form["questions"])
        elapsed   = round(time.time() - session.get("study_start", time.time()), 2)
        ts        = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        with out_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
            for i, q in enumerate(questions):
                writer.writerow({
                    "participant_id":       pid,
                    "trial_id":             request.form["trial_id"],
                    "trial":                request.form["trial"],
                    "n_examples":           request.form["n_examples"],
                    "group_id":             q["group_id"],
                    "target_position":      q["target_position"],
                    "memory_prediction":    request.form.get(f"memory_{i}", "").strip(),
                    "reasoning_prediction": request.form.get(f"reasoning_{i}", "").strip(),
                    "memory_truth":         q["memory_answer"],
                    "reasoning_truth":      q["reasoning_answer"],
                    "confidence":           request.form.get(f"conf_{i}", ""),
                    "elapsed_seconds":      elapsed,
                    "timestamp":            ts,
                })

        answered = session.get("answered", [])
        answered.append(f"{pid}:{request.form['trial_id']}")
        session["answered"] = answered

        return redirect(url_for("index", pid=pid))

    return app


def main() -> None:
    if not _FLASK_AVAILABLE:
        raise SystemExit("Flask is required:  pip install flask")

    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=Path("data/human_manifest.json"))
    p.add_argument("--out",      type=Path, default=Path("data/human_results.csv"))
    p.add_argument("--port",     type=int,  default=5050)
    p.add_argument("--host",     default="127.0.0.1")
    args = p.parse_args()

    if not args.manifest.exists():
        raise SystemExit(
            f"Manifest not found: {args.manifest}\n"
            "Generate it first:\n"
            "  python3 main.py --export-human-manifest data/human_manifest.json --manifest-only"
        )

    app = make_app(args.manifest, args.out)
    print(f"\nHuman study server running at  http://{args.host}:{args.port}")
    print(f"Participant URLs:  http://{args.host}:{args.port}?pid=P001")
    print(f"                  http://{args.host}:{args.port}?pid=P002  etc.")
    print(f"Results saved to:  {args.out}\n")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
