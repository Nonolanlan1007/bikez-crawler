"""Generate a static, local HTML page for reviewing/correcting zero-shot labels.

No server involved: open the generated file directly in a browser (``file://``). It
shows sampled images with their predicted category/subject, lets you correct them with
a couple of dropdowns, keeps your progress in the browser's localStorage as you go, and
lets you export everything you've reviewed as a corrections CSV that
``retrain_from_corrections.py`` consumes.

Images are prioritized lowest-confidence-first (most likely to be wrong), with a random
sample of higher-confidence ones mixed in to spot-check those too.

Usage:
    uv run python scripts/bootstrap_labels.py               # first, if not already done
    uv run python scripts/generate_review_page.py
    open scripts/review/review.html
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

from bikez_crawler.training.taxonomy import CATEGORY_PROMPTS, SUBJECT_PROMPTS

SCRIPT_DIR = Path(__file__).resolve().parent
REVIEW_DIR = SCRIPT_DIR / "review"

HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Bikez label review</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: -apple-system, system-ui, sans-serif; max-width: 720px; margin: 24px auto; padding: 0 16px; }
  header { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 12px; }
  #progress { color: #888; font-size: 0.9em; }
  .card { border: 1px solid #8884; border-radius: 8px; padding: 16px; }
  .card img { max-width: 100%; max-height: 60vh; display: block; margin: 0 auto 12px; border-radius: 4px; }
  .meta { font-size: 0.85em; color: #888; margin-bottom: 12px; word-break: break-all; }
  .field { display: flex; align-items: center; gap: 8px; margin-bottom: 10px; }
  .field label { width: 90px; font-weight: 600; }
  .field select { flex: 1; padding: 4px; }
  .conf { font-size: 0.8em; color: #888; }
  .buttons { display: flex; gap: 8px; margin-top: 16px; flex-wrap: wrap; }
  button { padding: 8px 14px; border-radius: 6px; border: 1px solid #8884; background: #8881; cursor: pointer; font-size: 0.95em; }
  button:hover { background: #8882; }
  button.primary { background: #2b6; border-color: #2b6; color: white; }
  #export-bar { margin-top: 20px; padding-top: 16px; border-top: 1px solid #8884; }
  #status { font-size: 0.85em; color: #888; margin-top: 8px; }
</style>
</head>
<body>
<header>
  <h2>Bikez label review</h2>
  <span id="progress"></span>
</header>
<div class="card">
  <img id="photo" alt="sampled bike image">
  <div class="meta" id="meta"></div>
  <div class="field">
    <label for="category">category</label>
    <select id="category"></select>
    <span class="conf" id="category-conf"></span>
  </div>
  <div class="field">
    <label for="subject">subject</label>
    <select id="subject"></select>
    <span class="conf" id="subject-conf"></span>
  </div>
  <div class="buttons">
    <button id="prev">&larr; prev</button>
    <button id="save" class="primary">save &amp; next &rarr;</button>
    <button id="skip">skip &rarr;</button>
  </div>
</div>
<div id="export-bar">
  <button id="export">export corrections.csv</button>
  <span id="status"></span>
</div>
<script>
const DATA = __DATA_JSON__;
const CATEGORY_LABELS = __CATEGORIES_JSON__;
const SUBJECT_LABELS = __SUBJECTS_JSON__;
const STORAGE_KEY = "bikez-corrections-v1";

function loadCorrections() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
  } catch (e) {
    return {};
  }
}

function saveCorrections(corrections) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(corrections));
  } catch (e) {
    // best-effort only; a full export still works from in-memory state
  }
}

let corrections = loadCorrections();
let index = 0;

function fillSelect(select, options, value) {
  select.innerHTML = "";
  for (const opt of options) {
    const el = document.createElement("option");
    el.value = opt;
    el.textContent = opt;
    if (opt === value) el.selected = true;
    select.appendChild(el);
  }
}

function render() {
  const row = DATA[index];
  const saved = corrections[row.image_key];
  document.getElementById("photo").src = row.image_src;
  document.getElementById("meta").textContent = row.image_key;
  fillSelect(document.getElementById("category"), CATEGORY_LABELS, saved ? saved.category : row.category);
  fillSelect(document.getElementById("subject"), SUBJECT_LABELS, saved ? saved.subject : row.subject);
  document.getElementById("category-conf").textContent = "predicted " + row.category + " (" + row.category_conf.toFixed(2) + ")";
  document.getElementById("subject-conf").textContent = "predicted " + row.subject + " (" + row.subject_conf.toFixed(2) + ")";
  const reviewed = Object.keys(corrections).length;
  document.getElementById("progress").textContent = index + 1 + " / " + DATA.length + " (" + reviewed + " saved)";
}

function save() {
  const row = DATA[index];
  corrections[row.image_key] = {
    local_path: row.local_path,
    category: document.getElementById("category").value,
    subject: document.getElementById("subject").value,
  };
  saveCorrections(corrections);
}

document.getElementById("save").addEventListener("click", () => {
  save();
  index = Math.min(index + 1, DATA.length - 1);
  render();
});
document.getElementById("skip").addEventListener("click", () => {
  index = Math.min(index + 1, DATA.length - 1);
  render();
});
document.getElementById("prev").addEventListener("click", () => {
  index = Math.max(index - 1, 0);
  render();
});
document.getElementById("export").addEventListener("click", () => {
  const rows = [["image_key", "local_path", "category", "subject", "category_conf", "subject_conf"]];
  for (const [imageKey, c] of Object.entries(corrections)) {
    rows.push([imageKey, c.local_path, c.category, c.subject, "1.0", "1.0"]);
  }
  const csv = rows.map((r) => r.map((v) => '"' + String(v).replace(/"/g, '""') + '"').join(",")).join("\\n");
  const blob = new Blob([csv], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "corrections.csv";
  a.click();
  URL.revokeObjectURL(url);
  document.getElementById("status").textContent = "exported " + rows.length + " rows (move it to scripts/data/corrections.csv)";
});

render();
</script>
</body>
</html>
"""


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=SCRIPT_DIR / "data" / "bootstrap_labels_all.csv")
    parser.add_argument("--out", type=Path, default=REVIEW_DIR / "review.html")
    parser.add_argument("--priority-count", type=int, default=120, help="lowest-confidence images to include")
    parser.add_argument("--random-count", type=int, default=80, help="extra random images to spot-check")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = load_rows(args.labels)
    if not rows:
        raise SystemExit(f"no rows in {args.labels}; run bootstrap_labels.py first")

    for row in rows:
        row["_min_conf"] = str(min(float(row["category_conf"]), float(row["subject_conf"])))

    rows_by_conf = sorted(rows, key=lambda r: float(r["_min_conf"]))
    priority = rows_by_conf[: args.priority_count]
    remaining = rows_by_conf[args.priority_count :]

    random.seed(args.seed)
    random.shuffle(remaining)
    spot_check = remaining[: args.random_count]

    selected = priority + spot_check
    random.shuffle(selected)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for row in selected:
        image_src = "../" + row["local_path"]
        records.append(
            {
                "image_key": row["image_key"],
                "local_path": row["local_path"],
                "image_src": image_src,
                "category": row["category"],
                "subject": row["subject"],
                "category_conf": float(row["category_conf"]),
                "subject_conf": float(row["subject_conf"]),
            }
        )

    html = (
        HTML_TEMPLATE.replace("__DATA_JSON__", json.dumps(records))
        .replace("__CATEGORIES_JSON__", json.dumps(list(CATEGORY_PROMPTS.keys())))
        .replace("__SUBJECTS_JSON__", json.dumps(list(SUBJECT_PROMPTS.keys())))
    )
    args.out.write_text(html)
    print(f"wrote {len(records)} images ({len(priority)} low-confidence + {len(spot_check)} spot-check) -> {args.out}")
    print("open it directly in a browser, e.g.: open " + str(args.out))


if __name__ == "__main__":
    main()
