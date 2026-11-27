"""Did tiles really move? Every pickup and return of a session, as the RealSense saw the mat before and after, for a
person to check what the scans said (above all an unattended run's self-reported bumps: bench.py --unattended).

Scans get this wrong both ways: answers to 212 scan changes up to 2026-10-30 were mostly detection glitches, and a bump
under interventions.KNOCK_M doesn't show as a change at all. So each row is a tile near the one the arm handled, whether
or not the scans saw it change, and a person can say either way. The raw frames flip in place (same crop, no drawing on
them), which is what shows a 5mm shift on a 25-pixel tile.

A verdict is a `scan_verdict` event appended to the session's events.jsonl, the latest per change standing. Nothing else
is rewritten: interventions.derive reads them, so the session's knocks and clean counts follow.
"""
import html
import json
import math
from datetime import datetime
from statistics import median

import cv2

from . import config, interventions as iv
from .session import append_jsonl, read_events

PAGE, FOLDER = "bump_report.html", "bump_report"
FRAME = "f0.jpg"            # a scan's frames are near copies; the camera doesn't move between scans, so crops line up
CROP_PAD_PX = 34
MIN_CROP_PX = (220, 150)
BOX_HALF_PX = (17, 20)      # clear of a tile's ~24x29px, so the box hides none of its edges
SHOWN_W, MAX_ZOOM = 720, 4
AMBER, WHITE, DARK = (0, 190, 255), (255, 255, 255), (20, 20, 20)
VERDICTS = ("moved", "not_moved")
SAID = {"knocked": "Bump (self-reported)", "scan_glitch": "A glitch, the scans say", "unsettled": "Not settled: the session ended",
        "moved_elsewhere": "Moved, away from the arm", None: "The scans saw no change"}
EXPLAINED = "Explained in the log"


def _said(reviews, seen, scan_n, category, change):
    """(what the session's scan_reviews concluded about a change: a kind, EXPLAINED or None; their note)."""
    for r in reversed(reviews):     # a review with no `changes` answers for its whole scan (older sessions, a look again)
        if r["kinds"] and ("changes" not in r or change in r["changes"].get(category, [])):
            return r["kinds"][0], r.get("note")
    for r in reversed(reviews):
        if not r["kinds"] and change in r.get("changes", {}).get(category, []) and not (r.get("note") or "").startswith("watching"):
            return EXPLAINED, r.get("note")
    if category == "gone" and (scan_n, tuple(change["xy"])) in seen:
        return "scan_glitch", "the wrist camera or a faint box saw it still there"
    return "unsettled", None


def _phase(before, after, proposal, reviews, seen, verdicts, attempts, landings, skipped_resets):
    scan_n, target, handled = after["scan"], after.get("target_xy"), iv.handled_points(after)
    items, shown = [], []

    def add(item_id, category, change, what, said, note=None):
        real = "knocked" if iv.near_any(change, handled, iv.NEIGHBOUR_M) else "moved_elsewhere"
        verdict = verdicts.get((scan_n, category, json.dumps(change, sort_keys=True)))
        items.append({"id": item_id, "scan": scan_n, "category": category, "change": change, "what": what, "said": said,
                      "note": note, "kind_if_real": real, "flagged": said == "knocked",
                      "verdict": verdict["verdict"] if verdict else None})

    for category, changes in (proposal[1] if proposal else {}).items():
        for i, change in enumerate(changes):
            said, note = _said(reviews, seen, scan_n, category, change)
            add(f"{scan_n}-{category}-{i}", category, change, iv.changes_text(iv.merge({category: [change]})), said, note)
            shown += iv.where(change)
    for i, tile in enumerate(before["tiles"]):      # the neighbours the scans saw no change in: a person can still see one
        if not handled or iv.near_any({"xy": tile["xy"]}, shown, iv.KNOCK_M) or not iv.on_mat(tile):
            continue
        if not config.TILE_MERGE_RADIUS_M < min(math.dist(tile["xy"], p) for p in handled) <= iv.NEIGHBOUR_M:
            continue
        shift = min((math.dist(tile["xy"], t["xy"]) for t in after["tiles"]), default=math.inf)
        # not moved or gone, so a tile too unsure for the scans to count its going (CHANGE_CONFIDENCE), or one matched elsewhere
        what = (f"{tile['tile_name']} shifted {shift * 1000:.1f}mm" if shift <= iv.KNOCK_M
                else f"{tile['tile_name']}, not detected after (too faint to count as gone)")
        add(f"{scan_n}-n-{i}", "turned", {"tile_name": tile["tile_name"], "xy": list(tile["xy"])}, what, None)
    during = after["label"].removeprefix("after_") if after["label"] in iv.ATTEMPT_SCANS else None
    return {"scan": scan_n, "before": before, "after": after, "during": during, "items": items, "target": target,
            "attempt_dir": after.get("attempt_dir"),
            "result": _result(after, during, attempts, landings, skipped_resets)}


def _result(after, during, attempts, landings, skipped_resets):
    attempt_dir = after.get("attempt_dir")
    if during == "pickup" and attempt_dir in attempts:
        a = attempts[attempt_dir]
        where = a.get("spot") or a.get("tile")
        return f"{where}, {a.get('policy')}: {'picked up' if a.get('success') else 'missed'}"
    if during == "return":
        landed = landings.get(attempt_dir)
        if not landed or landed.get("error_m") is None:
            return "the returned tile wasn't seen near its home"
        return (f"landed {landed['error_m'] * 1000:.0f}mm from its home"
                + ("; far off and no one to put it back, so left there" if attempt_dir in skipped_resets else ""))
    return "between attempts, the arm at rest"


def phases(events):
    """Each pickup and return in a session (and any scan between them that saw a change), oldest first: {scan, before,
    after (detections events), during, items: rows, one per change or neighbour, result}."""
    reviews, seen = {}, set()
    attempts, landings = {}, {}
    skipped_resets = {e["attempt_dir"] for e in events if e["type"] == "reset_skipped"}
    verdicts = iv.latest_verdicts(events)
    mat, out = iv.MatWatch(), []
    for e in events:        # first, as settled reviews are logged after the scan they're about
        if e["type"] == "scan_review":
            reviews.setdefault(e["scan"], []).append(e)
        elif e["type"] == "attempt":
            attempts[e["attempt_dir"]] = e
        elif e["type"] == "return_landed":
            landings[e["attempt_dir"]] = e
        elif (e["type"] == "wrist_check" and e.get("present")) or e["type"] == "tile_faint":
            seen.add((e["scan"], tuple(e["xy"])))
    for e in events:
        if (e["type"] == "wrist_check" and e.get("present")) or e["type"] == "tile_faint":
            mat.restore({"tile_name": e["tile"], "xy": e["xy"]})
        if e["type"] != "detections":
            continue
        before, proposal = mat.previous, mat.propose(e)
        mat.advance(e)
        if before is not None and (e["label"] in iv.ATTEMPT_SCANS or proposal):
            out.append(_phase(before, e, proposal, reviews.get(e["scan"], []), seen, verdicts, attempts, landings,
                              skipped_resets))
    return out


def counts(events, found):
    items = [i for p in found for i in p["items"]]
    flagged = [i for i in items if i["flagged"]]
    attempts = [e for e in events if e["type"] == "attempt"]
    return {"attempts": len(attempts), "picked": sum(bool(a.get("success")) for a in attempts), "flagged": len(flagged),
            "checked": sum(i["verdict"] is not None for i in flagged),
            "confirmed": sum(i["verdict"] == "moved" for i in flagged),
            "rejected": sum(i["verdict"] == "not_moved" for i in flagged),
            "missed": sum(i["verdict"] == "moved" and i["said"] != "knocked" for i in items),
            "left_off_home": sum(e["type"] == "reset_skipped" for e in events),
            "skipped": sum(e["type"] == "attempt_skipped" for e in events)}


# --- pictures ---

def _pixel(tile, reads):
    """Where a logged tile is in its scan's frames: its `pixel`, or (sessions before it was logged) by its name."""
    if any(tile.get("pixel") or ()):
        return tuple(tile["pixel"])
    boxes = [b["xyxy"] for frame in reads["frames"] for b in frame if b["tile_name"] == tile["tile_name"]]
    if not boxes:
        return None
    return median((b[0] + b[2]) / 2 for b in boxes), median((b[1] + b[3]) / 2 for b in boxes)


def _crop(points, width, height):
    """One crop (x0, y0, x1, y1) around the points, the same for both scans so the flip shows only what changed."""
    if not points:
        return 0, 0, width, height
    x0, x1 = min(p[0] for p in points) - CROP_PAD_PX, max(p[0] for p in points) + CROP_PAD_PX
    y0, y1 = min(p[1] for p in points) - CROP_PAD_PX, max(p[1] for p in points) + CROP_PAD_PX
    grow_x, grow_y = max(0, MIN_CROP_PX[0] - (x1 - x0)) / 2, max(0, MIN_CROP_PX[1] - (y1 - y0)) / 2
    return (max(0, int(x0 - grow_x)), max(0, int(y0 - grow_y)),
            min(width, int(x1 + grow_x) + 1), min(height, int(y1 + grow_y) + 1))


def _text(image, text, at, color=WHITE, scale=0.8):
    cv2.putText(image, text, at, cv2.FONT_HERSHEY_SIMPLEX, scale, DARK, 5, cv2.LINE_AA)
    cv2.putText(image, text, at, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def build_images(session_dir, phase):
    """Write the phase's crops under FOLDER: both scans with each tile named and changes boxed, and raw, to flip between.
    Sets phase["images"], or leaves it out if a scan kept no frames."""
    scans = phase["before"], phase["after"]
    if not all(s.get("frames") and (session_dir / s["frames"] / FRAME).exists() for s in scans):
        return
    frames = [cv2.imread(str(session_dir / s["frames"] / FRAME)) for s in scans]
    reads = [json.loads((session_dir / s["frames"] / "reads.json").read_text()) for s in scans]
    changed_at = [xy for item in phase["items"] if item["category"] and item["said"] for xy in iv.where(item["change"])]
    near = [xy for item in phase["items"] for xy in iv.where(item["change"])] + ([phase["target"]] if phase["target"] else [])
    placed = []             # per scan: (tile, its pixel) for the tiles in the picture
    for scan, read in zip(scans, reads):
        # a restored tile (confidence 0) with no pixel has no box here: looked up by name, it would be another tile's
        tiles = [t for t in scan["tiles"] if (t.get("confidence") != 0 or t.get("pixel"))
                 and iv.near_any({"xy": t["xy"]}, near, 2 * iv.KNOCK_M)]
        placed.append([(t, p) for t in tiles if (p := _pixel(t, read)) is not None])
    h, w = frames[0].shape[:2]
    x0, y0, x1, y1 = _crop([p for tiles in placed for _, p in tiles], w, h)
    zoom = min(MAX_ZOOM, SHOWN_W / (x1 - x0))
    folder = session_dir / FOLDER
    folder.mkdir(exist_ok=True)
    phase["images"] = {}
    for name, frame, tiles in zip(("before", "after"), frames, placed):
        big = cv2.resize(frame[y0:y1, x0:x1], None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
        raw = big.copy()
        _text(raw, name.upper(), (8, 28), AMBER)        # in the corner, off the tiles
        for tile, (u, v) in tiles:
            cx, cy = int((u - x0) * zoom), int((v - y0) * zoom)
            half_w, half_h = int(BOX_HALF_PX[0] * zoom), int(BOX_HALF_PX[1] * zoom)
            if iv.near_any({"xy": tile["xy"]}, changed_at, 2 * iv.KNOCK_M):
                cv2.rectangle(big, (cx - half_w, cy - half_h), (cx + half_w, cy + half_h), AMBER, 2)
            _text(big, tile["tile_name"], (cx - half_w, cy - half_h - 6), AMBER if tile["xy"] == phase["target"] else WHITE)
        _text(big, name.upper(), (8, 28), AMBER)
        for suffix, image in (("", big), ("_raw", raw)):
            path = f"{FOLDER}/{phase['scan']:03d}_{name}{suffix}.jpg"
            cv2.imwrite(str(session_dir / path), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
            phase["images"][name + suffix] = path


# --- the page ---

STYLE = """
:root { --bg:#f6f6f4; --card:#fff; --ink:#20201e; --soft:#67675f; --line:#dcdcd6; --flag:#b45f00; --flagbg:#fff3df;
        --yes:#a32a1c; --no:#23703f; --on:#fff; }
@media (prefers-color-scheme: dark) { :root { --bg:#191918; --card:#242422; --ink:#ececE6; --soft:#a4a49a; --line:#3a3a36;
        --flag:#f0a63c; --flagbg:#3a2c14; --yes:#e8705f; --no:#5cc484; --on:#191918; } }
* { box-sizing:border-box; }
body { margin:0; padding:20px 16px 80px; background:var(--bg); color:var(--ink); font:15px/1.45 system-ui,sans-serif; }
main { max-width:1180px; margin:auto; }
h1 { font-size:22px; margin:0 0 2px; } .soft { color:var(--soft); }
#summary { position:sticky; top:0; z-index:2; background:var(--bg); padding:10px 0; border-bottom:1px solid var(--line); margin-bottom:16px; }
#readonly { background:var(--flagbg); padding:8px 12px; border-radius:6px; margin:10px 0; display:none; }
details { background:var(--card); border:1px solid var(--line); border-radius:8px; margin:10px 0; }
details.flag { border-color:var(--flag); }
summary { cursor:pointer; padding:10px 14px; font-weight:600; } summary .soft { font-weight:400; }
.tag { color:var(--flag); font-weight:600; margin-left:6px; }
.body { padding:0 14px 14px; }
.pics { display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:8px; }
.pics figure { margin:0; } .pics img { width:100%; display:block; border-radius:4px; }
figcaption { font-size:13px; color:var(--soft); margin-top:3px; }
.flip { position:relative; } .flip img + img { position:absolute; inset:0; animation:flip 1.4s steps(1) infinite; }
.flip:hover img + img { animation-play-state:paused; }
@keyframes flip { 50% { opacity:0; } }
table { width:100%; border-collapse:collapse; margin-top:10px; }
td,th { text-align:left; padding:6px 8px; border-top:1px solid var(--line); vertical-align:top; }
th { font-size:13px; color:var(--soft); font-weight:500; border-top:0; }
tr.flag td { background:var(--flagbg); } .note { font-size:13px; color:var(--soft); }
td.act { white-space:nowrap; text-align:right; }
button { font:inherit; font-size:14px; padding:4px 10px; margin-left:4px; border:1px solid var(--line); border-radius:5px; background:transparent; color:var(--ink); cursor:pointer; }
button[data-verdict=moved].on { background:var(--yes); border-color:var(--yes); color:var(--on); }
button[data-verdict=not_moved].on { background:var(--no); border-color:var(--no); color:var(--on); }
@media (max-width:640px) { td.act { white-space:normal; } }
"""

SCRIPT = """
const readonly = location.protocol === 'file:';
if (readonly) {
  document.getElementById('readonly').style.display = 'block';
  document.querySelectorAll('button[data-verdict]').forEach(b => b.disabled = true);
}
function showCounts(c) {
  document.getElementById('progress').textContent =
    `${c.checked} of ${c.flagged} self-reported bumps checked: ${c.confirmed} really moved, ${c.rejected} didn't. ` +
    `${c.missed} other tile(s) you said moved.`;
}
document.addEventListener('click', async e => {
  const b = e.target.closest('button[data-verdict]');
  if (!b) return;
  const row = b.closest('tr');
  const r = await fetch('verdict', {method: 'POST', headers: {'Content-Type': 'application/json'},
                                    body: JSON.stringify({id: row.dataset.id, verdict: b.dataset.verdict})});
  if (!r.ok) { alert('Not saved: ' + await r.text()); return; }
  row.querySelectorAll('button').forEach(o => o.classList.toggle('on', o === b));
  showCounts(await r.json());
});
"""


def _row(item):
    said = SAID.get(item["said"], iv.KINDS[item["said"]][0] if item["said"] in iv.KINDS else item["said"])
    note = f"<div class=note>{html.escape(item['note'])}</div>" if item["note"] else ""
    buttons = "".join(f'<button data-verdict="{v}"{" class=on" if item["verdict"] == v else ""}>{label}</button>'
                      for v, label in zip(VERDICTS, ("Moved", "Didn't move")))
    return (f'<tr data-id="{item["id"]}"{" class=flag" if item["flagged"] else ""}><td>{html.escape(item["what"])}</td>'
            f"<td>{html.escape(said)}{note}</td><td class=act>{buttons}</td></tr>")


def _card(phase):
    flagged = sum(i["flagged"] for i in phase["items"])
    changed_rows = any(i["said"] for i in phase["items"])
    title = phase["attempt_dir"] or f"scan {phase['scan']}"
    during = f" {phase['during']}" if phase["during"] else ""
    tag = f'<span class=tag>{flagged} bump{"s" if flagged > 1 else ""}</span>' if flagged else ""
    pics = "<p class=soft>No frames were kept for these scans.</p>"
    if phase.get("images"):
        i = phase["images"]
        pics = (f'<div class=pics><figure><img loading=lazy src="{i["before"]}" alt="before"><figcaption>Before '
                f'(<a href="{phase["before"]["frames"]}/{FRAME}">whole frame</a>)</figcaption></figure>'
                f'<figure><img loading=lazy src="{i["after"]}" alt="after"><figcaption>After '
                f'(<a href="{phase["after"]["frames"]}/{FRAME}">whole frame</a>)</figcaption></figure>'
                f'<figure><div class=flip><img loading=lazy src="{i["after_raw"]}" alt="after, nothing drawn">'
                f'<img loading=lazy src="{i["before_raw"]}" alt="before, nothing drawn"></div>'
                "<figcaption>Flipping before / after, nothing drawn. Hover to pause.</figcaption></figure></div>")
    rows = "".join(_row(i) for i in sorted(phase["items"], key=lambda i: (not i["flagged"], not i["said"])))
    table = (f"<table><tr><th>Tile</th><th>What the scans said</th><th></th></tr>{rows}</table>" if rows
             else "<p class=soft>No neighbouring tiles in these scans.</p>")
    return (f'<details{" class=flag" if flagged else ""}{" open" if changed_rows else ""}><summary>{html.escape(title)}{during}'
            f'{tag} <span class=soft>{html.escape(phase["result"])}</span></summary><div class=body>{pics}{table}</div></details>')


def render(session_name, found, c, started=None):
    """The whole page for a session's phases (`found`) and counts (`c`)."""
    other = "; ".join(t for t in (
        f"{c['left_off_home']} tile(s) landed too far off to be picked up again" if c["left_off_home"] else "",
        f"{c['skipped']} attempt(s) skipped" if c["skipped"] else "") if t)
    return f"""<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Bump Report</title><style>{STYLE}</style></head><body><main>
<div id=summary><h1>Bump report: {html.escape(session_name)}</h1>
<div class=soft>{c['attempts']} attempts, {c['picked']} picked up{'. ' + html.escape(other) if other else ''}</div>
<div id=progress>{c['checked']} of {c['flagged']} self-reported bumps checked: {c['confirmed']} really moved, {c['rejected']} didn't.
{c['missed']} other tile(s) you said moved.</div></div>
<div id=readonly>Read-only: to save verdicts, open this with <code>python scripts/bump_report.py sessions/{html.escape(session_name)}</code>.</div>
<p class=soft>Cards with a change are open. A self-reported bump stands as a knock until you say it didn't move. Quiet cards
are closed: open one if you think the scans missed something, and mark the tile "Moved". Tiles moved less than {iv.KNOCK_M * 1000:.0f}mm don't count as a change in the scans.</p>
{''.join(_card(p) for p in found)}
{'<p class=soft>No pickups or returns with scans in this session.</p>' if not found else ''}
</main><script>{SCRIPT}</script></body></html>"""


def load(session_dir):
    """(events, phases with their pictures built)."""
    events = read_events(session_dir)
    found = phases(events)
    for phase in found:
        build_images(session_dir, phase)
    return events, found


def write(session_dir):
    """Write session_dir/PAGE (readable as a file; saving verdicts needs scripts/bump_report.py). Returns counts."""
    events, found = load(session_dir)
    c = counts(events, found)
    (session_dir / PAGE).write_text(render(session_dir.name, found, c))
    return c


def record_verdict(session_dir, found, item_id, verdict):
    """A person's word on one row: appended to the session's events.jsonl as a `scan_verdict`, and set on `found`."""
    if verdict not in VERDICTS:
        raise ValueError(f"unknown verdict {verdict!r}")
    item = next((i for p in found for i in p["items"] if i["id"] == item_id), None)
    if item is None:
        raise ValueError(f"no row {item_id!r} in this session")
    record = {"time": datetime.now().isoformat(timespec="seconds"), "type": "scan_verdict", "scan": item["scan"],
              "category": item["category"], "change": item["change"], "verdict": verdict, "kind": item["kind_if_real"]}
    append_jsonl(session_dir / "events.jsonl", [record])
    item["verdict"] = verdict
    return record
