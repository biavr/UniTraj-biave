"""E1 - GT-obstacle gate (plan step 2).

Question: does nuScenes have enough static obstacles near the predicted agents'
paths to justify obstacle-aware training?

FROZEN DECISION RULE (fixed 2026-11-23, before any results were seen):
    GO    if >= 30% of MOVING prediction targets have >= 1 static obstacle
          whose box footprint lies within 3.0 m of the target's GT path
          (current position + 6 s future).
    NO-GO otherwise.
    2 m and 5 m radii and the all-targets share are reported for information only.

Definitions (also frozen):
    targets   : official nuScenes prediction-challenge split tokens
                ({instance}_{sample}) for the chosen split(s).
    path      : current position + 12 future keyframes (6 s @ 2 Hz), global xy.
    moving    : straight-line displacement start->end of path >= 2.0 m.
    obstacles : annotations in keyframes t0..t0+6 s (deduplicated by instance,
                min distance over appearances), excluding the target itself:
                  cone             movable_object.trafficcone
                  barrier          movable_object.barrier
                  construction_veh vehicle.construction, not attributed 'vehicle.moving'
                  parked_vehicle   any other vehicle.* attributed 'vehicle.parked'
                  other_static     movable_object.debris / .pushable_pullable,
                                   static_object.bicycle_rack
                'stopped_vehicle' (vehicle.stopped) is reported for information
                only and is NOT counted in 'any_static'.
    distance  : shapely distance between the obstacle's oriented box footprint
                and the path polyline (0 if they intersect).

Usage (unitraj env):
    python e1_obstacle_gate.py --dataroot /workspace/datasets/nuscenes/nuscenes \
        --version v1.0-mini --splits mini_train mini_val --out /workspace/outputs/e1
"""
import argparse
import csv
import datetime
import json
import math
import os
import subprocess
from collections import defaultdict

import numpy as np
from pyquaternion import Quaternion
from shapely.geometry import LineString, Point, Polygon

import nuscenes
from nuscenes import NuScenes
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.prediction import PredictHelper

# ---- frozen rule -----------------------------------------------------------
PRIMARY_RADIUS = 3.0
GO_SHARE = 0.30
RADII = (2.0, 3.0, 5.0)
MOVING_MIN_DISP = 2.0
FUTURE_S = 6.0
N_FUTURE = 12
CATS = ("cone", "barrier", "construction_veh", "parked_vehicle", "other_static")
INFO_CATS = ("stopped_vehicle",)
# ---------------------------------------------------------------------------

OTHER_STATIC = {"movable_object.debris", "movable_object.pushable_pullable",
                "static_object.bicycle_rack"}


def classify(nusc, ann):
    cat = ann["category_name"]
    attrs = {nusc.get("attribute", t)["name"] for t in ann["attribute_tokens"]}
    if cat == "movable_object.trafficcone":
        return "cone"
    if cat == "movable_object.barrier":
        return "barrier"
    if cat in OTHER_STATIC:
        return "other_static"
    if cat == "vehicle.construction":
        return None if "vehicle.moving" in attrs else "construction_veh"
    if cat.startswith("vehicle.") and cat not in ("vehicle.bicycle", "vehicle.motorcycle"):
        if "vehicle.parked" in attrs:
            return "parked_vehicle"
        if "vehicle.stopped" in attrs:
            return "stopped_vehicle"
    return None


def footprint(ann):
    x, y = ann["translation"][:2]
    w, l = ann["size"][0], ann["size"][1]
    yaw = Quaternion(ann["rotation"]).yaw_pitch_roll[0]
    c, s = math.cos(yaw), math.sin(yaw)
    corners = [(+l / 2, +w / 2), (+l / 2, -w / 2), (-l / 2, -w / 2), (-l / 2, +w / 2)]
    return Polygon([(x + c * dx - s * dy, y + s * dx + c * dy) for dx, dy in corners])


def window_samples(nusc, sample_token, n_next):
    toks, tok = [sample_token], sample_token
    for _ in range(n_next):
        tok = nusc.get("sample", tok)["next"]
        if not tok:
            break
        toks.append(tok)
    return toks


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def git_commit(path):
    try:
        sha = subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", path, "status", "--porcelain"], text=True).strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", default="/workspace/datasets/nuscenes/nuscenes")
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--splits", nargs="+", default=["mini_train", "mini_val"])
    ap.add_argument("--out", default="/workspace/outputs/e1")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    nusc = NuScenes(args.version, dataroot=args.dataroot, verbose=False)
    helper = PredictHelper(nusc)

    rows = []
    for split in args.splits:
        for tok in get_prediction_challenge_split(split, dataroot=args.dataroot):
            inst, samp = tok.split("_")
            cur = np.array(helper.get_sample_annotation(inst, samp)["translation"][:2])
            fut = helper.get_future_for_agent(inst, samp, seconds=FUTURE_S, in_agent_frame=False)
            path_xy = np.vstack([cur[None], np.asarray(fut).reshape(-1, 2)])
            path = LineString(path_xy) if len(path_xy) > 1 else Point(path_xy[0])
            disp = float(np.linalg.norm(path_xy[-1] - path_xy[0]))

            mind = {}  # obstacle instance -> (cat, min dist)
            for st in window_samples(nusc, samp, N_FUTURE):
                for at in nusc.get("sample", st)["anns"]:
                    ann = nusc.get("sample_annotation", at)
                    if ann["instance_token"] == inst:
                        continue
                    cat = classify(nusc, ann)
                    if cat is None:
                        continue
                    d = footprint(ann).distance(path)
                    prev = mind.get(ann["instance_token"])
                    if prev is None or d < prev[1]:
                        mind[ann["instance_token"]] = (cat, d)

            scene = nusc.get("scene", nusc.get("sample", samp)["scene_token"])["name"]
            row = dict(token=tok, split=split, scene=scene, n_future=len(path_xy) - 1,
                       disp_m=round(disp, 2), moving=disp >= MOVING_MIN_DISP)
            for c in CATS + INFO_CATS:
                ds = [d for cc, d in mind.values() if cc == c]
                row[f"mind_{c}"] = round(min(ds), 3) if ds else ""
            static_ds = [d for cc, d in mind.values() if cc in CATS]
            row["mind_any_static"] = round(min(static_ds), 3) if static_ds else ""
            for r in RADII:
                row[f"n_static_within_{r:g}m"] = sum(d <= r for d in static_ds)
            rows.append(row)

    with open(os.path.join(args.out, "e1_per_target.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def share(subset, cat, r):
        k = sum(1 for x in subset if x[f"mind_{cat}"] != "" and x[f"mind_{cat}"] <= r)
        n = len(subset)
        lo, hi = wilson(k, n)
        return dict(k=k, n=n, share=k / n if n else float("nan"), ci95=[lo, hi])

    groups = {"moving": [x for x in rows if x["moving"]], "all": rows}
    summary = {g: {c: {f"{r:g}m": share(sub, c, r) for r in RADII}
                   for c in ("any_static",) + CATS + INFO_CATS}
               for g, sub in groups.items()}
    per_scene = defaultdict(list)
    for x in groups["moving"]:
        per_scene[x["scene"]].append(x)
    scene_tbl = {s: share(v, "any_static", PRIMARY_RADIUS) for s, v in sorted(per_scene.items())}

    primary = summary["moving"]["any_static"][f"{PRIMARY_RADIUS:g}m"]
    decision = "GO" if primary["share"] >= GO_SHARE else "NO-GO"
    meta = dict(date=datetime.date.today().isoformat(), version=args.version, splits=args.splits,
                dataroot=args.dataroot, nuscenes_devkit=getattr(nuscenes, "__version__", "1.2.0"),
                unitraj_commit=git_commit(os.path.dirname(os.path.abspath(__file__))),
                n_targets=len(rows), n_moving=len(groups["moving"]),
                n_short_future=sum(x["n_future"] < N_FUTURE for x in rows),
                rule=f"GO if share(moving, any_static, {PRIMARY_RADIUS:g} m) >= {GO_SHARE:.0%}",
                primary=primary, decision=decision)
    json.dump(dict(meta=meta, summary=summary, per_scene_moving_3m=scene_tbl),
              open(os.path.join(args.out, "e1_summary.json"), "w"), indent=2)

    # markdown report
    L = [f"# E1 GT-obstacle gate ({meta['date']})", "",
         f"- data: {args.version}, splits {', '.join(args.splits)} (official prediction split)",
         f"- nuscenes-devkit {meta['nuscenes_devkit']}, UniTraj {meta['unitraj_commit']}",
         f"- targets: {meta['n_targets']} (moving: {meta['n_moving']}, "
         f"<12 future frames: {meta['n_short_future']})",
         f"- rule (frozen before results): {meta['rule']}", "",
         "| category | group | 2 m | 3 m | 5 m |", "|---|---|---|---|---|"]
    for g in ("moving", "all"):
        for c in ("any_static",) + CATS + INFO_CATS:
            cells = []
            for r in RADII:
                s = summary[g][c][f"{r:g}m"]
                cells.append(f"{s['share']:.1%} ({s['k']}/{s['n']})")
            L.append(f"| {c} | {g} | " + " | ".join(cells) + " |")
    L += ["", "Per scene (moving targets, any_static, 3 m):", "",
          "| scene | share | k/n |", "|---|---|---|"]
    L += [f"| {s} | {v['share']:.1%} | {v['k']}/{v['n']} |" for s, v in scene_tbl.items()]
    lo, hi = primary["ci95"]
    L += ["", "| question | setup | result | decision |", "|---|---|---|---|",
          f"| Does nuScenes have enough static obstacles near predicted agents' 6 s paths? "
          f"| {args.version} {'+'.join(args.splits)}, devkit {meta['nuscenes_devkit']}, "
          f"UniTraj {meta['unitraj_commit'][:10]}, R={PRIMARY_RADIUS:g} m "
          f"| {primary['share']:.1%} of moving targets ({primary['k']}/{primary['n']}, "
          f"95% CI {lo:.0%}-{hi:.0%}) have >=1 static obstacle within {PRIMARY_RADIUS:g} m "
          f"| **{decision}** (threshold {GO_SHARE:.0%}) |"]
    md = "\n".join(L) + "\n"
    open(os.path.join(args.out, "e1_report.md"), "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
