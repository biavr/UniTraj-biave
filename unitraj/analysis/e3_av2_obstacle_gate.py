"""E3 - GT-obstacle gate on Argoverse 2 motion forecasting (same question as E1).

Question: does AV2 have enough static obstacles near the predicted agents'
paths to justify obstacle-aware training?

FROZEN DECISION RULE (fixed 2026-11-24, before any AV2 share was computed):
    GO    if >= 30% of targets have >= 1 static obstacle whose box lies within
          3.0 m of the target's GT path (t0 + 6 s future).
    NO-GO otherwise.
    2 m / 5 m radii, per-class shares, background/unknown and scored tracks
    are reported for information only.
    Like-for-like nuScenes reference (decided before results): E1 trainval
    'any_static OR stationary_beside' = 23.9% (E1 headline 20.4%).

Definitions (frozen):
    scenarios : random sample (seed 0) of --num_scenarios complete AV2 train
                scenario folders (parquet readable). Scenario ids are saved.
    targets   : the focal track (object_category == 3, = UniTraj/ScenarioNet
                tracks_to_predict), object_type vehicle or bus, moving.
    path      : focal positions at timesteps 49..109 (t0 + 6 s @ 10 Hz).
    moving    : straight-line displacement start->end of path >= 2.0 m.
    obstacles : ALL rows at timesteps 49..109 (the 'observed' flag is False for
                every future step in train, so it is NOT used), excluding the
                focal track and the AV's own track ('AV'; nuScenes has no ego
                annotation either). Deduplicated by track, min distance kept.
                  static            object_type 'static'
                  construction      object_type 'construction'
                  riderless_bicycle object_type 'riderless_bicycle'
                  parked            vehicle/bus whose WHOLE track (0..109) moves
                                    < 1.0 m and whose centre (at its closest
                                    row) is > 1.5 m from the path line
                info only: background, unknown.
    distance  : oriented box (AV2 heading, size = nuScenes v1.0-trainval median
                [w, l] of the mapped category) to the path polyline (shapely).
                  vehicle->vehicle.car, bus->vehicle.bus.rigid,
                  riderless_bicycle->vehicle.bicycle,
                  construction->movable_object.trafficcone,
                  static/background/unknown->movable_object.pushable_pullable

Usage (unitraj env):
    python e3_av2_obstacle_gate.py --av2_root /workspace/datasets/argoverse2/train \
        --nusc_root /workspace/datasets/nuscenes/v1.0-trainval \
        --num_scenarios 20000 --workers 32 --out /workspace/outputs/e3
"""
import argparse
import csv
import datetime
import glob
import json
import math
import os
import random
import subprocess
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import LineString

# ---- frozen rule -----------------------------------------------------------
PRIMARY_RADIUS = 3.0
GO_SHARE = 0.30
RADII = (2.0, 3.0, 5.0)
MOVING_MIN_DISP = 2.0
T0, T_END = 49, 109
STAT_M, BESIDE_M = 1.0, 1.5
TARGET_TYPES = {"vehicle", "bus"}
CATS = ("static", "construction", "riderless_bicycle", "parked")
INFO_CATS = ("background", "unknown")
NUSC_REF = {"stationary_beside_upper_bound": 0.239, "e1_headline": 0.204}
SIZE_MAP = {
    "vehicle": "vehicle.car", "bus": "vehicle.bus.rigid",
    "riderless_bicycle": "vehicle.bicycle", "construction": "movable_object.trafficcone",
    "static": "movable_object.pushable_pullable", "background": "movable_object.pushable_pullable",
    "unknown": "movable_object.pushable_pullable",
}
# ---------------------------------------------------------------------------

SIZES = {}  # av2 object_type -> (w, l), set in main / worker init


def nusc_median_sizes(nusc_root, version, cache):
    if os.path.exists(cache):
        return json.load(open(cache))
    meta = os.path.join(nusc_root, version)
    cat = {c["token"]: c["name"] for c in json.load(open(os.path.join(meta, "category.json")))}
    inst = {i["token"]: cat[i["category_token"]] for i in json.load(open(os.path.join(meta, "instance.json")))}
    wanted = set(SIZE_MAP.values())
    acc = defaultdict(list)
    for a in json.load(open(os.path.join(meta, "sample_annotation.json"))):
        name = inst[a["instance_token"]]
        if name in wanted:
            acc[name].append(a["size"][:2])  # nuScenes size = [w, l, h]
    med = {name: [float(np.median([s[0] for s in v])), float(np.median([s[1] for s in v])), len(v)]
           for name, v in acc.items()}
    json.dump(med, open(cache, "w"), indent=2)
    return med


def boxes(x, y, yaw, w, l):
    """Vectorised oriented-box polygons; all inputs are length-n arrays."""
    c, s = np.cos(yaw), np.sin(yaw)
    dx = np.stack([l / 2, l / 2, -l / 2, -l / 2], axis=1)
    dy = np.stack([w / 2, -w / 2, -w / 2, w / 2], axis=1)
    px = x[:, None] + c[:, None] * dx - s[:, None] * dy
    py = y[:, None] + s[:, None] * dx + c[:, None] * dy
    return shapely.polygons(np.stack([px, py], axis=-1))


def init_worker(sizes):
    SIZES.update(sizes)


def target_row(df, tid, track_disp, kind):
    t = df[df.track_id == tid].sort_values("timestep")
    t = t[(t.timestep >= T0) & (t.timestep <= T_END)]
    if len(t) != T_END - T0 + 1 or t.object_type.iloc[0] not in TARGET_TYPES:
        return None
    pxy = t[["position_x", "position_y"]].values
    path = LineString(pxy)
    disp = float(np.linalg.norm(pxy[-1] - pxy[0]))

    o = df[(df.timestep >= T0) & (df.timestep <= T_END) & (df.track_id != tid) & (df.track_id != "AV")]
    o = o[o.object_type.isin(set(CATS[:3]) | set(INFO_CATS) | TARGET_TYPES)]
    is_veh = o.object_type.isin(TARGET_TYPES).values
    stat = o.track_id.map(track_disp).values < STAT_M
    o = o[~is_veh | stat]
    res = {c: math.inf for c in CATS + INFO_CATS}
    if len(o):
        wl = np.array([SIZES[ot] for ot in o.object_type])
        d = shapely.distance(boxes(o.position_x.values, o.position_y.values, o.heading.values,
                                   wl[:, 0], wl[:, 1]), path)
        o = o.assign(d=d)
        best = o.loc[o.groupby("track_id").d.idxmin()]
        for _, r in best.iterrows():
            if r.object_type in TARGET_TYPES:
                if shapely.distance(shapely.points(r.position_x, r.position_y), path) <= BESIDE_M:
                    continue
                c = "parked"
            else:
                c = r.object_type
            res[c] = min(res[c], float(r.d))
    row = dict(scenario_id=df.scenario_id.iloc[0], city=df.city.iloc[0], kind=kind, track_id=tid,
               object_type=t.object_type.iloc[0], disp_m=round(disp, 2), moving=disp >= MOVING_MIN_DISP)
    for c in CATS + INFO_CATS:
        row[f"mind_{c}"] = "" if math.isinf(res[c]) else round(res[c], 3)
    s = [res[c] for c in CATS if not math.isinf(res[c])]
    row["mind_any_static"] = round(min(s), 3) if s else ""
    return row


def process(scen_dir):
    try:
        f = glob.glob(os.path.join(scen_dir, "scenario_*.parquet"))
        df = pd.read_parquet(f[0])
    except Exception as e:
        return dict(error=f"{scen_dir}: {e!r}")
    disp = {}
    for tid, g in df.groupby("track_id"):
        xy = g[["position_x", "position_y"]].values
        disp[tid] = float(np.max(np.linalg.norm(xy - xy[0], axis=1)))
    focal = df.focal_track_id.iloc[0]
    scored = df.loc[df.object_category == 2, "track_id"].unique().tolist()
    rows = [target_row(df, focal, disp, "focal")] + [target_row(df, s, disp, "scored") for s in scored]
    return dict(rows=[r for r in rows if r is not None])


def complete(d):
    fs = os.listdir(d)
    return any(x.startswith("scenario_") and x.endswith(".parquet") for x in fs) and \
        any(x.startswith("log_map_archive_") for x in fs)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
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
    ap.add_argument("--av2_root", default="/workspace/datasets/argoverse2/train")
    ap.add_argument("--nusc_root", default="/workspace/datasets/nuscenes/v1.0-trainval")
    ap.add_argument("--nusc_version", default="v1.0-trainval")
    ap.add_argument("--num_scenarios", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", default="/workspace/outputs/e3")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    med = nusc_median_sizes(args.nusc_root, args.nusc_version, os.path.join(args.out, "nusc_median_sizes.json"))
    sizes = {k: tuple(med[v][:2]) for k, v in SIZE_MAP.items()}
    SIZES.update(sizes)

    # Shuffle all folder names (seed), then check completeness only until enough
    # are found: avoids ~200k NFS listings. Identical selection to filtering
    # everything first whenever all folders are complete.
    dirs = sorted(os.path.join(args.av2_root, n) for n in os.listdir(args.av2_root))
    order = dirs[:]
    random.Random(args.seed).shuffle(order)
    ok, n_checked = [], 0
    for d in order:
        n_checked += 1
        if os.path.isdir(d) and complete(d):
            ok.append(d)
            if len(ok) == args.num_scenarios:
                break
    sel = sorted(ok)
    with open(os.path.join(args.out, "e3_scenario_ids.txt"), "w") as f:
        f.write("\n".join(os.path.basename(d) for d in sel) + "\n")

    rows, errors = [], []
    with Pool(args.workers, initializer=init_worker, initargs=(sizes,)) as pool:
        for i, r in enumerate(pool.imap_unordered(process, sel, chunksize=16), 1):
            if "error" in r:
                errors.append(r["error"])
            else:
                rows.extend(r["rows"])
            if i % 2000 == 0:
                print(f"{i}/{len(sel)} scenarios", flush=True)

    with open(os.path.join(args.out, "e3_per_target.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def share(sub, cat, r):
        k = sum(1 for x in sub if x[f"mind_{cat}"] != "" and x[f"mind_{cat}"] <= r)
        n = len(sub)
        lo, hi = wilson(k, n)
        return dict(k=k, n=n, share=k / n if n else float("nan"), ci95=[lo, hi])

    groups = {
        "focal_moving": [x for x in rows if x["kind"] == "focal" and x["moving"]],
        "focal_all": [x for x in rows if x["kind"] == "focal"],
        "scored_moving": [x for x in rows if x["kind"] == "scored" and x["moving"]],
    }
    summary = {g: {c: {f"{r:g}m": share(sub, c, r) for r in RADII}
                   for c in ("any_static",) + CATS + INFO_CATS}
               for g, sub in groups.items()}
    per_city = defaultdict(list)
    for x in groups["focal_moving"]:
        per_city[x["city"]].append(x)
    city_tbl = {c: share(v, "any_static", PRIMARY_RADIUS) for c, v in sorted(per_city.items())}

    primary = summary["focal_moving"]["any_static"][f"{PRIMARY_RADIUS:g}m"]
    decision = "GO" if primary["share"] >= GO_SHARE else "NO-GO"
    import av2
    meta = dict(date=datetime.date.today().isoformat(), av2_root=args.av2_root,
                n_dirs_total=len(dirs), n_dirs_checked=n_checked,
                n_incomplete_skipped=n_checked - len(ok), n_selected=len(sel), seed=args.seed,
                n_read_errors=len(errors), read_errors=errors[:50],
                av2_devkit=getattr(av2, "__version__", "0.3.6"),
                unitraj_commit=git_commit(os.path.dirname(os.path.abspath(__file__))),
                n_focal=len(groups["focal_all"]), n_focal_moving=len(groups["focal_moving"]),
                box_sizes_wl=sizes, nusc_reference=NUSC_REF,
                rule=f"GO if share(focal vehicle/bus moving, any_static, {PRIMARY_RADIUS:g} m) >= {GO_SHARE:.0%}",
                primary=primary, decision=decision)
    json.dump(dict(meta=meta, summary=summary, per_city_focal_moving_3m=city_tbl),
              open(os.path.join(args.out, "e3_summary.json"), "w"), indent=2)

    L = [f"# E3 GT-obstacle gate, Argoverse 2 ({meta['date']})", "",
         f"- data: AV2 motion-forecasting train, {len(sel)} scenarios sampled (seed {args.seed}) "
         f"of {len(dirs)} folders ({n_checked - len(ok)} incomplete skipped); read errors: {len(errors)}",
         f"- av2 devkit {meta['av2_devkit']}, UniTraj {meta['unitraj_commit']}",
         f"- targets: {meta['n_focal']} focal vehicle/bus (moving: {meta['n_focal_moving']})",
         f"- box sizes [w, l] from nuScenes medians: {sizes}",
         f"- rule (frozen before results): {meta['rule']}", "",
         "| category | group | 2 m | 3 m | 5 m |", "|---|---|---|---|---|"]
    for g in groups:
        for c in ("any_static",) + CATS + INFO_CATS:
            cells = [f"{summary[g][c][f'{r:g}m']['share']:.1%} ({summary[g][c][f'{r:g}m']['k']}/"
                     f"{summary[g][c][f'{r:g}m']['n']})" for r in RADII]
            L.append(f"| {c} | {g} | " + " | ".join(cells) + " |")
    L += ["", "Per city (focal moving, any_static, 3 m):", "", "| city | share | k/n |", "|---|---|---|"]
    L += [f"| {c} | {v['share']:.1%} | {v['k']}/{v['n']} |" for c, v in city_tbl.items()]
    lo, hi = primary["ci95"]
    L += ["", "| question | setup | result | decision |", "|---|---|---|---|",
          f"| Does AV2 have enough static obstacles near predicted agents' 6 s paths? "
          f"| AV2 MF train, {len(sel)} scenarios (seed {args.seed}), focal vehicle/bus, av2 {meta['av2_devkit']}, "
          f"UniTraj {meta['unitraj_commit'][:10]}, R={PRIMARY_RADIUS:g} m, nuScenes-median boxes "
          f"| {primary['share']:.1%} of moving focal targets ({primary['k']}/{primary['n']}, "
          f"95% CI {lo:.0%}-{hi:.0%}) have >=1 static obstacle within {PRIMARY_RADIUS:g} m "
          f"(nuScenes like-for-like ref: {NUSC_REF['stationary_beside_upper_bound']:.1%}) "
          f"| **{decision}** (threshold {GO_SHARE:.0%}) |"]
    md = "\n".join(L) + "\n"
    open(os.path.join(args.out, "e3_report.md"), "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
