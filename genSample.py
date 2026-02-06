#!/usr/bin/env python3
#python3 genSample.py \
#  --db-root ../clump-DB/dataset/shapes \
#  --gradation grad.csv \
#  --out-dir ms_data \
#  --n-shapes 20 \
#  --seed 1234 \
#  --molecule-name clumps_01 \
#  --molecule-out MC \
#  --pour-out POUR \
#  --manifest-base MB \
#  --pour-fixid pour_clumps1 \
#  --pour-ninsert 280 \
#  --pour-nsteps 0 \
#  --pour-seed 4767548 \
#  --pour-region gen_area \
#  --rigid-fix make_clumps_1 \
#  --L-range 0.9,1.1 \
#  --e-range 0.7,0.8 \
#  --f-range 0.6,0.7 \
#  --sphericity-range 0.5,0.9 \
#  --roundness-R1-range 0.4,0.6

import argparse
import hashlib
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any


# -----------------------------
# Helpers
# -----------------------------
def read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def write_text(p: Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def parse_optional_range(s: str) -> Optional[Tuple[float, float]]:
    """
    Accept:
      ""        -> None (no filtering)
      "a,b"     -> (a,b)
      "a,"      -> (a, +inf)
      ",b"      -> (-inf, b)
      "x"       -> (x, x)  (exact match range)
    """
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None

    if "," not in s:
        x = float(s)
        return (x, x)

    a, b = s.split(",", 1)
    a = a.strip()
    b = b.strip()
    lo = float(a) if a else float("-inf")
    hi = float(b) if b else float("inf")
    return (lo, hi)


def within(x: float, r: Optional[Tuple[float, float]]) -> bool:
    if r is None:
        return True
    lo, hi = r
    return (x >= lo) and (x <= hi)


def load_gradation_csv(path: Path) -> List[Tuple[float, float]]:
    """
    Returns sorted [(D, passing)] with passing in [0,1].
    CSV header can be D,passing (case-insensitive) or no header.
    """
    lines = [
        ln.strip()
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    if not lines:
        raise ValueError(f"Empty gradation CSV: {path}")

    rows: List[Tuple[float, float]] = []

    first = lines[0].lower()
    start = 1 if ("pass" in first and "d" in first) else 0

    for ln in lines[start:]:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 2:
            continue
        D = float(parts[0])
        P = float(parts[1])
        rows.append((D, P))

    if not rows:
        raise ValueError(f"Empty/invalid gradation CSV (no data rows): {path}")

    # normalize passing: if looks like 0~100, scale down
    maxP = max(p for _, p in rows)
    if maxP > 1.5:
        rows = [(d, p / 100.0) for d, p in rows]

    rows.sort(key=lambda x: x[0])
    rows = [(d, clamp(p, 0.0, 1.0)) for d, p in rows]
    return rows


def gradation_to_bin_fractions(
    grad: List[Tuple[float, float]]
) -> List[Tuple[float, float, float]]:
    """
    Convert cumulative passing curve to per-bin fractions.
    For points (D0,P0), (D1,P1), ...:
      bin (D0, D1] fraction = P1 - P0
    Return list of (D_lo, D_hi, frac) normalized to sum=1.
    """
    out: List[Tuple[float, float, float]] = []
    for i in range(1, len(grad)):
        d0, p0 = grad[i - 1]
        d1, p1 = grad[i]
        frac = p1 - p0
        if frac < 0:
            raise ValueError("Passing curve must be non-decreasing.")
        if frac > 0:
            out.append((d0, d1, frac))

    s = sum(f for _, _, f in out)
    if s <= 0:
        raise ValueError("Gradation fractions sum to 0.")
    out = [(a, b, f / s) for a, b, f in out]
    return out


def repr_D_geomean(dlo: float, dhi: float) -> float:
    # geometric mean
    return math.sqrt(float(dlo) * float(dhi))


def short_hash8(payload: Dict[str, Any]) -> str:
    """
    Stable 8-hex hash from JSON-serialized payload (sorted keys).
    """
    b = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(b).hexdigest()[:8]


def fmt_float(x: Any) -> str:
    if x is None:
        return "null"
    try:
        return f"{float(x):.8g}"
    except Exception:
        return str(x)


# -----------------------------
# Data model
# -----------------------------
@dataclass
class Case:
    case_id: str
    case_dir: Path
    meta: Dict[str, Any]
    L: float
    e: float
    f: float
    sphericity: Optional[float]
    round_R1: Optional[float]
    round_R2: Optional[float]
    r_in: Optional[float]
    D_in: Optional[float]


def extract_case(case_dir: Path) -> Optional[Case]:
    meta_path = case_dir / "meta.json"
    mol_path = case_dir / "molecule_mc.data"
    if not meta_path.exists() or not mol_path.exists():
        return None

    meta = read_json(meta_path)
    cid = meta.get("case_id") or case_dir.name

    sp = meta.get("shape_params", {})
    L = float(sp.get("L", float("nan")))
    e = float(sp.get("e", sp.get("I_over_L", float("nan"))))
    f = float(sp.get("f", sp.get("S_over_I", float("nan"))))

    metrics = meta.get("metrics", {})
    sph = metrics.get("sphericity_riley1941_style", None)
    wad = metrics.get("wadell", {})
    R1 = wad.get("R1", None)
    R2 = wad.get("R2", None)
    r_in = wad.get("r_in", None)
    D_in = wad.get("D_in", None)

    def fopt(x):
        return None if x is None else float(x)

    return Case(
        case_id=str(cid),
        case_dir=case_dir,
        meta=meta,
        L=float(L),
        e=float(e),
        f=float(f),
        sphericity=fopt(sph),
        round_R1=fopt(R1),
        round_R2=fopt(R2),
        r_in=fopt(r_in),
        D_in=fopt(D_in),
    )


def list_cases(root: Path) -> List[Case]:
    cases: List[Case] = []
    for d in root.iterdir():
        if not d.is_dir():
            continue
        c = extract_case(d)
        if c is not None:
            cases.append(c)
    return cases


# -----------------------------
# Emitters (LAMMPS includes)
# -----------------------------
def wrap_molecule_line(
    molecule_name: str,
    template_parts: List[str],
    max_per_line: int = 5,
) -> str:
    """
    Build wrapped:
      molecule  NAME part part part &
                     part part ...
    where each part is: "ms_data/msr1.data scale 0.123"
    """
    head = f"molecule         {molecule_name} "
    lines: List[str] = []

    cur = head
    cnt = 0
    for p in template_parts:
        if cnt >= max_per_line:
            lines.append(cur.rstrip() + " &")
            cur = " " * len("molecule         ") + " " * len(molecule_name) + " " + p + " "
            cnt = 1
        else:
            cur += p + " "
            cnt += 1
    lines.append(cur.rstrip())
    return "\n".join(lines) + "\n"


def wrap_molfrac(
    molfracs: List[float],
    prefix: str,
    per_line: int = 10,
) -> str:
    """
    Wrap molfrac numeric list across multiple lines with '&' continuation.
    prefix should include everything before first number (e.g., 'fix ... molfrac ').
    """
    out_lines: List[str] = []
    cur = prefix
    cnt = 0
    for v in molfracs:
        s = f"{v:.8g}"
        if cnt >= per_line:
            out_lines.append(cur.rstrip() + " &")
            cur = " " * len(prefix) + s + " "
            cnt = 1
        else:
            cur += s + " "
            cnt += 1
    out_lines.append(cur.rstrip())
    return "\n".join(out_lines) + "\n"


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Pick N shapes once, copy ONLY N molecule files, then create ONE molecule definition "
            "that contains (bins*N) scaled templates + ONE fix pour include using molfrac to mix bins."
        )
    )

    # essentials
    ap.add_argument("--db-root", default="dataset/shapes", help="clump-DB case root (contains <case_id>/meta.json)")
    ap.add_argument("--gradation", required=True, help="gradation CSV: D,passing")
    ap.add_argument("--out-dir", default="ms_data", help="output dir for copied molecule files (msr1..msrN)")
    ap.add_argument("--n-shapes", type=int, default=20, help="how many unique shapes to keep (ONLY these files are created)")
    ap.add_argument("--seed", type=int, default=1234, help="random seed for selection")
    ap.add_argument("--molecule-name", default="clumps_01", help="LAMMPS molecule name (single ID)")

    # output base names (hash will be appended)
    ap.add_argument("--molecule-out", default="molecule_table.in", help="base name for molecule include (hash will be inserted)")
    ap.add_argument("--pour-out", default="fix_pour_clumps_01.in", help="base name for pour include (hash will be inserted)")
    ap.add_argument("--manifest-base", default="selection_manifest", help="manifest base name (hash will be appended)")

    # pour parameters (kept minimal & useful)
    ap.add_argument("--pour-fixid", default="pour_clumps1", help="fix id for pour")
    ap.add_argument("--pour-ninsert", type=int, default=280, help="fix pour: N to insert per step (first numeric after 'pour')")
    ap.add_argument("--pour-nsteps", type=int, default=0, help="fix pour: total steps argument (second numeric after 'pour')")
    ap.add_argument("--pour-seed", type=int, default=4767548, help="seed used in fix pour (independent from selection seed)")
    ap.add_argument("--pour-region", default="gen_area", help="region name used in fix pour")
    ap.add_argument("--rigid-fix", default="make_clumps_1", help="rigid/small fix id used at end of fix pour line")

    # optional filters (empty => no filtering)
    ap.add_argument("--L-range", default="", help="e.g. 0.8,1.2  or 0.9, or ,1.1 ; empty => no filter")
    ap.add_argument("--e-range", default="", help="e.g. 0.7,0.8  or 0.7, or ,0.8 ; empty => no filter")
    ap.add_argument("--f-range", default="", help="e.g. 0.6,0.7  or 0.6, or ,0.7 ; empty => no filter")
    ap.add_argument("--sphericity-range", default="", help="e.g. 0.65,0.75 or 0.65, or ,0.75 ; empty => no filter")
    ap.add_argument("--roundness-R1-range", default="", help="e.g. 0.7,0.95 or 0.7, or ,0.95 ; empty => no filter")
    ap.add_argument("--roundness-R2-range", default="", help="e.g. 0.7,0.95 or 0.7, or ,0.95 ; empty => no filter")

    args = ap.parse_args()
    random.seed(int(args.seed))

    db_root = Path(args.db_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # parse filters
    Lr = parse_optional_range(args.L_range)
    er = parse_optional_range(args.e_range)
    fr = parse_optional_range(args.f_range)
    sr = parse_optional_range(args.sphericity_range)
    r1r = parse_optional_range(args.roundness_R1_range)
    r2r = parse_optional_range(args.roundness_R2_range)

    # load cases
    cases = list_cases(db_root)
    if not cases:
        raise SystemExit(f"No valid cases found under: {db_root}")

    def case_ok(c: Case) -> bool:
        if not within(c.L, Lr):
            return False
        if not within(c.e, er):
            return False
        if not within(c.f, fr):
            return False
        # if metric missing, we accept it (do not fail)
        if c.sphericity is not None and not within(c.sphericity, sr):
            return False
        if c.round_R1 is not None and not within(c.round_R1, r1r):
            return False
        if c.round_R2 is not None and not within(c.round_R2, r2r):
            return False
        # must have a positive base diameter
        return c.D_in is not None and c.D_in > 0

    pool = [c for c in cases if case_ok(c)]
    if not pool:
        raise SystemExit("No cases satisfy the given parameter ranges / required fields (need D_in > 0).")

    n_shapes = int(args.n_shapes)
    if n_shapes <= 0:
        raise SystemExit("--n-shapes must be > 0")
    if len(pool) < n_shapes:
        raise SystemExit(f"Not enough cases in pool ({len(pool)}) for n-shapes={n_shapes}")

    # ---- Pick shapes ONCE (unique) ----
    chosen_shapes = random.sample(pool, n_shapes)

    # ---- Copy ONLY these N files ----
    shape_table: List[Dict[str, Any]] = []
    for i, c in enumerate(chosen_shapes, start=1):
        src = c.case_dir / "molecule_mc.data"
        dst = out_dir / f"msr{i}.data"
        shutil.copy2(src, dst)
        shape_table.append(
            {
                "msr": i,
                "case_id": c.case_id,
                "case_dir": str(c.case_dir),
                "src": str(src),
                "dst": str(dst),
                "shape_params": {"L": c.L, "e": c.e, "f": c.f},
                "metrics": {
                    "sphericity": c.sphericity,
                    "round_R1": c.round_R1,
                    "round_R2": c.round_R2,
                    "D_in": c.D_in,
                    "r_in": c.r_in,
                },
            }
        )

    # ---- Gradation bins ----
    grad = load_gradation_csv(Path(args.gradation))
    bins = gradation_to_bin_fractions(grad)  # (Dlo, Dhi, frac), sum=1
    nbins = len(bins)
    if nbins <= 0:
        raise SystemExit("No bins derived from gradation curve.")

    # ---- Build ONE molecule definition containing ALL templates (bins*N) ----
    # Template ordering: BIN1 shape1..N, BIN2 shape1..N, ...
    # Each template is (ms_data/msr{i}.data scale <scale_for_this_bin>)
    template_parts: List[str] = []

    # also compute molfrac list aligned with template order above
    # fraction per template = bin_frac / N
    molfracs: List[float] = []

    # store bin blocks for header info
    bin_info: List[Dict[str, Any]] = []

    for bi, (dlo, dhi, frac) in enumerate(bins, start=1):
        Dtarget = repr_D_geomean(dlo, dhi)
        bin_info.append(
            {
                "bin_index": bi,
                "Dlo": float(dlo),
                "Dhi": float(dhi),
                "bin_frac": float(frac),
                "Dtarget_geomean": float(Dtarget),
                "templates": n_shapes,
            }
        )

        for i, c in enumerate(chosen_shapes, start=1):
            baseD = float(c.D_in)
            scale = float(Dtarget) / baseD
            rel = (out_dir / f"msr{i}.data").as_posix()
            template_parts.append(f"{rel}  scale {scale:.8g}")
            molfracs.append(float(frac) / float(n_shapes))

    # sanity
    molfrac_sum = sum(molfracs)
    # (floating errors allowed)
    if not (0.999999 <= molfrac_sum <= 1.000001):
        raise SystemExit(f"Internal error: molfrac sum = {molfrac_sum} (expected 1.0)")

    # ---- Compose header (human-readable, rich info) ----
    header_lines: List[str] = []
    header_lines.append("# Auto-generated by genSample.py")
    header_lines.append("#")
    header_lines.append("# === Filters (omitted/empty => no constraint) ===")
    header_lines.append(f"# L-range             : {args.L_range!r}")
    header_lines.append(f"# e-range             : {args.e_range!r}")
    header_lines.append(f"# f-range             : {args.f_range!r}")
    header_lines.append(f"# sphericity-range    : {args.sphericity_range!r}")
    header_lines.append(f"# roundness-R1-range  : {args.roundness_R1_range!r}")
    header_lines.append(f"# roundness-R2-range  : {args.roundness_R2_range!r}")
    header_lines.append("#")
    header_lines.append("# === Gradation points (D, passing) ===")
    for (d, p) in grad:
        header_lines.append(f"#   D={fmt_float(d)}  passing={fmt_float(p)}")
    header_lines.append("#")
    header_lines.append("# === Derived bins (Dlo, Dhi, frac, Dtarget_geomean) ===")
    for b in bin_info:
        header_lines.append(
            f"#   BIN {b['bin_index']:02d}: ({fmt_float(b['Dlo'])}, {fmt_float(b['Dhi'])}] "
            f"frac={fmt_float(b['bin_frac'])}  Dtarget={fmt_float(b['Dtarget_geomean'])}"
        )
    header_lines.append("#")
    header_lines.append("# === Selected shapes (N unique) ===")
    for s in shape_table:
        sp = s["shape_params"]
        mt = s["metrics"]
        header_lines.append(
            "#   msr{msr:02d} case={cid}  L={L} e={e} f={f}  sph={sph} R1={R1} R2={R2}  D_in={Din}".format(
                msr=s["msr"],
                cid=s["case_id"],
                L=fmt_float(sp.get("L")),
                e=fmt_float(sp.get("e")),
                f=fmt_float(sp.get("f")),
                sph=fmt_float(mt.get("sphericity")),
                R1=fmt_float(mt.get("round_R1")),
                R2=fmt_float(mt.get("round_R2")),
                Din=fmt_float(mt.get("D_in")),
            )
        )
    header_lines.append("#")
    header_lines.append(f"# bins={nbins}, n_shapes={n_shapes}, templates={nbins*n_shapes}")
    header_lines.append("#")

    # ---- Build manifest payload for hashing ----
    hash_payload = {
        "db_root": str(db_root),
        "gradation": str(Path(args.gradation)),
        "n_shapes": n_shapes,
        "seed": int(args.seed),
        "molecule_name": args.molecule_name,
        "filters": {
            "L_range": args.L_range,
            "e_range": args.e_range,
            "f_range": args.f_range,
            "sphericity_range": args.sphericity_range,
            "roundness_R1_range": args.roundness_R1_range,
            "roundness_R2_range": args.roundness_R2_range,
        },
        "bins": [(float(a), float(b), float(f)) for (a, b, f) in bins],
        "selected_case_ids": [c.case_id for c in chosen_shapes],
        "pour": {
            "fixid": args.pour_fixid,
            "ninsert": int(args.pour_ninsert),
            "nsteps": int(args.pour_nsteps),
            "seed": int(args.pour_seed),
            "region": args.pour_region,
            "rigid_fix": args.rigid_fix,
        },
    }
    hid = short_hash8(hash_payload)

    # ---- Output filenames with hash inserted ----
    def with_hash(base: str) -> str:
        p = Path(base)
        if p.suffix:
            return f"{p.stem}_{hid}{p.suffix}"
        return f"{base}_{hid}"

    molecule_out = Path(with_hash(args.molecule_out))
    pour_out = Path(with_hash(args.pour_out))
    manifest_out = Path(with_hash(args.manifest_base)).with_suffix(".json")

    # ---- Write molecule include ----
    molecule_body = wrap_molecule_line(args.molecule_name, template_parts, max_per_line=5)
    write_text(molecule_out, "\n".join(header_lines) + "\n" + molecule_body)

    # ---- Write pour include (ONE fix pour line) ----
    # IMPORTANT: molfrac list length MUST match number of templates in molecule definition.
    # The template count is bins * n_shapes.
    prefix = (
        f"# Auto-generated by genSample.py (hash={hid})\n"
        f"# molfrac count = {len(molfracs)} (must match molecule templates)\n"
        f"fix {args.pour_fixid} all pour {args.pour_ninsert} {args.pour_nsteps} {args.pour_seed} "
        f"region {args.pour_region} mol {args.molecule_name} molfrac "
    )
    pour_line = wrap_molfrac(molfracs, prefix=prefix, per_line=10)
    # add rigid fix at end (same line) -> easiest: append to last line
    # Our wrap_molfrac returns multi-line with last line not ending with '\n'? It ends with '\n' already.
    # We'll modify by trimming the trailing '\n' and adding the rigid clause.
    pour_line = pour_line.rstrip("\n") + f" rigid {args.rigid_fix}\n"
    write_text(pour_out, pour_line)

    # ---- Write manifest JSON (rich info) ----
    manifest = {
        "hash_id": hid,
        "db_root": str(db_root),
        "out_dir": str(out_dir),
        "gradation_csv": str(Path(args.gradation)),
        "seed": int(args.seed),
        "n_shapes": n_shapes,
        "molecule_name": args.molecule_name,
        "filters": hash_payload["filters"],
        "gradation_points": [{"D": float(d), "passing": float(p)} for (d, p) in grad],
        "bins": bin_info,
        "selected_shapes": shape_table,
        "template_count": nbins * n_shapes,
        "molfrac_count": len(molfracs),
        "molfrac_sum": molfrac_sum,
        "outputs": {
            "molecule_include": str(molecule_out),
            "pour_include": str(pour_out),
            "manifest": str(manifest_out),
        },
        "pour": hash_payload["pour"],
        # mapping for debugging: template index -> (bin, msr, scale, molfrac)
        "template_map": [
            {
                "template_index": idx + 1,
                "bin_index": (idx // n_shapes) + 1,
                "msr": (idx % n_shapes) + 1,
                "mol_file": (out_dir / f"msr{(idx % n_shapes) + 1}.data").as_posix(),
                "molfrac": molfracs[idx],
            }
            for idx in range(len(molfracs))
        ],
    }
    write_text(manifest_out, json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    # ---- Console summary ----
    print("[OK] Copied ONLY N shape molecules into:", out_dir)
    print(f"[OK] N = {n_shapes}  (msr1..msr{n_shapes}.data)")
    print("[OK] Bins =", nbins, " Templates =", nbins * n_shapes)
    print("[OK] Molecule include :", molecule_out)
    print("[OK] Pour include     :", pour_out)
    print("[OK] Manifest JSON    :", manifest_out)
    print("[OK] hash_id          :", hid)
    print("[OK] molfrac_sum      :", molfrac_sum, "(should be 1.0)")


if __name__ == "__main__":
    main()