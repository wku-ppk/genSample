#!/usr/bin/env python3
#python3 genSample.py --db-root "/경로/to/clump-DB/dataset/shapes" --gradation grad.csv
#python3 genSample.py --db-root ""/Volumes/z640/GitHub/clump-DB/dataset/shapes"" --gradation grad.csv

#python3 genSample.py \
#  --db-root /Volumes/z640/GitHub/clump-DB/dataset/shapes \
#  --gradation gradation.csv \
#  --out-dir ms_data \
#  --n-shapes 20 \
#  --seed 1234 \
#  --L-range 0.9,1.1 \
#  --e-range 0.7,0.8 \
#  --f-range 0.6,0.7 \
#  --out-script molecule_table.in \
#  --manifest-json selection_manifest.json

#!/usr/bin/env python3
import argparse
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# -----------------------------
# Helpers
# -----------------------------
def read_json(p: Path) -> Dict:
    return json.loads(p.read_text(encoding="utf-8"))

def write_text(p: Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def parse_float_pair(s: str) -> Tuple[float, float]:
    a, b = s.split(",")
    return float(a), float(b)

def parse_optional_range(s: str) -> Optional[Tuple[float, float]]:
    if not s:
        return None
    return parse_float_pair(s)

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
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"Empty gradation CSV: {path}")

    rows = []
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
        raise ValueError(f"Empty gradation CSV: {path}")

    maxP = max(p for _, p in rows)
    if maxP > 1.5:  # assume 0~100
        rows = [(d, p / 100.0) for d, p in rows]

    rows.sort(key=lambda x: x[0])
    rows = [(d, clamp(p, 0.0, 1.0)) for d, p in rows]
    return rows

def gradation_to_bin_fractions(grad: List[Tuple[float, float]]) -> List[Tuple[float, float, float]]:
    """
    Convert cumulative passing curve to per-bin fractions:
      bin (D0, D1] fraction = P1 - P0
    Return list of (D_lo, D_hi, frac) normalized to sum=1.
    """
    out = []
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

def wrap_pairs(prefix1: str, prefix_cont: str, pairs: List[str], max_per_line: int = 5) -> str:
    """
    Wrap a list of tokens (already formatted as "file scale x") with '&' line continuation.
    """
    lines = []
    cur = prefix1
    cnt = 0
    for tok in pairs:
        if cnt >= max_per_line:
            lines.append(cur.rstrip() + " &")
            cur = prefix_cont + tok + " "
            cnt = 1
        else:
            cur += tok + " "
            cnt += 1
    lines.append(cur.rstrip())
    return "\n".join(lines) + "\n"

def wrap_numbers(prefix1: str, prefix_cont: str, nums: List[float], per_line: int = 10, fmt: str = "{:.8g}") -> str:
    """
    Wrap numeric list with '&' and continuation indentation.
    """
    lines = []
    cur = prefix1
    cnt = 0
    for x in nums:
        tok = fmt.format(x)
        if cnt >= per_line:
            lines.append(cur.rstrip() + " &")
            cur = prefix_cont + tok + " "
            cnt = 1
        else:
            cur += tok + " "
            cnt += 1
    lines.append(cur.rstrip())
    return "\n".join(lines) + "\n"

# -----------------------------
# Data model
# -----------------------------
@dataclass
class Case:
    case_id: str
    case_dir: Path
    meta: Dict
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
    cases = []
    for d in root.iterdir():
        if not d.is_dir():
            continue
        c = extract_case(d)
        if c is not None:
            cases.append(c)
    return cases

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Generate LAMMPS include files: molecule_<name>.in and fix_pour_<name>.in (single molecule ID with BIN-mixed molfrac)."
    )
    ap.add_argument("--db-root", default="dataset/shapes", help="clump-DB case root")
    ap.add_argument("--gradation", required=True, help="gradation CSV: D,passing")

    ap.add_argument("--out-dir", default="ms_data", help="output dir for copied molecule files (msr1..msrN)")
    ap.add_argument("--n-shapes", type=int, default=20, help="number of unique shape files to copy (msr1..msrN)")
    ap.add_argument("--seed", type=int, default=1234, help="random seed")

    ap.add_argument("--Din-field", choices=["D_in", "2r_in"], default="D_in", help="base diameter field in STL units")
    ap.add_argument("--bin-D-repr", choices=["geomean", "upper", "lower"], default="geomean", help="representative D for each bin")

    ap.add_argument("--molecule-name", default="clumps_01", help="LAMMPS molecule template ID")
    ap.add_argument("--molecule-include", default="molecule_clumps_01.in", help="output include for molecule command (220 templates)")
    ap.add_argument("--pour-include", default="fix_pour_clumps_01.in", help="output include for fix pour line")
    ap.add_argument("--manifest-json", default="selection_manifest.json", help="selection record")

    # pour line parts (사용자 인풋에 맞게 바꿀 수 있게 옵션화)
    ap.add_argument("--pour-fixid", default="pour_clumps1")
    ap.add_argument("--pour-group", default="all")
    ap.add_argument("--pour-ninsert", type=int, default=280)
    ap.add_argument("--pour-type", type=int, default=0)
    ap.add_argument("--pour-seed", type=int, default=4767548)
    ap.add_argument("--pour-region", default="gen_area")
    ap.add_argument("--rigid-fixid", default="make_clumps_1")  # rigid/small fix id

    # optional filters
    ap.add_argument("--L-range", default="", help="e.g. 0.8,1.2")
    ap.add_argument("--e-range", default="", help="e.g. 0.7,0.8")
    ap.add_argument("--f-range", default="", help="e.g. 0.6,0.7")
    ap.add_argument("--sphericity-range", default="", help="e.g. 0.65,0.75")
    ap.add_argument("--roundness-R1-range", default="", help="e.g. 0.7,0.95")
    ap.add_argument("--roundness-R2-range", default="", help="e.g. 0.7,0.95")

    args = ap.parse_args()
    random.seed(int(args.seed))

    db_root = Path(args.db_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # parse ranges
    Lr = parse_optional_range(args.L_range)
    er = parse_optional_range(args.e_range)
    fr = parse_optional_range(args.f_range)
    sr = parse_optional_range(args.sphericity_range)
    r1r = parse_optional_range(args.roundness_R1_range)
    r2r = parse_optional_range(args.roundness_R2_range)

    cases = list_cases(db_root)
    if not cases:
        raise SystemExit(f"No valid cases found under: {db_root}")

    def case_ok(c: Case) -> bool:
        if not within(c.L, Lr): return False
        if not within(c.e, er): return False
        if not within(c.f, fr): return False
        if c.sphericity is not None and not within(c.sphericity, sr): return False
        if c.round_R1 is not None and not within(c.round_R1, r1r): return False
        if c.round_R2 is not None and not within(c.round_R2, r2r): return False
        if args.Din_field == "D_in":
            return c.D_in is not None and c.D_in > 0
        else:
            return c.r_in is not None and c.r_in > 0

    pool = [c for c in cases if case_ok(c)]
    if not pool:
        raise SystemExit("No cases satisfy the given parameter ranges / required fields.")

    n_shapes = int(args.n_shapes)
    if n_shapes <= 0:
        raise SystemExit("--n-shapes must be > 0")
    if len(pool) < n_shapes:
        raise SystemExit(f"Not enough cases in pool ({len(pool)}) for n-shapes={n_shapes}")

    # ---- Pick shapes ONCE (unique) ----
    chosen_shapes = random.sample(pool, n_shapes)

    def base_D(c: Case) -> float:
        if args.Din_field == "D_in":
            return float(c.D_in)
        return float(2.0 * c.r_in)

    # ---- Copy ONLY N files ----
    shapes_manifest = []
    for i, c in enumerate(chosen_shapes, start=1):
        src = c.case_dir / "molecule_mc.data"
        dst = out_dir / f"msr{i}.data"
        shutil.copy2(src, dst)
        shapes_manifest.append({
            "msr": i,
            "case_id": c.case_id,
            "src": str(src),
            "dst": str(dst),
            "base_D": base_D(c),
            "shape_params": {"L": c.L, "e": c.e, "f": c.f},
            "metrics": {"sphericity": c.sphericity, "R1": c.round_R1, "R2": c.round_R2, "D_in": c.D_in, "r_in": c.r_in},
        })

    # ---- Gradation bins ----
    grad = load_gradation_csv(Path(args.gradation))
    bins = gradation_to_bin_fractions(grad)  # sum=1

    def repr_D(dlo, dhi) -> float:
        if args.bin_D_repr == "upper":
            return float(dhi)
        if args.bin_D_repr == "lower":
            return float(dlo)
        return math.sqrt(float(dlo) * float(dhi))

    # ---- Build molecule include (single ID with 220 templates) ----
    mol_name = args.molecule_name
    molecule_pairs: List[str] = []
    molfrac_numbers: List[float] = []
    template_map = []  # for manifest: (bin, msr, scale, weight)

    for bi, (dlo, dhi, frac) in enumerate(bins, start=1):
        Dtarget = repr_D(dlo, dhi)
        w = float(frac) / float(n_shapes)  # BIN fraction distributed equally to N shapes

        for i, c in enumerate(chosen_shapes, start=1):
            scale = float(Dtarget) / float(base_D(c))
            rel = (out_dir / f"msr{i}.data").as_posix()
            molecule_pairs.append(f"{rel}  scale {scale:.8g}")
            molfrac_numbers.append(w)
            template_map.append({
                "template_index_1based": len(molecule_pairs),
                "bin_index": bi,
                "Dlo": float(dlo),
                "Dhi": float(dhi),
                "bin_frac": float(frac),
                "Dtarget": float(Dtarget),
                "msr": i,
                "case_id": c.case_id,
                "scale": float(scale),
                "molfrac": float(w),
            })

    # molecule include content
    prefix1 = f"molecule         {mol_name} "
    prefix_cont = " " * len("molecule         ") + " " * len(mol_name) + " "
    molecule_text = (
        "# Auto-generated by genSample.py\n"
        f"# total templates = {len(molecule_pairs)} (= bins {len(bins)} * n_shapes {n_shapes})\n"
        + wrap_pairs(prefix1, prefix_cont, molecule_pairs, max_per_line=5)
    )
    write_text(Path(args.molecule_include), molecule_text)

    # ---- Build fix pour include (ONE LINE) ----
    # We include a full fix command with wrapped molfrac numbers
    fix_prefix1 = (
        f"fix {args.pour_fixid} {args.pour_group} pour "
        f"{args.pour_ninsert} {args.pour_type} {args.pour_seed} "
        f"region {args.pour_region} mol {mol_name} molfrac "
    )
    fix_prefix_cont = " " * len("fix ")

    # molfrac numbers wrapped
    molfrac_text = wrap_numbers(fix_prefix1, " " * len(fix_prefix1), molfrac_numbers, per_line=10, fmt="{:.8g}")

    # append rigid clause at end of last line (avoid breaking syntax)
    molfrac_lines = molfrac_text.rstrip("\n").splitlines()
    molfrac_lines[-1] = molfrac_lines[-1].rstrip() + f" rigid {args.rigid_fixid}\n"

    pour_text = (
        "# Auto-generated by genSample.py\n"
        f"# molfrac count = {len(molfrac_numbers)} (must match molecule templates)\n"
        + "\n".join(molfrac_lines)
    )
    write_text(Path(args.pour_include), pour_text)

    # ---- Manifest ----
    manifest = {
        "db_root": str(db_root),
        "gradation": str(Path(args.gradation)),
        "out_dir": str(out_dir),
        "n_shapes": n_shapes,
        "seed": int(args.seed),
        "Din_field": args.Din_field,
        "bin_repr": args.bin_D_repr,
        "molecule_name": mol_name,
        "bins": [{"Dlo": a, "Dhi": b, "frac": f} for (a, b, f) in bins],
        "shapes": shapes_manifest,
        "templates": template_map,
        "total_templates": len(molecule_pairs),
        "molfrac_sum": float(sum(molfrac_numbers)),
        "molecule_include": args.molecule_include,
        "pour_include": args.pour_include,
    }
    write_text(Path(args.manifest_json), json.dumps(manifest, indent=2))

    print("[OK] Copied ONLY N shape molecules into:", out_dir)
    print("[OK] Wrote molecule include:", args.molecule_include)
    print("[OK] Wrote fix pour include:", args.pour_include)
    print("[OK] Wrote manifest:", args.manifest_json)
    print("[INFO] total templates =", len(molecule_pairs), " molfrac_sum =", sum(molfrac_numbers), "(should be 1.0)")

if __name__ == "__main__":
    main()