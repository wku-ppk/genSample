# genSample

`genSample.py` selects a reproducible sample of clump shapes from a
[`clump-DB`](https://github.com/wku-ppk/clump-DB) dataset and turns a cumulative
particle-size gradation into LAMMPS molecule-template and `fix pour` include
files.

The generator:

1. finds valid clump cases in a shape database;
2. optionally filters them by shape and roundness metrics;
3. randomly selects `N` unique shapes once;
4. copies only those selected LAMMPS molecule files to a working directory;
5. converts the cumulative gradation into size-bin probabilities;
6. creates one LAMMPS molecule collection containing every
   size-bin/shape combination; and
7. creates a matching `fix pour` command and a JSON manifest.

> [!IMPORTANT]
> The shape database itself is not included in this repository. The committed
> `MC_55867233`, `POUR_55867233`, and `MB_55867233.json` files are example
> outputs. They refer to `ms_data/msr*.data`, which is not committed, so they
> are not a self-contained runnable LAMMPS case.

## Repository contents

| Path | Purpose |
| --- | --- |
| `genSample.py` | Command-line generator. |
| `grad.csv` | Example cumulative gradation, with passing values expressed as percentages. |
| `MC_55867233` | Example generated LAMMPS `molecule` include. |
| `POUR_55867233` | Matching example generated LAMMPS `fix pour` include. |
| `MB_55867233.json` | Matching example selection/output manifest. |
| `.gitignore` | Ignores `ms_data/` and untracked `grad.csv` files (the repository's existing sample `grad.csv` remains tracked). |

## Requirements

### To run the generator

- Python 3.7 or newer (`dataclasses` is used).
- No third-party Python packages are required; `genSample.py` uses only the
  standard library.
- A populated clump shape database. The expected layout and metadata match the
  output of [`wku-ppk/clump-DB`](https://github.com/wku-ppk/clump-DB).

### To use the generated files

- A compatible [LAMMPS](https://www.lammps.org/) build.
- The LAMMPS `GRANULAR` package for
  [`fix pour`](https://docs.lammps.org/fix_pour.html).
- The LAMMPS `RIGID` package for
  [`fix rigid/small`](https://docs.lammps.org/fix_rigid.html) when the generated
  `rigid` clause is used.

The script does not install or run LAMMPS and does not validate the generated
includes against a particular LAMMPS release.

## Expected shape-database layout

`--db-root` must contain one immediate subdirectory per case:

```text
dataset/shapes/
├── <case-id-1>/
│   ├── meta.json
│   └── molecule_mc.data
├── <case-id-2>/
│   ├── meta.json
│   └── molecule_mc.data
└── ...
```

A directory is considered a candidate only when both `meta.json` and
`molecule_mc.data` exist. Directories missing either file are skipped.

The practical minimum metadata structure is:

```json
{
  "case_id": "6f6baf9491ae",
  "shape_params": {
    "L": 1.0,
    "e": 0.75,
    "f": 0.65
  },
  "metrics": {
    "sphericity_riley1941_style": 0.6581,
    "wadell": {
      "R1": 0.5079,
      "R2": 0.4787,
      "r_in": 0.4164,
      "D_in": 0.8329
    }
  }
}
```

Field behavior:

- `case_id` is optional; the case-directory name is used when it is absent.
- `shape_params.L` is the longest-axis measure used for filtering.
- `shape_params.e` may instead be named `I_over_L`.
- `shape_params.f` may instead be named `S_over_I`.
- `metrics.sphericity_riley1941_style`, `wadell.R1`, `wadell.R2`, and
  `wadell.r_in` are optional.
- `metrics.wadell.D_in` must exist and be greater than zero. It is the base
  diameter used to scale every selected shape.

Malformed JSON or nonnumeric values are not silently skipped; they stop the
run with an exception.

## Gradation input

Pass a two-column CSV through `--gradation`:

```csv
D,passing
0.0001,0
0.0002,5
0.0003,10
0.0004,20
0.0005,30
```

The first column is diameter `D`. The second is cumulative passing. A header is
optional. The parser accepts either fractions or percentages:

- if the maximum passing value is greater than `1.5`, all passing values are
  divided by `100`;
- otherwise, the values are treated as fractions;
- values are then clamped to `[0, 1]`;
- rows are sorted by increasing `D`; and
- passing must be non-decreasing after sorting.

Use positive diameters. At least two usable points and at least one positive
increase in passing are required. Zero-increase intervals are omitted.

The input does not have to begin at passing `0` or end at `1`. The differences
between consecutive points are normalized to sum to one.

> [!NOTE]
> The resulting fractions become `fix pour molfrac` selection probabilities.
> They are therefore **molecule-count probabilities**. If the input gradation
> represents mass or volume fractions, the realized mass/volume distribution
> will generally differ unless an appropriate conversion is performed before
> running this script.

## Quick start

Clone this repository and prepare a populated `clump-DB` dataset. A convenient
directory layout is:

```text
work/
├── clump-DB/
│   └── dataset/shapes/<case-id>/...
└── genSample/
    ├── genSample.py
    └── grad.csv
```

From the `genSample` directory, run:

```bash
python3 genSample.py \
  --db-root ../clump-DB/dataset/shapes \
  --gradation grad.csv \
  --out-dir ms_data \
  --n-shapes 20 \
  --seed 1234
```

With the default output names, a successful run prints paths similar to:

```text
[OK] Copied ONLY N shape molecules into: ms_data
[OK] N = 20  (msr1..msr20.data)
[OK] Bins = 11  Templates = 220
[OK] Molecule include : molecule_table_<hash>.in
[OK] Pour include     : fix_pour_clumps_01_<hash>.in
[OK] Manifest JSON    : selection_manifest_<hash>.json
[OK] hash_id          : <hash>
[OK] molfrac_sum      : 1.0 (should be 1.0)
```

Output filenames vary with the selected cases and relevant options.

## Example matching the committed outputs

The following command is recorded in the header of `genSample.py` and matches
the options represented by the committed example artifacts:

```bash
python3 genSample.py \
  --db-root ../clump-DB/dataset/shapes \
  --gradation grad.csv \
  --out-dir ms_data \
  --n-shapes 20 \
  --seed 1234 \
  --molecule-name clumps_01 \
  --molecule-out MC \
  --pour-out POUR \
  --manifest-base MB \
  --pour-fixid pour_clumps1 \
  --pour-ninsert 280 \
  --pour-nsteps 0 \
  --pour-seed 4767548 \
  --pour-region gen_area \
  --rigid-fix make_clumps_1 \
  --L-range 0.9,1.1 \
  --e-range 0.7,0.8 \
  --f-range 0.6,0.7 \
  --sphericity-range 0.5,0.9 \
  --roundness-R1-range 0.4,0.6
```

For the database state used by the author, this produced hash `55867233` and:

```text
ms_data/msr1.data ... ms_data/msr20.data
MC_55867233
POUR_55867233
MB_55867233.json
```

The same command can produce a different hash when the available database
cases or their enumeration order differ.

## Important `fix pour` option-name compatibility note

Two current command-line option names/help strings do not match the actual
LAMMPS argument positions emitted by the script:

| Script option | Position in generated command | Actual LAMMPS meaning |
| --- | --- | --- |
| `--pour-ninsert` | first value after `pour` | `N`: total number of molecules to insert, not the number inserted per step |
| `--pour-nsteps` | second value after `pour` | `type`: atom-type offset for inserted molecules, not a number of steps |

For example:

```text
fix pour_clumps1 all pour 280 0 4767548 ...
```

means: insert 280 molecules in total, use atom-type offset `0`, and use random
seed `4767548`. The default `0` is appropriate when the atom types stored in the
molecule files should be preserved. See the official
[`fix pour` syntax](https://docs.lammps.org/fix_pour.html) before changing these
two values.

## Selection filters

All range endpoints are inclusive. Every filter accepts:

| Value | Meaning |
| --- | --- |
| empty string | no constraint |
| `a,b` | `a <= value <= b` |
| `a,` | `value >= a` |
| `,b` | `value <= b` |
| `x` | exact floating-point match with `x` |

Examples:

```bash
--L-range 0.9,1.1
--e-range 0.7,
--f-range ,0.8
--sphericity-range 0.65,0.75
```

Available filters are:

- `--L-range`
- `--e-range`
- `--f-range`
- `--sphericity-range`
- `--roundness-R1-range`
- `--roundness-R2-range`

`L`, `e`, and `f` are always read as floating-point values. When one of their
filters is active, a missing/nonfinite value will not pass it.

The optional sphericity, `R1`, and `R2` metrics behave differently: a missing
metric is accepted even when its corresponding filter is active. If every
selected case must contain and satisfy an optional metric, validate the
database first or change this behavior in `case_ok()`.

## How the generator builds the mixture

For consecutive cumulative-gradation points `(D0, P0)` and `(D1, P1)`, the
unnormalized bin fraction is:

```text
fraction = P1 - P0
```

Positive fractions are normalized so all retained bins sum to one. The
representative diameter of each bin is its geometric mean:

```text
Dtarget = sqrt(Dlo * Dhi)
```

The script samples `N = --n-shapes` unique cases once, without replacement.
Those same shapes are reused in every size bin. For selected shape `j`:

```text
scale(bin, j) = Dtarget(bin) / D_in(j)
molfrac(bin, j) = normalized_bin_fraction(bin) / N
```

Templates are emitted in bin-major order:

```text
bin 1: shape 1, shape 2, ... shape N
bin 2: shape 1, shape 2, ... shape N
...
```

This guarantees:

```text
template_count = number_of_nonzero_bins * N
molfrac_count  = template_count
sum(molfrac)   = 1.0, within floating-point tolerance
```

LAMMPS applies the generated `scale` keyword when reading each molecule
template. See the official
[`molecule` command documentation](https://docs.lammps.org/molecule.html) for
the attributes affected by scaling.

## Outputs

### 1. Selected molecule files

The selected source files are copied verbatim to:

```text
<out-dir>/msr1.data
<out-dir>/msr2.data
...
<out-dir>/msrN.data
```

`msr` numbering records selection order, not the original case ID. The manifest
maps every `msr` number back to its source case.

### 2. Molecule include

The molecule include contains one LAMMPS `molecule` command with
`number_of_bins * N` templates. The same copied `msr*.data` file is referenced
once per size bin with a different scale factor.

Example structure:

```lammps
molecule clumps_01 ms_data/msr1.data scale <scale-bin1-shape1> &
                   ms_data/msr2.data scale <scale-bin1-shape2> &
                   ...
```

The real file uses LAMMPS `&` line continuations and includes a detailed comment
header describing filters, gradation bins, and selected shapes.

### 3. Pour include

The pour include contains one matching command:

```lammps
fix <pour-fixid> all pour <N> <type-offset> <seed> region <region-id> &
  mol <molecule-id> molfrac <one-value-per-template> &
  rigid <rigid-fix-id>
```

The `molfrac` values are ordered exactly like the templates in the molecule
include. Do not combine includes from different hashes.

### 4. Manifest

The JSON manifest records:

- input paths and seeds;
- active filters;
- normalized gradation points and derived bins;
- selected case IDs, source/destination paths, and shape metrics;
- template and `molfrac` counts;
- `molfrac_sum`;
- generated output filenames;
- emitted `fix pour` settings; and
- a template-index mapping to bin number, `msr` file, and `molfrac`.

The current `template_map` does not store the per-template scale, although it is
present in the molecule include and can be recomputed from each bin's
`Dtarget_geomean` and each shape's `D_in`.

### Hash-based names

The generator computes an eight-character SHA-1-derived identifier from the
parsed bins, selected case IDs, selection/filter options, and pour settings.
The hash is inserted before an output extension:

```text
molecule_table.in -> molecule_table_<hash>.in
MC                -> MC_<hash>
```

This identifier is a run/selection fingerprint, not a content-integrity hash.
It does not hash the contents of `meta.json` or `molecule_mc.data`. If a source
file changes without changing its case ID or the relevant options, the output
hash may remain unchanged.

## Command-line reference

Run `python3 genSample.py --help` for the parser's built-in help. The table below
documents the effective behavior of the current code.

### Inputs and selection

| Option | Default | Meaning |
| --- | --- | --- |
| `--db-root` | `dataset/shapes` | Root containing immediate `<case-id>` directories. |
| `--gradation` | required | Two-column cumulative `D,passing` CSV. |
| `--out-dir` | `ms_data` | Destination for copied `msr1..msrN.data` files. |
| `--n-shapes` | `20` | Number of unique cases sampled without replacement; must be greater than zero. |
| `--seed` | `1234` | Python random seed used only for shape selection. |
| `--L-range` | empty | Inclusive `L` filter. |
| `--e-range` | empty | Inclusive `e`/`I_over_L` filter. |
| `--f-range` | empty | Inclusive `f`/`S_over_I` filter. |
| `--sphericity-range` | empty | Inclusive sphericity filter when the metric exists. |
| `--roundness-R1-range` | empty | Inclusive Wadell `R1` filter when the metric exists. |
| `--roundness-R2-range` | empty | Inclusive Wadell `R2` filter when the metric exists. |

### Output and LAMMPS settings

| Option | Default | Meaning |
| --- | --- | --- |
| `--molecule-name` | `clumps_01` | LAMMPS molecule-template ID. |
| `--molecule-out` | `molecule_table.in` | Base filename for the hash-suffixed molecule include. |
| `--pour-out` | `fix_pour_clumps_01.in` | Base filename for the hash-suffixed pour include. |
| `--manifest-base` | `selection_manifest` | Base filename for the hash-suffixed JSON manifest. |
| `--pour-fixid` | `pour_clumps1` | ID of the generated LAMMPS `fix pour`. |
| `--pour-ninsert` | `280` | Emitted as LAMMPS `N`: total molecules to insert. |
| `--pour-nsteps` | `0` | Emitted as LAMMPS `type`: atom-type offset. The option name is misleading. |
| `--pour-seed` | `4767548` | Positive random seed used by LAMMPS `fix pour`; independent of `--seed`. |
| `--pour-region` | `gen_area` | Existing LAMMPS insertion-region ID. |
| `--rigid-fix` | `make_clumps_1` | Existing LAMMPS `fix rigid/small` ID. |

The script does not validate LAMMPS identifier syntax or quote paths. Use valid
LAMMPS IDs and avoid whitespace in generated paths.

## Using the files in LAMMPS

A surrounding LAMMPS input script must define the simulation box, compatible
atom types, insertion region, gravity, and rigid-body fix. A minimal ordering
outline is:

```lammps
# Define units, atom style, simulation box, material/contact settings, etc.

region gen_area block ... side in

include MC_<hash>

fix make_clumps_1 all rigid/small molecule mol clumps_01

# Define gravity as required by fix pour.
fix gravity_for_pour all gravity ...

include POUR_<hash>

run ...
```

Replace the placeholders and filenames with the values from the manifest. The
generated `--molecule-name`, `--pour-region`, and `--rigid-fix` values must
match the IDs in the surrounding input script.

LAMMPS imposes additional `fix pour` requirements. In particular, the
insertion region must be a static block or, in 3D, a z-axis cylinder with
`side in`; gravity must point in `-z` for 3D or `-y` for 2D. Consult the
[`fix pour` documentation](https://docs.lammps.org/fix_pour.html) for the
version of LAMMPS you run.

All paths in the generated includes are relative paths. Run LAMMPS from a
directory where `<out-dir>/msr*.data` resolves correctly, or edit/regenerate the
include paths.

## Reproducibility and units

- `--seed` controls shape selection; `--pour-seed` is written into the LAMMPS
  command and controls pouring independently.
- A repeat run is deterministic only while candidate order, database contents,
  Python behavior, and arguments remain unchanged. Candidate directories are
  not sorted before sampling, so the same seed is not a strict cross-filesystem
  reproducibility guarantee.
- The script performs no unit conversion. Gradation diameters and metadata
  `D_in` values must use the same length basis. The resulting LAMMPS files must
  also be consistent with the surrounding simulation's unit style.

Preserve the manifest together with both include files and the selected
`msr*.data` files to retain a complete record of one generated mixture.

## Current limitations and cautions

- The output directory is created if necessary but is **not cleaned**. Reusing
  an output directory after a run with a larger `--n-shapes` value can leave
  stale `msr*.data` files. Use an empty or run-specific directory.
- Existing files with the same generated names are overwritten.
- Molecule and manifest outputs are written relative to the current working
  directory, while selected molecule files are written under `--out-dir`.
- Parent directories supplied in output *base-name* arguments are not reliably
  preserved by the current hash-name construction. Use simple filenames and
  choose the working directory explicitly.
- Missing optional sphericity/roundness metrics pass their filters.
- Gradation fractions are used as number probabilities, not mass fractions.
- The eight-character hash does not verify source-file contents.
- No automated test suite is included in the repository.

## Troubleshooting

### `No valid cases found under: ...`

Check `--db-root`. Each immediate child case must contain both `meta.json` and
`molecule_mc.data`.

### `No cases satisfy the given parameter ranges / required fields`

Relax the filters and confirm every intended case has a positive
`metrics.wadell.D_in`.

### `Not enough cases in pool (...) for n-shapes=...`

Reduce `--n-shapes`, relax filters, or add more valid cases to the database.

### `Passing curve must be non-decreasing`

Sort/repair the cumulative passing values. The script sorts by `D`, but the
passing sequence at those sorted diameters must not decrease.

### LAMMPS cannot open `ms_data/msr*.data`

Run LAMMPS from the generation working directory, copy the complete output set,
or regenerate using an `--out-dir` path that resolves from the LAMMPS run
directory.

### LAMMPS reports a molecule-template/`molfrac` mismatch

Use molecule and pour includes with the same hash. The number and order of
`molfrac` entries must match the templates in the molecule collection.

## License

This repository currently contains no `LICENSE` file. Do not assume permission
to copy, modify, or redistribute the code beyond what applicable law permits;
contact the repository owner or add an explicit license before redistribution.
