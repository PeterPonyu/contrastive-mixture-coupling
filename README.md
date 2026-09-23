# Seed and schedule sensitivity of mixture coupling in a contrastive autoencoder

## Repository, archive and citation

Study repository: [PeterPonyu/contrastive-mixture-coupling](https://github.com/PeterPonyu/contrastive-mixture-coupling). Versioned archive: [10.5281/zenodo.22914495](https://doi.org/10.5281/zenodo.22914495). Use `CITATION.cff` for release 1.0.0; `release-manifest.json` records distributed-file checksums. The archive and source repository describe this study only.

Author-owned software is MIT licensed. The author's manuscript, figures, generated results and model weights are CC BY 4.0. Source-study data, labels, annotations and other third-party materials retain their original terms; they are not relicensed. Read `LICENSE` and `NOTICE.md` before reusing mixed-content files.

This repository accompanies a study of mixture coupling in a frozen contrastive-autoencoder implementation. It contains paired Setty endpoint results for three model seeds and two coupling weights, recorded batch permutations, mixture parameters and responsibilities, historical summaries for four datasets, model source snapshots, figures and manuscript sources. The recorded Setty protocol shares the initial state and a 180-epoch warmup before the two coupling continuations; its six endpoints represent 10,560 unique historical scientific updates. Running the checks below adds zero neural updates.

The reproducible scope is CPU analysis of saved results, synthetic invariant probes of the frozen model, optional archived-checkpoint inspection, and figure generation. This package does not provide a complete neural-training reproduction.

## CPU verification from the included inputs

Run commands from the repository root. Python 3.11 or later is needed for the file-integrity check. The saved-result audit uses NumPy and SciPy; the invariant probe additionally imports PyTorch and scikit-learn. The tested environment used Python 3.13, NumPy 2.2.6, SciPy 1.16.3, scikit-learn 1.8.0 and PyTorch 2.12.0. A CPU-capable PyTorch installation is sufficient; no command installs dependencies.

```bash
export CUDA_VISIBLE_DEVICES=""
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1
python3 scripts/verify_public_manifest.py
python3 scripts/verify_saved_results.py
python3 scripts/probe_moco_invariants.py
```

All required inputs for these commands are included on GitHub; no `.pt` checkpoint is loaded. Output directories under `validation/recomputed/` are created automatically. The input check validates public file hashes. The result audit validates the saved batch schedules and refit subsets, recorded pairing identities and per-endpoint JSON records; reconstructs training-mixture and observer probabilities and occupancies from six NPZ files; and recomputes paired contrasts and leave-one-seed-out means. It checks recorded state digests for consistency but does not load the full warmup states.

The synthetic probe verifies the exact SHA-256 identities of the bundled model and base class before importing them. It checks query/key initialization differences, one-forward momentum and queue updates, batch-normalization calls and the scalar contrastive-weight contribution. It performs no optimizer step or mixture fit. An optional `--source-root` must contain the same two source files with matching hashes.

Fresh reports are `validation/recomputed/input-integrity.json`, `validation/recomputed/robustness-audit.json` and `validation/recomputed/invariant-probe.json`. The archived `validation/robustness-audit.json` and `evidence/invariant-probe.json` remain available as reference results and are not overwritten.

## Optional checkpoint metadata audit

The large `.pt` files are supplied through this repository's corresponding Zenodo supplementary checkpoint archive, separately from GitHub's small scientific inputs. Extract that archive so its `evidence/run-archive/` directory contains `seed0/`, `seed1/` and `seed2/`, each with `initial.pt`, `warmup.pt`, `w0.pt` and `w1.pt`. If extracted at the repository root, run:

```bash
python3 scripts/verify_public_manifest.py --checkpoints
python3 scripts/audit_saved_states.py
```

Alternatively, pass the extracted directory to `scripts/audit_saved_states.py --checkpoint-root /path/to/extracted/evidence/run-archive`. The script reports missing archive files explicitly. It verifies all 12 file hashes and byte sizes before loading them on CPU, verifies the registered warmup-model digests against the recorded pairing results, and compares queue, normalization and query/key metadata with the archived report. These historical checkpoints contain serialized RNG state, so only the matching published archive should be loaded. This audit neither imports the model source nor makes optimizer updates. It writes `validation/recomputed/checkpoint-metadata.json` and preserves `evidence/checkpoint-metadata.json`.

## Figures and manuscript build

All figure inputs are local JSON/CSV files enumerated in `figure-build.json`; checkpoints are unnecessary. R requires ggplot2, gridExtra, jsonlite, grid and Cairo graphics. The schematic renderer requires a Python interpreter with PyGObject, Rsvg 2.0 and the system Cairo/librsvg libraries. Arial must be installed for the supplied typography. The tested R environment was R 4.3.3, ggplot2 4.0.3, gridExtra 2.3 and jsonlite 2.0.0.

```bash
# Choose the Python interpreter that can import gi and Rsvg.
# On the tested Linux system this was /usr/bin/python3.
SVG_PYTHON=/usr/bin/python3 Rscript scripts/build_figures.R
bash build.sh
```

`SVG_PYTHON` defaults to `python3` if omitted. The R entry point regenerates five data figures as PDF/PNG, the vector architecture PDF and `figures/plot_manifest.json`. Each of six UMAP panels uses its supplied 450-point coordinates and labels; UMAP and the encoder are not rerun. Historical non-finite JSON entries are treated as missing only during plotting, without changing the saved inputs.

The manuscript build requires XeLaTeX, BibTeX, Arial, TeX Gyre Termes Math and the packages named in `standalone-assets/preamble.tex`. `bash build.sh` compiles the local manuscript and copies `build_independent/paper.pdf` to `paper.pdf`. Figure regeneration and manuscript compilation are separate commands.

## Contents and provenance

- `evidence/run-archive/`: complete result/protocol JSON, six endpoint JSON/NPZ pairs and three batch-permutation NPY files. Optional checkpoints use the same directory structure.
- `evidence/new_results.json` and `evidence/new_protocol.json`: the paired result report and protocol used by the figures and verifier.
- Historical sweep JSON files and `evidence/umap/`: the saved figure inputs.
- `evidence/frozen-source/`: unchanged `dpmm_contrastive.py`, its base class and their existing MIT copyright notice. This source snapshot is sufficient for the synthetic probes.
- `evidence/invariant-probe.json`, `evidence/checkpoint-metadata.json` and `validation/robustness-audit.json`: preserved numerical reference reports.
- `scripts/`: independent result checks, optional checkpoint inspection and figure generation.
- `public-manifest.json`: public input/tool hashes and a separate `checkpoint_files` inventory for the 12 Zenodo files. Descriptive path normalization is recorded separately from historical scientific and producer hashes; those identities are not replaced.
- `paper.tex`, `standalone-assets/`, `extra.bib`, `build.sh`, `figures/` and `paper.pdf`: manuscript sources, build inputs and supplied article.

## Scientific limits

This is a coupling comparison within one frozen contrastive implementation. Both arms inherit the same contrastive path, including unequal query/key initialization and additional query batch-normalization calls. Setting `moco_weight=0` suppresses its scalar loss contribution but does not skip forward queue or normalization updates. The paired coupling comparison is therefore not a contrastive-on/off causal test or a test of a corrected implementation.

Three model seeds and paired computational continuations are not independent biological samples. The historical four-dataset and 200/400-epoch summaries are schedule-specific; the newly matched pairing is Setty-only. Effective occupancy is not a validated biological cell-type count, and stored UMAP labels are not a new annotation analysis.

Raw expression/preprocessing inputs, PCA reference inputs, branch targets and a complete neural-training pipeline are absent. The saved-result audit independently differences the published scores but cannot recompute edge or branch metrics from raw data. Neither the small-input checks nor the optional checkpoints reproduce neural training, an expanded seed study, corrected-model training or biological replication.
