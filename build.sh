#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build_independent
export TEXINPUTS="build_independent:.:${TEXINPUTS:-}"
xelatex -recorder -interaction=nonstopmode -halt-on-error -output-directory=build_independent paper.tex >/dev/null
(cd build_independent && BIBINPUTS="..:${BIBINPUTS:-}" bibtex paper >/dev/null)
xelatex -recorder -interaction=nonstopmode -halt-on-error -output-directory=build_independent paper.tex >/dev/null
xelatex -recorder -interaction=nonstopmode -halt-on-error -output-directory=build_independent paper.tex >/dev/null
cp build_independent/paper.pdf paper.pdf
