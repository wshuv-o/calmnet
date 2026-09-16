#!/usr/bin/env bash
# Regenerate every figure in the paper. Run from anywhere.
cd "$(dirname "$0")/../src"
python render_drawio.py ../paper/fig_arch.drawio ../results/fig_arch.pdf
python figures.py
