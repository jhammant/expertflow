#!/bin/bash
#
# Compile ExpertFlow paper to PDF
#
# Requirements:
#   - pdflatex (from TeX Live or MacTeX)
#   - On macOS: brew install --cask mactex-no-gui
#   - On Ubuntu/Debian: sudo apt-get install texlive-latex-extra texlive-fonts-recommended
#
# Usage: ./compile.sh

set -e

# Check for pdflatex
if ! command -v pdflatex &> /dev/null; then
    echo "Error: pdflatex not found!"
    echo ""
    echo "Please install LaTeX:"
    echo "  macOS:   brew install --cask mactex-no-gui"
    echo "  Ubuntu:  sudo apt-get install texlive-latex-extra texlive-fonts-recommended"
    echo ""
    exit 1
fi

PAPER="expertflow"
DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$DIR"

echo "==> Compiling $PAPER.tex..."

# First pass
pdflatex -interaction=nonstopmode "$PAPER.tex" > /dev/null 2>&1 || {
    echo "Error on first pass. Showing last 20 lines of output:"
    pdflatex -interaction=nonstopmode "$PAPER.tex" | tail -20
    exit 1
}

# Second pass (for references)
pdflatex -interaction=nonstopmode "$PAPER.tex" > /dev/null 2>&1

# Third pass (for cross-references)
pdflatex -interaction=nonstopmode "$PAPER.tex" > /dev/null 2>&1

# Clean up auxiliary files
rm -f "$PAPER.aux" "$PAPER.log" "$PAPER.out"

echo "==> Success! Generated $PAPER.pdf"
echo "    $(wc -c < "$PAPER.pdf" | awk '{printf "%.1f KB", $1/1024}')"

# Open PDF if on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "==> Opening PDF..."
    open "$PAPER.pdf"
fi
