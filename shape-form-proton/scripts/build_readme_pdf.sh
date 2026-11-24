#!/usr/bin/env bash
# Rebuild README.pdf of shape-form-proton with Pandoc and XeLaTeX.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="$(mktemp -d /tmp/shape-readme-pdf.XXXXXX)"
trap 'rm -rf -- "$build_dir"' EXIT
cat > "$build_dir/header.tex" <<'TEX'
\usepackage{etoolbox}
\usepackage{fvextra}
\DefineVerbatimEnvironment{Highlighting}{Verbatim}{breaklines,breakanywhere,fontsize=\footnotesize,commandchars=\\\{\}}
\DefineVerbatimEnvironment{verbatim}{Verbatim}{breaklines,breakanywhere,fontsize=\footnotesize}
\AtBeginEnvironment{longtable}{\footnotesize}
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.12}
\setlength{\emergencystretch}{3em}
\usepackage{newunicodechar}
\newfontfamily\fallbackfont{TeX Gyre Pagella}
\newunicodechar{✓}{{\fallbackfont ✓}}
\newunicodechar{ℓ}{{\fallbackfont ℓ}}
\newunicodechar{≳}{{\fallbackfont ≳}}
TEX
pandoc "$project_root/README.md" \
  --from=markdown \
  --standalone \
  --pdf-engine=xelatex \
  --lua-filter="$project_root/scripts/readme-pdf.lua" \
  --include-in-header="$build_dir/header.tex" \
  --metadata=lang:hu-HU \
  --variable=papersize:a4 \
  --variable=geometry:margin=14mm \
  --variable=fontsize:10pt \
  --variable=mainfont:'DejaVu Serif' \
  --variable=sansfont:'DejaVu Sans' \
  --variable=monofont:'DejaVu Sans Mono' \
  --variable=colorlinks:true \
  --output="$build_dir/README.pdf"
mv -- "$build_dir/README.pdf" "$project_root/README.pdf"
printf 'Elkészült: %s/README.pdf\n' "$project_root"
