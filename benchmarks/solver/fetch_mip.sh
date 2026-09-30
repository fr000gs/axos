#!/bin/sh
# Downloads small MIP instances (plain MPS, as shipped in the HiGHS repository) into
# build/mip. Instances are not committed. Reference values: benchmarks/solver/RESULTS.md.
set -e
DIR=${1:-build/mip}
BASE=https://raw.githubusercontent.com/ERGO-Code/HiGHS/master/check/instances
mkdir -p "$DIR"
for f in bell5 dcmulti egout flugpl gams10am gas11 gesa2 gt2 lseu p01 p0548 rgn; do
    [ -s "$DIR/$f.mps" ] || curl -sfL "$BASE/$f.mps" -o "$DIR/$f.mps" || echo "missing: $f"
done
ls "$DIR" | wc -l
