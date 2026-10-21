#!/bin/bash
# End-to-end test of ../linearity_check_cell.py with real cryoDRGN backproject_voxel on synthetic data.
# Needs a python with cryodrgn (4.3.1 tested) and the `cryodrgn` CLI on PATH. About 8 min on 4 CPU cores.
# Usage: PY=/path/to/python bash test_lin.sh [WORKDIR]
set -euo pipefail
T=$(cd "$(dirname "$0")" && pwd)
CELL=$T/../linearity_check_cell.py
PY=${PY:-python3}
ROOT=${1:-$(mktemp -d)}/runs
mkdir -p $ROOT $ROOT/../work
cd $ROOT/../work

bp () {  # bp OUTDIR POSES IND
  cryodrgn backproject_voxel $ROOT/$SC/stack/particles.txt --poses "$2" --ctf $ROOT/$SC/proj/ctf.pkl \
    --ind "$3" -o "$1" -b 8 --reg-weight 1.0 --uninvert-data --lazy > /dev/null 2>&1
}

run () {  # run DATA TAG M7_POSES M6_POSES IND_KIND(swap|conc) CELL_J486
  SC=$1; P=$ROOT/$SC/proj; Q=$P/pseudo_c2_check
  BPD=$ROOT/$SC/drive_$2
  mkdir -p $BPD
  IX=$Q/bp_J486/c2swap_J486.pkl
  [ "$5" = conc ] && IX=$ROOT/$SC/conc_ind.pkl
  [ -f $BPD/concordant_eq/backproject.mrc ] || bp $BPD/concordant_eq $P/pose.pkl $ROOT/$SC/conc_ind.pkl
  [ -f $BPD/c2swap_J487_eq/backproject.mrc ] || bp $BPD/c2swap_J487_eq "$3" $IX
  [ -f $BPD/c2swap_J486_eq/backproject.mrc ] || bp $BPD/c2swap_J486_eq "$4" $IX
  PARAMS=$(cat <<EOP
{"POSE_J487": "$P/pose.pkl", "POSE_J486": "$6", "IND": "$IX",
 "SUMMARY": "$Q/pc2_summary.json", "CTF": "$P/ctf.pkl", "PARTICLES": "$ROOT/$SC/stack/particles.txt",
 "BP_DIR": "$BPD", "V_NAME": "concordant_eq", "M7_NAME": "c2swap_J487_eq", "M6_NAME": "c2swap_J486_eq",
 "OUT_DIR": "$BPD/linearity", "LOCAL_DIR": "$ROOT/$SC/local_$2", "UNINVERT_DATA": true, "LOWPASS_A": 8.0,
 "MASK_KDA": 60.0, "BATCH": 8}
EOP
)
  $PY -I $T/run_cell.py $CELL "$PARAMS" $ROOT/${SC}_$2.json > $ROOT/${SC}_$2.log 2>&1
  echo "=== $SC/$2"; sed -n '/^CHECKS/,$p;/STOPPED/p' $ROOT/${SC}_$2.log
}

for sc in consistent gap; do [ -f $ROOT/$sc/truth.json ] || $PY -I $T/gen_lin.py $ROOT/$sc $sc; done
C=$ROOT/consistent/proj; G=$ROOT/gap/proj; J6=pseudo_c2_check/pose_J486_in_J487_order.pkl
[ -f $ROOT/consistent/pose_J486_frame.pkl ] || $PY -I $T/make_frame.py $ROOT/consistent $ROOT/consistent/pose_J486_frame.pkl 2.0
[ -f $ROOT/consistent/pose_J486_blur.pkl ] || $PY -I $T/make_frame.py $ROOT/consistent $ROOT/consistent/pose_J486_blur.pkl 0.0 1.5
run consistent ok        $C/pose.pkl $C/$J6 swap $C/$J6
echo "=== resume:"; run consistent ok $C/pose.pkl $C/$J6 swap $C/$J6 > /dev/null; grep -c "already made from these inputs" $ROOT/consistent_ok.log || true
run consistent mislabel7 $C/$J6 $C/$J6 swap $C/$J6
run consistent mislabel6 $C/pose.pkl $C/pose.pkl swap $C/$J6
run consistent wrongind  $C/pose.pkl $C/$J6 conc $C/$J6
run consistent frame     $C/pose.pkl $ROOT/consistent/pose_J486_frame.pkl swap $ROOT/consistent/pose_J486_frame.pkl
run consistent blur      $C/pose.pkl $ROOT/consistent/pose_J486_blur.pkl swap $ROOT/consistent/pose_J486_blur.pkl
run gap ok               $G/pose.pkl $G/$J6 swap $G/$J6
$PY -I $T/check_lin.py $ROOT
