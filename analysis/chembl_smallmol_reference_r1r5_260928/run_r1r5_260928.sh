#!/usr/bin/env bash
# Serial recomputation of R1-R5 reference rates for the 260928 ChEMBL
# small-molecule reference set, plus a cross-check on the old reference to
# confirm the script reproduces the previously reported 84.72%.
#
# Run detached:
#   setsid nohup bash analysis/chembl_smallmol_reference_r1r5_260928/run_r1r5_260928.sh \
#     > analysis/chembl_smallmol_reference_r1r5_260928/wrapper.log 2>&1 < /dev/null &
set -u

cd /home/md666/data/disk6/DiffB110 || exit 1
PY=/home/md666/miniforge3/envs/dif/bin/python
D=analysis/chembl_smallmol_reference_r1r5_260928

echo "START $(date)" > "$D/run_status.txt"

echo "[$(date)] new reference R1-R5 ..."
"$PY" "$D/compute_reference_r1r5.py" \
  --input data/chembl_smallmol_reference/reference_smallmol.csv.gz \
  --output-dir "$D" --label new_smallmol_reference --workers 18 \
  > "$D/new_ref_r1r5.log" 2>&1
echo "NEW_REF_EXIT=$?" >> "$D/run_status.txt"

echo "[$(date)] old reference cross-check ..."
"$PY" "$D/compute_reference_r1r5.py" \
  --input data/chembl/chembl_36_molecules.csv.gz \
  --output-dir "$D" --label old_reference_crosscheck --workers 18 \
  > "$D/old_ref_crosscheck.log" 2>&1
echo "OLD_REF_EXIT=$?" >> "$D/run_status.txt"

echo "ALL_DONE $(date)" >> "$D/run_status.txt"
echo "[$(date)] all done"
