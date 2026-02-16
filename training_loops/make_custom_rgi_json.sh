#!/usr/bin/env bash
set -euo pipefail

IDS_CSV="${1:-Model_ID.csv}"
CARD_JSON="${2:-card.json}"
OUT_JSON="${3:-custom_card.json}"

tmp_ids_obj="$(mktemp)"
tmp_ids_keys="$(mktemp)"
tmp_missing="$(mktemp)"

echo "[INFO] Input IDs CSV:  ${IDS_CSV}"
echo "[INFO] Input CARD JSON: ${CARD_JSON}"
echo "[INFO] Output JSON:     ${OUT_JSON}"
echo

# ----------------------------
# 1) Extract Model_ID -> int, build an "ID set" object for jq membership tests
#    - Handles floats like 5800.0
#    - Drops header and blanks
# ----------------------------
awk -F',' '
  NR==1 { next }                 # skip header
  {
    gsub(/^[ \t]+|[ \t]+$/, "", $1)
    if ($1 == "") next
    id = int($1 + 0)
    print id
  }
' "$IDS_CSV" \
| jq -Rn '
    reduce inputs as $id ({}; . + {($id|tostring): 1})
  ' > "$tmp_ids_obj"

jq -r 'keys[]' "$tmp_ids_obj" > "$tmp_ids_keys"

echo "[STATS] IDs file rows (excluding header): $(($(wc -l < "$IDS_CSV") - 1))"
echo "[STATS] Unique Model_ID (int) extracted:  $(wc -l < "$tmp_ids_keys")"
echo

# ----------------------------
# 2) CARD JSON stats pre-filter
#    Count numeric model keys vs metadata keys
# ----------------------------
card_numeric_keys="$(jq -r 'keys[] | select(test("^[0-9]+$"))' "$CARD_JSON" | wc -l | tr -d ' ')"
card_meta_keys="$(jq -r 'keys[] | select(test("^[0-9]+$")|not)' "$CARD_JSON" | wc -l | tr -d ' ')"

echo "[STATS] CARD numeric model entries (top-level numeric keys): ${card_numeric_keys}"
echo "[STATS] CARD metadata/non-numeric keys:                    ${card_meta_keys}"
echo

# ----------------------------
# 3) Compute overlap + missing IDs against numeric model keys only
# ----------------------------
jq -n \
  --argfile ids "$tmp_ids_obj" \
  --argfile card "$CARD_JSON" '
    # numeric keys present in CARD
    ($card | keys | map(select(test("^[0-9]+$")))) as $card_num
    | ($ids  | keys) as $req
    | {
        requested_ids: $req,
        card_numeric_ids: $card_num,
        overlap_ids: ($req - ($req - $card_num)),
        missing_ids: ($req - $card_num)
      }
  ' > "$tmp_missing"

missing_n="$(jq '.missing_ids | length' "$tmp_missing")"
overlap_n="$(jq '.overlap_ids | length' "$tmp_missing")"

echo "[STATS] Requested IDs found in CARD (numeric key match): ${overlap_n}"
echo "[STATS] Requested IDs missing from CARD:                ${missing_n}"
if [ "$missing_n" -gt 0 ]; then
  echo "[INFO] Writing missing IDs to missing_Model_ID.txt"
  jq -r '.missing_ids[]' "$tmp_missing" > missing_Model_ID.txt
fi
echo

# ----------------------------
# 4) Filter CARD JSON safely:
#    - Keep ALL non-numeric keys (_comment/_timestamp/_version/etc.)
#    - For numeric keys, keep ONLY those in requested ID set
# ----------------------------
jq \
  --argfile ids "$tmp_ids_obj" '
    with_entries(
      if (.key | test("^[0-9]+$")) then
        select($ids[.key] == 1)
      else
        .
      end
    )
  ' "$CARD_JSON" > "$OUT_JSON"

# ----------------------------
# 5) Output stats post-filter
# ----------------------------
out_numeric_keys="$(jq -r 'keys[] | select(test("^[0-9]+$"))' "$OUT_JSON" | wc -l | tr -d ' ')"
out_meta_keys="$(jq -r 'keys[] | select(test("^[0-9]+$")|not)' "$OUT_JSON" | wc -l | tr -d ' ')"

echo "[STATS] Output numeric model entries kept: ${out_numeric_keys}"
echo "[STATS] Output metadata/non-numeric keys: ${out_meta_keys}"
echo "[CHECK] Output kept == overlap?          ${out_numeric_keys} vs ${overlap_n}"
echo

# Sanity: list the preserved metadata keys (first few)
echo "[INFO] Preserved non-numeric keys (head):"
jq -r 'keys[] | select(test("^[0-9]+$")|not)' "$OUT_JSON" | head
echo

echo "[DONE] Wrote ${OUT_JSON}"

rm -f "$tmp_ids_obj" "$tmp_ids_keys" "$tmp_missing"
