#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

usage() {
  echo "Usage: $0 [-n] [-d <db_file>] <images_dir>"
  echo "  -n            Dry run (print files that would be deleted, do not delete)"
  echo "  -d <db_file>  Path to classifications.db (default: <script_dir>/classifications.db)"
  exit 1
}

DRY_RUN=false
DB_FILE="$SCRIPT_DIR/classifications.db"

while getopts ":nd:" opt; do
  case $opt in
    n) DRY_RUN=true ;;
    d) DB_FILE="$OPTARG" ;;
    *) usage ;;
  esac
done
shift $((OPTIND - 1))

[ $# -ne 1 ] && usage

IMAGES_DIR="$1"

[ ! -f "$DB_FILE" ]   && { echo "Error: Database not found: $DB_FILE"; exit 1; }
[ ! -d "$IMAGES_DIR" ] && { echo "Error: Images directory not found: $IMAGES_DIR"; exit 1; }

# Images with consensus: 2+ annotations agreeing on the same label
QUERY="
SELECT DISTINCT image_name
FROM classifications
GROUP BY image_name, label
HAVING COUNT(*) >= 2;"

deleted=0
skipped=0

while IFS= read -r fname; do
  [ -z "$fname" ] && continue
  fpath="$IMAGES_DIR/$fname"

  if [ ! -f "$fpath" ]; then
    ((skipped++)) || true
    continue
  fi

  if $DRY_RUN; then
    echo "[DRY RUN] Would delete: $fpath"
  else
    rm "$fpath"
    echo "Deleted: $fpath"
  fi
  ((deleted++)) || true

done < <(sqlite3 "$DB_FILE" "$QUERY")

echo ""
if $DRY_RUN; then
  echo "Dry run complete. Would delete: $deleted file(s). Not found (skipped): $skipped."
else
  echo "Done. Deleted: $deleted file(s). Not found (skipped): $skipped."
fi
