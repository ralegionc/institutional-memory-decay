#!/usr/bin/env bash
# Full real-data run. Expects IMD_CONTACT to be set (Wikimedia API etiquette).
set -euo pipefail
: "${IMD_CONTACT:?set IMD_CONTACT to an email or project URL before collecting}"
imd corpus  "$@"
imd collect
imd analyze
imd figures
echo "done -- see outputs/findings.json and outputs/figures/"
