#!/bin/bash

# https://stackoverflow.com/a/246128/8459583
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

html="$1"
txt=$(basename "$html" .html)

mkdir -p txt

pandoc -s -f html --wrap=none --template "$SCRIPT_DIR/html2md.template" -t plain -o "txt/$txt.txt" "$html"
