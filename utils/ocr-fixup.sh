#!/bin/bash
# Apply some simple transformations to OCR'd text.

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SCRATCH_FILE=$SCRIPT_DIR/scratchpad

if [ ! -f "$SCRATCH_FILE" ]; then
    echo "Scratch file not found: $SCRATCH_FILE"
    exit 1
fi

# remove hyphenated word breaks
sed -i ':a;N;$!ba;s/-\n//g' "$SCRATCH_FILE"

# join sentences that were split across lines
sed -i ':a;N;$!ba;s/\([^\n]\)\(\n\)\(^[a-z]\)/\1 \3/g' "$SCRATCH_FILE"

# prettyfy
sed -i "s/negative (-)/negative (\`-\`)/g" "$SCRATCH_FILE"
sed -i "s/positive (+)/positive (\`+\`)/g" "$SCRATCH_FILE"
sed -i "s/^CAUTION:/:warning: **CAUTION**:/g" "$SCRATCH_FILE"
