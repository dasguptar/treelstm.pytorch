#!/bin/bash
set -e
bash scripts/download.sh

CLASSPATH="lib:lib/stanford-parser/stanford-parser.jar:lib/stanford-parser/stanford-parser-3.5.1-models.jar"
javac -cp $CLASSPATH lib/*.java
python scripts/preprocess-sick.py

mkdir -p checkpoints/
