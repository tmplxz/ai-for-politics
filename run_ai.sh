#! /bin/bash

# run each individual step of our processing pipeline
for m in "translate" "summarize" "reason" "translate_results" "generate_images"
do
    python3 src/main.py --mode $m
done
