#!/bin/bash
matryoshka_vis_token_scale=$1
CKPT=llava-v1.5-7b-m3
python -m llava.eval.model_vqa \
    --model-path /media/volume/new_volume/mvlm/checkpoints/llava-v1.57b-stf+m3-ori \
    --question-file /media/volume/new_volume/mvlm/playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder /media/volume/new_volume/mvlm/playground/data/eval/mm-vet/images \
    --answers-file /media/volume/new_volume/mvlm/playground/data/eval/mm-vet/answers/$CKPT.jsonl \
    --temperature 0 \
    --matryoshka_vis_token_scale 576 \
    --conv-mode vicuna_v1

mkdir -p /media/volume/new_volume/mvlm/playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src /media/volume/new_volume/mvlm/playground/data/eval/mm-vet/answers/$CKPT.jsonl \
    --dst /media/volume/new_volume/mvlm/playground/data/eval/mm-vet/results/$CKPT.json

