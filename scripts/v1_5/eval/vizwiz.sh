#!/bin/bash
matryoshka_vis_token_scale=$1
CKPT=llava-v1.5-7b-m3-${matryoshka_vis_token_scale}
python -m llava.eval.model_vqa_loader \
    --model-path /media/volume/new_volume/mvlm/checkpoints/llava-v1.57b-stf+m3-ori \
    --question-file /media/volume/new_volume/mvlm/playground/data/eval/vizwiz/llava_test.jsonl \
    --image-folder /media/volume/new_volume/mvlm/playground/data/eval/vizwiz/test \
    --answers-file /media/volume/new_volume/mvlm/playground/data/eval/vizwiz/answers/${CKPT}.jsonl \
    --temperature 0 \
    --matryoshka_vis_token_scale 576 \
    --conv-mode vicuna_v1

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file /media/volume/new_volume/mvlm/playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file /media/volume/new_volume/mvlm/playground/data/eval/vizwiz/answers/${CKPT}.jsonl \
    --result-upload-file /media/volume/new_volume/mvlm/playground/data/eval/vizwiz/answers_upload/${CKPT}.json
