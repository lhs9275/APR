#!/bin/sh

python3 -u evaluate.py \
    --dataset defects4j \
    --model_inference_dirs "deepseek_coder_6.7b_instruct_fp16_Instruction,deepseek_coder_v2_16b_lite_instruct_fp16_Instruction" \
    --history_settings "1,2,3,4,5,6,7,8" \
#    --bug_id_list 1,2,3,4,5,6,7,8,9,10,11,12 \

#    --model_inference_dirs "deepseek_coder_6.7b_instruct_fp16_Instruction,deepseek_coder_v2_16b_lite_instruct_fp16_Instruction" \
#    --history_settings "1,2,3,4,5,6,7,8" \