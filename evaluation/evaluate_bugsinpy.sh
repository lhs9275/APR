#!/bin/sh

#python3 -u evaluate.py \
#    --dataset bugsinpy \
#    --model_inference_dirs "codellama_7b_instruct_fp16_Instruction,codellama_7b_instruct_fp16_InstructionLabel,codellama_7b_instruct_fp16_InstructionMask" \
#    --history_settings "1,2,3,4,5,6,7,8" \
#    --bug_id_list 4 \

#  2,3,4,5,6,7,8,9,10,11
#  16,17,18,19,20,21,22,23,24,25
#  26,27,28,29,30,31,32,33,35,36
#  37,38,41,42,43,44,45,46,47,48
#  49,50,51,52,53,54,60,61,66,67,68


python3 -u evaluate.py \
    --dataset bugsinpy \
    --model_inference_dirs "codellama_7b_instruct_fp16_Instruction,deepseek_coder_6.7b_instruct_fp16_Instruction,deepseek_coder_v2_16b_lite_instruct_fp16_Instruction" \
    --history_settings "1"
