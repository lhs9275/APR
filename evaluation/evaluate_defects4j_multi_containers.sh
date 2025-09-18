#!/bin/bash

# Define bug ID groups
# no 67 and 112
bug_groups=(
  "1,2,3,4,5,6,7,8,9,10,11,12"
  "13,14,15,16,17,20,21,23,24"
  "25,26,27,28,29,30,31,32,33,34,35,36"
  "37,38,39,40,41,42,43,44,45,46,47,48,18"
  "49,50,51,52,53,54,55,56,57,58,59,60,63"
  "61,62,64,66,68,69,70,71,72"
  "73,74,75,76,77,78,79,80,81,82,83,84,19"
  "85,86,87,88,89,90,91,92,93,94,95,96,22"
  "97,98,99,100,101,102,103,104,105,106,107,65"
  "108,109,110,111,113,114,115,116,117,118"
)
#bug_groups=(
#  "1,2,3,4,5,6,7,8,9,10,11,12" #9 11
#  "13,14,15,16,17,18,19,20,21,22,23,24" #15 17 18 19 20 22 23 24
#  "25,26,27,28,29,30,31,32,33,34,35,36" #25 28
#  "37,38,39,40,41,42,43,44,45,46,47,48"
#  "49,50,51,52,53,54,55,56,57,58,59,60" #55 56 57 58 60
#  "61,62,63,64,65,66,68,69,70,71,72" #62 63 65
#  "73,74,75,76,77,78,79,80,81,82,83,84"
#  "85,86,87,88,89,90,91,92,93,94,95,96"
#  "97,98,99,100,101,102,103,104,105,106,107" #102 103
#  "108,109,110,111,113,114,115,116,117,118" #108 109
#)

# Create and run a container for each group
mkdir -p log/defects4j
log_tag=all_models_label_mask
for i in {1..10}; do
  bug_ids=${bug_groups[$((i-1))]}
  docker run -dit --name defects4j_eval_$i \
    -v /home/22ys22/project/fm-apr-replay:/fm-apr-replay \
    defects4j:latest \
    bash -c "cd /fm-apr-replay/evaluation && python3 -u evaluate.py \
      --dataset defects4j \
      --model_inference_dirs 'codellama_7b_instruct_fp16_InstructionLabel,codellama_7b_instruct_fp16_InstructionMask,deepseek_coder_6.7b_instruct_fp16_InstructionLabel,deepseek_coder_6.7b_instruct_fp16_InstructionMask,deepseek_coder_v2_16b_lite_instruct_fp16_InstructionLabel,deepseek_coder_v2_16b_lite_instruct_fp16_InstructionMask' \
      --history_settings '1,2,3,4,5,6,7,8' \
      --bug_id_list $bug_ids"
done

echo "All containers started. Saving logs..."

# Optional wait to allow evaluation to start producing logs
sleep 2

# Assign non-overlapping 8-core per container, limit the cpus usage as it might consume all cores
for i in {1..10}; do
  start=$(( (i-1)*8 ))
  end=$(( start+7 ))

  docker update \
    --cpus=8.0 \
    --cpuset-cpus="${start}-${end}" \
    defects4j_eval_$i
done

# Save logs from containers to files
for i in {1..10}; do
  echo "Capturing logs for container $i..."
  docker logs -f defects4j_eval_$i > log/defects4j/defects4j_eval_${log_tag}_models_$i.log 2>&1 &
done

echo "Logs are being saved to log/defects4j/"
echo "Monitor logs with: tail -f log/defects4j/defects4j_eval_${log_tag}_models_*.log"
echo "Or monitor specific container: docker logs -f defects4j_eval_1"

# remove all containers after finish
#docker rm -f defects4j_eval_{1..10}
