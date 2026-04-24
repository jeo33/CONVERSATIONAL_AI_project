#!/bin/bash
# Resubmit failed LongBench jobs (gov_report, qmsum, vcsum).
# These datasets only have a 'test' split — previous run used 'validation' and crashed.

LOG_DIR=/home/gjp1993/links/scratch/Conversational_AI_Project/CONVERSATIONAL_AI_project/logs
mkdir -p "${LOG_DIR}"

NUM_SAMPLES=-1
SPLIT=test   # LongBench only has 'test', not 'validation'

DATASETS=(gov_report qmsum vcsum)

for DATASET in "${DATASETS[@]}"; do
  for MODE in full \
      random_b1 random_b2 random_b4 random_b6 random_b8 \
      h2o_b1_r1 \
      h2o_b2_r1 h2o_b2_r2 \
      h2o_b3_r1 h2o_b3_r2 h2o_b3_r3 \
      h2o_b4_r1 h2o_b4_r2 h2o_b4_r3 h2o_b4_r4 \
      h2o_b5_r1 h2o_b5_r2 h2o_b5_r3 h2o_b5_r4 h2o_b5_r5 \
      h2o_b6_r1 h2o_b6_r2 h2o_b6_r3 h2o_b6_r4 h2o_b6_r5 \
      h2o_b7_r1 h2o_b7_r2 h2o_b7_r3 h2o_b7_r4 h2o_b7_r5 \
      h2o_b8_r1 h2o_b8_r2 h2o_b8_r3 h2o_b8_r4 h2o_b8_r5 \
      local_b1 local_b2 local_b3 local_b4 local_b5 local_b6 local_b7 local_b8; do

      STRATEGIES=(per_head layer_shared)
      if [[ "${MODE}" == "full" || "${MODE}" == random_b* ]]; then
          STRATEGIES=(per_head)
      fi

      for STRATEGY in "${STRATEGIES[@]}"; do
          JOB_NAME="h2o_${DATASET}_${MODE}_${STRATEGY}"

          sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --account=def-gzhang-ab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=${LOG_DIR}/${JOB_NAME}_%j.out
#SBATCH --error=${LOG_DIR}/${JOB_NAME}_%j.err

export HF_HOME=/scratch/gjp1993/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

set -e

echo "Job started at: \$(date '+%Y-%m-%d %H:%M:%S')"
echo "Node: \$SLURMD_NODENAME"
echo "GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader)"

source /scratch/\$USER/activate_kv.sh

mkdir -p ${LOG_DIR}

export HF_TOKEN=\${HF_TOKEN:?HF_TOKEN must be set in the environment}

cd /scratch/gjp1993/Conversational_AI_Project/CONVERSATIONAL_AI_project/src

python h20.py --dataset ${DATASET} --num-samples ${NUM_SAMPLES} --split ${SPLIT} --kv-mode ${MODE} --h2o-strategy ${STRATEGY}

echo "Job finished at: \$(date '+%Y-%m-%d %H:%M:%S')"
EOF

          echo "Submitted: ${JOB_NAME}"
      done
  done
done
