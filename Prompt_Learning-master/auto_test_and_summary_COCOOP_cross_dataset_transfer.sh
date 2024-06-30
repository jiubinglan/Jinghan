#!/bin/bash

# 示例运行命令: bash auto_test_and_summary_COCOOP.sh path/to/dataset Visda_2017 yzq_output target_dataset_1 target_dataset_2 ...
# 因为目标域不止一个

# Check if the correct number of arguments is provided
if [ "$#" -lt 4 ]; then
    echo "Usage: \$0 <dataset_name> <output_file_path>"
    exit 1
fi

echo "---------------------------"
echo "参数符合要求，开始执行脚本..."

DATA=$1
shift
SOURCE_DATASET=$1
shift
output_file=$1
shift
TRAINER=$1
shift
shots=$1
shift
TARGET_DATASETS=("$@")

# Assign command-line arguments to variables
log_file="$output_file/$SOURCE_DATASET-COCOOP-DA-EXPRIMENT.log"

# Experiment: COOP
echo "---------------------"
echo "$SOURCE_DATASET dataset - COCOOP-DA-EXPRIMENT" 
echo "---------------------"

# Redirecting the output to the log file
# exec > $log_file

set -e
# set -e命令会让脚本在任何命令返回非零状态时立即退出

# seed=1
echo "---------------------"
echo "now is seed-1 traing."
# bash /home/yzq/yzq_code/multimodal-prompt-learning-main/scripts/cocoop/xd_train.sh 1 $SOURCE_DATASET $DATA > /dev/null
bash /home/yzq/yzq_code/multimodal-prompt-learning-main/scripts/cocoop/xd_train.sh 1 $SOURCE_DATASET $DATA
wait
echo "---------------------"

# seed=2
echo "---------------------"
echo "now is seed-2 traing."
# bash /home/yzq/yzq_code/multimodal-prompt-learning-main/scripts/cocoop/xd_train.sh 2 $SOURCE_DATASET $DATA > /dev/null
bash /home/yzq/yzq_code/multimodal-prompt-learning-main/scripts/cocoop/xd_train.sh 2 $SOURCE_DATASET $DATA
wait
echo "---------------------"

# seed=3
echo "---------------------"
echo "now is seed-3 traing."
# bash /home/yzq/yzq_code/multimodal-prompt-learning-main/scripts/cocoop/xd_train.sh 3 $SOURCE_DATASET $DATA > /dev/null
bash /home/yzq/yzq_code/multimodal-prompt-learning-main/scripts/cocoop/xd_train.sh 3 $SOURCE_DATASET $DATA
wait
echo "---------------------"

# Wait for all experiments to finish before parsing results

# Parse and print results
echo "---------------------"
echo "Parsing results..."
echo "---------------------"

# Loop over the SEEDS array
for SEED in 1 2 3
do
    # Loop over the TARGET_DATASETS array
    for TARGET_DATASET in "${TARGET_DATASETS[@]}"
    do
        # Run the script with the current SEED and TARGET_DATASET
        echo "---------------------"
        echo "now is seed-$SEED testing on $TARGET_DATASET."
        echo "---------------------"
        # bash scripts/cocoop/xd_test.sh $TARGET_DATASET $SEED $DATA $SOURCE_DATASET | grep "* accuracy"
        bash scripts/cocoop/xd_test.sh $TARGET_DATASET $SEED $DATA $SOURCE_DATASET    
        wait
        echo "---------------------"
        echo "now the seed-$SEED testing on $TARGET_DATASET is over."
        echo "---------------------"
    done
done



# Loop over the TARGET_DATASETS array
for TARGET_DATASET in "${TARGET_DATASETS[@]}"
do
    OUTPUT_DIR=OUTPUT/evaluation/Domain-Adaptation/${TRAINER}/${SOURCE_DATASET}_${TARGET_DATASET}/${CFG}_${SHOTS}shots
    # Run the script with the current SEED and TARGET_DATASET
    echo "---------------------"
    echo "now is parsing the results of $TARGET_DATASET."
    echo "---------------------"
    # python scripts/parse_results.py $output_file $TARGET_DATASET、
    python parse_test_res.py --path $OUTPUT_DIR --trainer_nmae $
    wait
    echo "---------------------"
    echo "now the parsing of $TARGET_DATASET is over."
    echo "---------------------"
done