#!/bin/bash

set -e

echo "---------------------"
echo "now is WHU traing."
bash auto_test_and_summary_VPT_few_shot.sh WHU_RS19 /home/yzq/yzq_code/multimodal-prompt-learning-main/yzq_output
wait
echo "now the WHU traing is finished."
echo "---------------------"

echo "---------------------"
echo "now is NWPU_RESISC45 traing."
bash auto_test_and_summary_VPT_few_shot.sh NWPU_RESISC45 /home/yzq/yzq_code/multimodal-prompt-learning-main/yzq_output
wait
echo "now the NWPU_RESISC45 traing is finished."
echo "---------------------"

echo "---------------------"
echo "now is AID traing."
bash auto_test_and_summary_VPT_few_shot.sh AID /home/yzq/yzq_code/multimodal-prompt-learning-main/yzq_output
wait
echo "now the AID traing is finished."
echo "---------------------"

echo "---------------------"
echo "now is RS_IMAGES_2800 traing."
bash auto_test_and_summary_VPT_few_shot.sh RS_IMAGES_2800 /home/yzq/yzq_code/multimodal-prompt-learning-main/yzq_output
wait
echo "now the RS_IMAGES_2800 traing is finished."
echo "---------------------"

echo "---------------------"
echo "now is UCM traing."
bash auto_test_and_summary_VPT_few_shot.sh UCM /home/yzq/yzq_code/multimodal-prompt-learning-main/yzq_output
wait
echo "now the UCM traing is finished."

echo "---------------------"
echo "all traing is finished."
echo "ready to send message to wechat."
python /home/yzq/yzq_code/multimodal-prompt-learning-main/send_message_to_wechat.py
wait
echo "message is sent."
echo "all done."