GPU_ID=0
export CUDA_VISIBLE_DEVICES=${GPU_ID}
bash scripts/mmadapter/base2new_train.sh oxford_pets 2 vit_b16_ep5
bash scripts/mmadapter/base2new_test.sh oxford_pets 2 vit_b16_ep5 5