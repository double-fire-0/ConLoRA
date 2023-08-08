id=0
free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i $id | grep -Eo [0-9]+)
echo $free_mem # this prints out: memory.free [MiB] 1954 MiB
while [ $free_mem -lt 10000 ]
do
    free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i $id | grep -Eo [0-9]+)
    sleep 5
done
torchrun --nproc_per_node 8 train.py --cfg-path train_configs/conlora_ocr_2dataset_bodyllm.yaml
# python demo_conlora.py --cfg-path eval_configs/conlora_det_bodyllm_eval.yaml  --gpu-id 0
