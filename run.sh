echo "***Training NL-ACT***"




echo "Cloning dataset: act-grasp-stack-transfer"
git clone https://huggingface.co/datasets/kevin510/act-grasp-stack-transfer dataset

echo "Training with dataset: act-grasp-stack-transfer"
xvfb-run -a python train.py --dataset_dir ./dataset --train_instr_path ./dataset/instruction_embeddings.train.csv --val_instr_path ./dataset/instruction_embeddings.val.csv  --ckpt_dir ./output/checkpoints --eval --num_rollouts 15 --eval_instr_path ./dataset/instruction_embeddings.val.csv  --videos_dir ./output/videos --wandb_run_name act-grasp-stack-transfer

echo "Cleaning up..."
rm -rf ./dataset
rm -rf ./output
echo "Done cleaning up"




echo "Cloning dataset: act-grasp-stack-transfer-noisy"
git clone https://huggingface.co/datasets/kevin510/act-grasp-stack-transfer-noisy dataset

echo "Training with dataset: act-grasp-stack-transfer-noisy"
xvfb-run -a python train.py --dataset_dir ./dataset --train_instr_path ./dataset/instruction_embeddings.train.csv --val_instr_path ./dataset/instruction_embeddings.val.csv  --ckpt_dir ./output/checkpoints --eval --num_rollouts 15 --eval_instr_path ./dataset/instruction_embeddings.val.csv  --videos_dir ./output/videos --wandb_run_name act-grasp-stack-transfer-noisy

echo "Cleaning up..."
rm -rf ./dataset
rm -rf ./output
echo "Done cleaning up"



echo "***Training Complete***"
if [ -n "$RUNPOD_POD_ID" ]; then
    echo "Terminating Pod"
    runpodctl remove pod $RUNPOD_POD_ID
fi

