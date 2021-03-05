#!/usr/bin/env bash
export BERT_DIR=/data/xueyou/data/bert_pretrain/chinese_L-12_H-768_A-12
export OUTPUT_DIR=/nfs/users/xueyou/data/chinese_nlp/clue_ner/processed/biaffine_ner_pretrain
export DATA_DIR=/nfs/users/xueyou/data/chinese_nlp/clue_ner/processed

python run_ner.py \
  --task_name=cluener \
	--vocab_file=${BERT_DIR}/vocab.txt \
	--bert_config_file=${BERT_DIR}/bert_config.json \
	--init_checkpoint=/nfs/users/xueyou/data/chinese_nlp/clue_ner/pretrain/pretraining_output/model.ckpt-10500 \
  --do_lower_case=True \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=3e-5 \
  --num_train_epochs=5.0 \
  --neg_sample=1.0 \
  --save_checkpoints_steps=5000 \
	--do_train=True \
  --focal_loss=False \
  --dice_loss=False \
  --data_dir=${DATA_DIR} \
  --output_dir=${OUTPUT_DIR}
