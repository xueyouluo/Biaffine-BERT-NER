#!/usr/bin/env bash
export BERT_DIR=/data/xueyou/data/bert_pretrain/chinese_L-12_H-768_A-12
export OUTPUT_DIR=/nfs/users/xueyou/data/chinese_nlp/clue_ner/processed/biaffine_ner_pretrain
export DATA_DIR=/nfs/users/xueyou/data/chinese_nlp/clue_ner/processed

python run_ner.py \
  --task_name=cluener \
	--vocab_file=${BERT_DIR}/vocab.txt \
	--bert_config_file=${BERT_DIR}/bert_config.json \
	--init_checkpoint=${BERT_DIR}/bert_model.ckpt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --predict_batch_size=1 \
  --output_score=False \
	--do_predict=False \
	--do_eval=True \
  --data_dir=${DATA_DIR} \
  --output_dir=${OUTPUT_DIR}
