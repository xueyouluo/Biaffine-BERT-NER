export BERT_DIR=/data/xueyou/data/bert_pretrain/chinese_L-12_H-768_A-12
export DATA_DIR=/nfs/users/xueyou/data/chinese_nlp/clue_ner/pretrain

# python create_pretraining_data.py \
#   --input_file=$DATA_DIR/raw_text.txt \
#   --output_file=$DATA_DIR/tf_examples.tfrecord \
#   --vocab_file=$BERT_DIR/vocab.txt \
#   --do_lower_case=True \
#   --max_seq_length=128 \
#   --max_predictions_per_seq=20 \
#   --masked_lm_prob=0.15 \
#   --random_seed=12345 \
#   --dupe_factor=5

python run_pretraining.py \
  --input_file=$DATA_DIR/tf_examples.tfrecord \
  --output_dir=$DATA_DIR/pretraining_output \
  --do_train=False \
  --do_eval=True \
  --bert_config_file=$BERT_DIR/bert_config.json \
  --init_checkpoint=$BERT_DIR/bert_model.ckpt \
  --train_batch_size=32 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=10500 \
  --num_warmup_steps=1000 \
  --learning_rate=2e-5
