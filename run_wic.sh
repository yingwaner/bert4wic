export BERT_BASE_DIR=bert/cased_L-12_H-768_A-12
export GLUE_DIR=data/
export TRAINED_CLASSIFIER=checkpoints/baseline
export CUDA_VISIBLE_DEVICES=0,1

python run_classifier.py \
  --task_name=wic \
  --do_train=True \
  --do_eval=True \
  --do_predict=True \
  --data_dir=$GLUE_DIR/ \
  --do_lower_case=False \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --output_dir=$TRAINED_CLASSIFIER
