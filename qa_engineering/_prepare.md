
mkdir -p model
curl -o model/multilingual_L-12_H-768_A-12.zip https://storage.googleapis.com/bert_models/2018_11_03/multilingual_L-12_H-768_A-12.zip
unzip model/multilingual_L-12_H-768_A-12.zip -d model/bert_base

mkdir -p data/raw
curl -o data/raw/train-v1.1.json https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
curl -o data/raw/dev-v1.1.json https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
curl -o data/raw/evaluate-v1.1.py https://raw.githubusercontent.com/allenai/bi-att-flow/master/squad/evaluate-v1.1.py

BERT_BASE_DIR=model/bert_base/multilingual_L-12_H-768_A-12
SQUAD_DIR=data/raw

python run_squad.py \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --do_train=True \
  --train_file=$SQUAD_DIR/train-v1.1.json \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/dev-v1.1.json \
  --train_batch_size=1 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=80 \
  --doc_stride=40 \
  --output_dir=/tmp/squad_base/