import numpy as np
import math
import re
import pandas as pd
from bs4 import BeautifulSoup
import random

from DCNN import DCNN
import MyCustomCallback as mccb
"""
for google colab env:
try:
    %tensorflow_version 2.x
except Exception:
    pass
"""
import tensorflow as tf
import tensorflow_hub as hub
import bert

cols = ["sentiment", "id", "date", "query", "user", "text"]
data = pd.read_csv(
    "testdata.csv",
    header=None,
    names=cols,
    engine="python",
    encoding="latin1"
)

data.drop(["id", "date", "query", "user"],
          axis=1,
          inplace=True)

print("data", data.head(6))

def clean_tweet(tweet):
    tweet = BeautifulSoup(tweet, "lxml").get_text()
    # Eliminar el @
    tweet = re.sub(r"@[A-Za-z0-9]+", ' ', tweet)
    # Eliminar los links de la URL
    tweet = re.sub(r"https?://[A-Za-z0-9./]+", ' ', tweet)
    # Conservamos solamente las letras
    tweet = re.sub(r"[^a-zA-Z.!?']", ' ', tweet)
    # Eliminamos espacios en blanco adicionales
    tweet = re.sub(r" +", ' ', tweet)
    return tweet

data_clean = [clean_tweet(tweet) for tweet in data.text]
print("data_clean", data_clean)

data_labels = data.sentiment.values
data_labels[data_labels == 4] = 1

FullTokenizer = bert.bert_tokenization.FullTokenizer
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                            trainable=False)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = FullTokenizer(vocab_file, do_lower_case)

def encode_sentence(sent):
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent))

data_inputs = [encode_sentence(sentence) for sentence in data_clean]

data_with_len = [[sent, data_labels[i], len(sent)]
                 for i, sent in enumerate(data_inputs)]
random.shuffle(data_with_len)
data_with_len.sort(key=lambda x: x[2])
sorted_all = [(sent_lab[0], sent_lab[1])
              for sent_lab in data_with_len if sent_lab[2] > 7]

all_dataset = tf.data.Dataset.from_generator(lambda: sorted_all,
                                             output_types=(tf.int32, tf.int32))
next(iter(all_dataset))

BATCH_SIZE = 32
all_batched = all_dataset.padded_batch(BATCH_SIZE, padded_shapes=((None, ), ()))

next(iter(all_batched))


NB_BATCHES = math.ceil(len(sorted_all) / BATCH_SIZE)
NB_BATCHES_TEST = NB_BATCHES // 10
all_batched.shuffle(NB_BATCHES)
test_dataset = all_batched.take(NB_BATCHES_TEST)
train_dataset = all_batched.skip(NB_BATCHES_TEST)

VOCAB_SIZE = len(tokenizer.vocab)
EMB_DIM = 200
NB_FILTERS = 100
FFN_UNITS = 256
NB_CLASSES = 2

DROPOUT_RATE = 0.2

NB_EPOCHS = 5

Dcnn = DCNN(vocab_size=VOCAB_SIZE,
            emb_dim=EMB_DIM,
            nb_filters=NB_FILTERS,
            FFN_units=FFN_UNITS,
            nb_classes=NB_CLASSES,
            dropout_rate=DROPOUT_RATE)

if NB_CLASSES == 2:
    Dcnn.compile(loss="binary_crossentropy",
                 optimizer="adam",
                 metrics=["accuracy"])
else:
    Dcnn.compile(loss="sparse_categorical_crossentropy",
                 optimizer="adam",
                 metrics=["sparse_categorical_accuracy"])

checkpoint_path = "./drive/My Drive/Curso de NLP/BERT/ckpt_bert_tok/"

ckpt = tf.train.Checkpoint(Dcnn=Dcnn)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print("Último checkpoint restaurado!!")

#definir My Custtom Callback
Dcnn.fit(train_dataset,
         epochs=NB_EPOCHS,
         callbacks=[mccb.MyCustomCallback(ckpt_manager)])
