BASEPATH = "/usr/local/Development/"
MASTERPATH = BASEPATH + "master/"
SNORKPATH = MASTERPATH + "snorkdst/"

CONFIG = {
    "MODEL": {
        "BERT_VOCAB_LOC": BASEPATH + "bert/vocab.txt",
        "MODEL_DIR": "/run/media/adriantysnes/HDD/models/",
        "DATASETPATH": MASTERPATH + "data/clean/",
        "SENT_MAX_LEN": 180,
        "BERT_VERSION": "bert-base-uncased",
        "NUM_EPOCHS": 30,
        "TRAIN_BATCH_SIZE": 16,
        "LEARN_RATE": 2e-5,
        "DROPOUT": 0.3,
        "ASR_HYPS": 1
    }
}
