BASEPATH = "/usr/local/Development/"
MASTERPATH = BASEPATH + "master/"
SNORKPATH = MASTERPATH + "snorkdst/"
TORCH_DATA_DIR = "/run/media/adriantysnes/HDD/"

CONFIG = {
    "MODEL": {
        "BERT_VOCAB_LOC": BASEPATH + "bert/vocab.txt",
        "MODEL_DIR": TORCH_DATA_DIR + "models/",
        "DATASETPATH": MASTERPATH + "data/clean/",
        "SENT_MAX_LEN": 180,
        "BERT_VERSION": "bert-base-uncased",
        "NUM_EPOCHS": 50,
        "TRAIN_BATCH_SIZE": 4,
        "LEARN_RATE": 2e-5,
        "DROPOUT": 0.0,
        "ASR_HYPS": 5,
        "ASR_THRESHOLD": 0.15
    },
    "VECTORS": TORCH_DATA_DIR + "vectors/",
    "VECTOR_BATCH_SIZE": 500
}
