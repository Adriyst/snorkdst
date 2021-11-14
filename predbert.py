# coding: utf-8
import torch
from bertbst_model import BertNet
from bertbst_model import MODEL_DIR
bn = BertNet()
bn.load_state_dict(torch.load(f"{MODEL_DIR}bertdst-light-train-2.pt"))
bn.bert.load_state_dict(torch.load(f"{MODEL_DIR}light-bertstate-train-2.pt"))
bn.predict(mode="test")
