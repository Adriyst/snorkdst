import json
import torch
from tqdm import tqdm

from bertbst_model import BertNet, load_model_state, PredictionSet
from config import CONFIG

gold_bn = BertNet()
load_model_state(gold_bn, "train", 16)

maj_bn = BertNet()
load_model_state(maj_bn, "snorkel", 18)
maj_bn.eval()
maj_bn.bert.eval()


predset = maj_bn.preprep.fetch_set(
        "validate",
        use_asr_hyp=CONFIG["MODEL"]["ASR_HYPS"],
        exclude_unpointable=False
)
batch_generator = maj_bn.loader.fetch_batch(
        predset, 1, maj_bn.slots, maj_bn.tokenizer, pred=True, slot_value_dropout=0
)

maj_counter = {"good": 0, "tot": 0}
gold_counter = {"good": 0, "tot": 0}
with torch.no_grad():
    for batch in tqdm(batch_generator):
        dc_slot = ""
        for slot in maj_bn.slots:
            if batch.cat_map[slot]["class"].item() == 1:
                dc_slot = slot
                break
        if len(dc_slot) == 0:
            continue
        m_class_logits, m_pos_logits = maj_bn(batch, train=False, weighted=True)
        g_class_logits, g_pos_logits = gold_bn(batch, train=False, weighted=True)
        maj_preds = PredictionSet(*m_class_logits, *m_pos_logits)
        gold_preds = PredictionSet(*g_class_logits, *g_pos_logits)
        m_score = int(maj_preds.cat_map[dc_slot]["class"].argmax().item() == 1)
        g_score = int(gold_preds.cat_map[dc_slot]["class"].argmax().item() == 1)
        maj_counter["good"] += m_score
        maj_counter["tot"] += 1
        gold_counter["good"] += g_score
        gold_counter["tot"] += 1

    
