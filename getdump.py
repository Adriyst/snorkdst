# coding: utf-8
from bertbst_model import BertNet, load_model_state
import json
bn = BertNet()
for mode, num in [("train", 16), ("majority", 20), ("snorkel", 18)]:
    print("starting model with mode %s and num %s" % (mode, num))
    load_model_state(bn, mode, num)
    bn.eval()
    bn.bert.eval()
    for dataset in ("validate", "test"):
        print("starting dataset %s" % dataset)
        predset = bn.preprep.fetch_set(dataset, use_asr_hyp=5, exclude_unpointable=False)
        slot_preds, all_pred_info, all_preds = bn.get_prediction_format(predset, weighted=True)
        dump = bn.get_prediction_dump(slot_preds)
        json.dump(dump.to_json(), 
                open("./dumps/%s_%s_dump.json" % (mode, dataset), "w"), 
                ensure_ascii=False)
