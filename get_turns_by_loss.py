import json
import torch

from config import CONFIG
from bertbst_model import BertNet, load_model_state, PredictionSet
from dumps.evaluate_dump import dials_only_gold_good

bn = BertNet()
load_model_state(bn, "majority", 20)
bn.eval()
bn.bert.eval()

dumpz = [json.load(open(f"./dumps/{v}_test_dump.json")) for v in ["train", "majority",
    "snorkel"]]
maj_dump = dumpz[1]
right_dials = dials_only_gold_good(*dumpz)
maj_dials = [x[1] for x in right_dials]
lenz = {x["dial_idx"]: len(x["turns"]) for x in maj_dump}
losses = bn.get_loss_for_set(mode="test", weighted=True)
cls_losses = losses[1]["class"]
loss_dicts = [{"food": cls_losses[0], "area": cls_losses[1] - cls_losses[0],
    "price range": cls_losses[2] - cls_losses[1]}]
for i in range(3, len(cls_losses), 3):
    loss_dicts.append(
        {
            "food": cls_losses[i] - cls_losses[i-1], 
            "area": cls_losses[i+1] - cls_losses[i],
            "price range": cls_losses[i+2] - cls_losses[i+1]
        }
    )

fails = []
for maj in maj_dials:
    dial_idx = maj["dial_idx"]
    totlen = sum([lenz[i] for i in range(dial_idx)]) + maj["fail_idx"]
    slot_failed, loss = list(sorted(
        loss_dicts[totlen].items(), key=lambda x:x[1], reverse=True
    ))[0]
    fails.append({"dial": dial_idx, "turn": maj["fail_idx"], "slot": slot_failed,
        "loss": loss})
fails = list(sorted(fails, key= lambda x: x["loss"], reverse=True))
predset = bn.preprep.fetch_set("test", 
        use_asr_hyp=CONFIG["MODEL"]["ASR_HYPS"],
        exclude_unpointable=False
)
batch_generator = bn.loader.fetch_batch(
        predset, 1, bn.slots, bn.tokenizer, pred=True, slot_value_dropout=0
)
idxs_to_watch = list(sorted(
    [(f["dial"], sum([lenz[i] for i in range(f["dial"])]) + f["turn"]) for f in fails]
))
to_watch = idxs_to_watch.pop(0)

for b_idx, batch in enumerate(batch_generator):
    if b_idx == to_watch[1]:
        with torch.no_grad():
            class_logits, pos_logits = bn(batch, train=False, weighted=True)
            preds = PredictionSet(*class_logits, *pos_logits)
            f_to_update = [f for f in fails if f["dial"] == to_watch[0]][0]
            f_to_update["preds"] = \
            preds.cat_map["".join(f_to_update["slot"].split(" "))]["class"].cpu().detach().numpy()
            try:
                to_watch = idxs_to_watch.pop(0)
            except IndexError:
                break




