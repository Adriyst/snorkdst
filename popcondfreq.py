# coding: utf-8
for slot, turns in slot_preds.items():
    for t in turns:
        if t["class_label_id"] == t["class_prediction"] and t["class_prediction"] == 0:
            continue
        if t["class_label_id"] == t["class_prediction"]:
            cnts[slot]["tp"] += 1
        elif t["class_prediction"] != t["class_label_id"] and t["class_label_id"] == 0:
            cnts[slot]["fp"] += 1
        else:
            cnts[slot]["fn"] += 1
            
