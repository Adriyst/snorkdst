# coding: utf-8
from bertbst_model import BertNet, load_model_state
import numpy as np
bn = BertNet()
load_model_state(bn, "majority", 20)
perps = bn.get_average_perplexity(mode="test", weighted=True)
for slot in bn.slots:
    print("slot: %s" % slot)
    for perc in (10, 25, 50, 75, 90):
        print("perc: %s" % perc)
        print(np.percentile(perps[slot], perc))
    print()
