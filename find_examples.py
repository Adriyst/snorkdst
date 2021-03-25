# coding: utf-8
from snorkel_center import SnorkelCenter, Analyzer, LatexFormatter
import random
import re
sc = SnorkelCenter()
sc._vote_to_frame(sc.vote_to_frame)
examples = []
relevant_cols = [x for x in sc.dataframe.columns if "_lf_" in x]
for _, row in sc.dataframe.iterrows():
    if sum(row[relevant_cols].apply(
        lambda x: x + 1
    ).apply(lambda x: max(x, 0))) > 0:
        continue
    if re.search(r"(bye|phone|address|code)", row.transcription):
        continue
    examples.append(row)
random.shuffle(examples)

for ex in examples:
    print(ex.system_transcription)
    print(ex.transcription)
    q = input()
    if q == "m":
        print(ex)
    elif q == "q":
        break
    elif len(q) > 0:
        found_ex = [x for x in examples if q in x.transcription
                or q in x.system_transcription]
        for fex in found_ex[:10]:
            print(fex.transcription)
            print(fex.system_transcription)
            print(fex)
            input()
        input()



