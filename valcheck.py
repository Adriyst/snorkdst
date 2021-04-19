# coding: utf-8
from snorkel_center import SnorkelCenter
from nltk import FreqDist
asr_sc = SnorkelCenter()
asr_sc.cast_votes()
real_sc = SnorkelCenter()
real_sc.cast_votes(real=True)
for (turn_idx, turn), (rl_turn_idx, rl_turn) in zip(
        asr_sc.dataframe.iterrows(), real_sc.dataframe.iterrows()):
    for slot in ("food", "area", "pricerange"):
        rel_cols = [col for col in real_sc.dataframe.columns if f"{slot}_lf_" in col]
        vote_cols = [rl_turn[col] for col in rel_cols if rl_turn[col] > -1]
        if len(vote_cols) == 0:
            continue
        maxvote = FreqDist(vote_cols).max()
        asr_sc.dataframe.at[turn_idx, slot] = maxvote
            
df = asr_sc.dataframe.sample(frac=1)
for row_idx, row in df.iterrows():
    for slot in ("food", "area", "pricerange"):
        rel_cols = [col for col in asr_sc.dataframe.columns if f"{slot}_lf_" in col]
        vote_cols = [row[col] for col in rel_cols if row[col] > -1]
        if not len(vote_cols):
            if row[slot] > -1:
                print(row.transcription)
                print(row.system_transcription)
                print(row.real_transcription)
                print(row)
                input()
            continue
        maxvote = FreqDist(vote_cols).max()
        if maxvote != row[slot]:
            print(row.transcription)
            print(row.system_transcription)
            print(row.real_transcription)
            print(row)
            input()
            
