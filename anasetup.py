# coding: utf-8
from snorkel_center import SnorkelCenter, Analyzer, LatexFormatter
sc = SnorkelCenter()
sc._vote_to_frame(sc.vote_to_frame)
ana = Analyzer(sc.dataframe)
#for slot in ("food", "area", "pricerange"):
analyses = {"food": None, "area": None, "pricerange": None}
for slot in ("food",):
    overview = ana.analyze(slot)
    analyses[slot] = overview

