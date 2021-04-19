# coding: utf-8
from snorkel_center import SnorkelCenter, Analyzer, LatexFormatter
sc = SnorkelCenter()
sc.cast_votes()
ana = Analyzer(sc.dataframe)
analyses = {"food": None, "area": None, "pricerange": None}
for slot in analyses.keys():
    overview = ana.analyze(slot)
    new_df = ana.structure_analysis(overview)
    analyses[slot] = new_df

