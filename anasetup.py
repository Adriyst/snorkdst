# coding: utf-8
from snorkel_center import SnorkelCenter, Analyzer
sc = SnorkelCenter()
sc._vote_to_frame(sc.vote_to_frame)
ana = Analyzer(sc.dataframe)
overview = ana.analyze("food")
new_df = ana.structure_analysis(overview, "food")

