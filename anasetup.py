# coding: utf-8
from snorkel_center import SnorkelCenter, Analyzer, LatexFormatter
slot = "food"
sc = SnorkelCenter()
sc._vote_to_frame(sc.vote_to_frame)
ana = Analyzer(sc.dataframe)
overview = ana.analyze(slot)
new_df = ana.structure_analysis(overview)
formatter = LatexFormatter(new_df)
formatted_table = formatter.format()

