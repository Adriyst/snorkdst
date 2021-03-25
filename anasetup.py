# coding: utf-8
from snorkel_center import SnorkelCenter, Analyzer, LatexFormatter
sc = SnorkelCenter()
sc._vote_to_frame(sc.vote_to_frame)
ana = Analyzer(sc.dataframe)
for slot in ("food", "area", "pricerange"):
    overview = ana.analyze(slot)
    new_df = ana.structure_analysis(overview)
    formatter = LatexFormatter(new_df)
    formatted_table = formatter.format()
    with open(f"appendix_{slot}_table_model.txt", "w") as f:
        f.writelines([f"{x}\n" for x in formatted_table])

