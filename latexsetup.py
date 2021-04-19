# coding: utf-8
from snorkel_center import SnorkelCenter, Analyzer, LatexFormatter
def outline(tab, slot):
    s = "\\vspace{1cm}\n"
    s += "\\begin{table}[h!]\n"
    s += "\\begin{center}\n"
    s += "\\begin{tabular}{|| c | c | c | c | c | c ||}\n"
    s += "\\hline\n"
    s += "Function & Coverage & Overlap & $O_{pi}$* & Conflict & $C_{pi}$* \\\\\n"
    s += "\\hline\n"
    s += tab
    s += "\\hline\n"
    s += "\\end{tabular}\n"
    s += "\\end{center}\n"
    s += "\\caption{Results of labeling functions for the %s slot.}\n" % slot
    s += "\\label{table:initial-%s}\n" % slot
    s += "\\end{table}"
    return s

sc = SnorkelCenter()
sc._vote_to_frame(sc.vote_to_frame)
ana = Analyzer(sc.dataframe)
for slot in ("food", "area", "pricerange"):
    overview = ana.analyze(slot)
    new_df = ana.structure_analysis(overview)
    formatter = LatexFormatter(new_df)
    formatted_table = formatter.format()
    slotname = slot if slot != "pricerange" else "price"
    with open(f"initial_{slotname}_table.tex", "w") as f:
        f.write(outline("".join([f"{x}\n" for x in formatted_table]), slot))

