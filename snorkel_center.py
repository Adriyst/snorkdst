from snorkel.labeling import LFAnalysis, PandasLFApplier, labeling_function
from snorkel.labeling.model import LabelModel

import re
import pandas as pd
import numpy as np
from nltk import FreqDist
from tqdm import tqdm
import json

import snorkel_values_vote
import get_data


class SnorkelCenter:

    ABSTAIN = -1
    NEGATIVE = 0
    VOTE = 1

    def __init__(self, data=None):
        self.ontology = get_data.DataFetcher.fetch_dstc_ontology()["informable"]
        self.data = data if data is not None else \
                get_data.DataFetcher.fetch_clean_dstc("train")
        self.dataframe = self.data.to_pandas()
        self.value_voter = snorkel_values_vote.ValueVoter(self)
        self.fixing_voter = snorkel_values_vote.SnorkelFixingVoter(self)
        self.dialogue_voter = snorkel_values_vote.SnorkelDialogueVoter(self)

        self.func_order = [
                "val_in_text", "val_in_response", "val_in_request",
                "dontcare_value", "vote_whatabout", "dontcare_x",
                "support_val", "give_para_vote", "find_example",
                "keyword_found", "is_negative", #"woz_label"
        ]
        base_funcs = self.func_order.copy()
        self.func_order.extend([f"{x}_last_vote_no" for x in base_funcs])

        self.slots = ("food", "area", "pricerange")

    def cast_votes(self, **kwargs):
        for slot in tqdm(self.slots):
            labfuncs = self.value_voter.get_labeling_functions_for_slot(slot)
            fixvotes = self.fixing_voter.get_functions(slot)
            dialvotes = self.dialogue_voter.get_functions(slot)

            # First: cast utter votes

            for row_idx, row in tqdm(self.dataframe.iterrows()):
                for lf in labfuncs:
                    lfname = f"{slot}_lf_{lf.__name__}"
                    self.dataframe.at[row_idx, lfname] = lf(row, **kwargs)

            # Second: cast dial votes

            # need to instantite column name before iterating
            for lf in dialvotes:
                lfname = f"{slot}_lf_{lf.__name__}"
                if lfname not in self.dataframe.columns:
                    self.dataframe[lfname] = -1


            for row_idx, row in tqdm(self.dataframe.iterrows()):
                for lf in dialvotes:
                    lfname = f"{slot}_lf_{lf.__name__}"
                    res_tup = lf(row, **kwargs)
                    if len(res_tup) > 0:
                        for tup in res_tup:
                            rownum, val = tup
                            self.dataframe.at[rownum, lfname] = val

            # Third: cast fixing votes

            for row_idx, row in tqdm(self.dataframe.iterrows()):
                for lf in fixvotes:
                    res = lf(row, **kwargs)
                    if not isinstance(res, tuple):
                        continue

                    rownam, val, affect_idx = res
                    realname = rownam.split(f"{slot}_lf_")[1]
                    lfname = f"{slot}_lf_{realname}_{lf.__name__}"
                    if lfname not in self.dataframe.columns:
                        self.dataframe[lfname] = -1
                    self.dataframe.at[affect_idx, lfname] = val

            # Donezo!





    def get_analyses(self, vote=True, snorkel=True):
        # 1. create matrix
        if vote:
            self._vote_to_frame(self.vote_to_frame)

        analyses = []

        for slot in ["food", "area", "pricerange"]:
            rel_cols = [col for col in self.dataframe.columns if f"{slot}_lf" in col]
            matrix = np.array([
                [int(row[col]) for col in rel_cols]
                for _, row in self.dataframe.iterrows()
            ])
            # For analyses, need all labeling functions
            # all 5x funcs, then 4x exclude and 5x whatabout
            fixing_fns = [
                    labeling_function()(self.value_voter.exclude_slot(slot))
                    for _ in range(4)]
            if slot != "area":
                fixing_fns.extend([labeling_function()(self.value_voter.whatabout_vote(slot))
                    for _ in range(5)])
            for fn_idx, fn in enumerate(fixing_fns): 
                fn.name += f"_{fn_idx}"
            lab_fns = [labeling_function()(fn) for fn in 
                    self.value_voter.get_labeling_functions_for_slot(slot)]
            lab_fns.extend(fixing_fns)
            analysed = LFAnalysis(L=matrix, lfs=lab_fns)
            analyses.append(analysed.lf_summary())

        return analyses
 
    def _file_format(self):
        return [{
            "dialogue_idx": dix,
            "dialogue": [{
                "turn_label": [],
                "asr": turn.asr,
                "system_transcript": turn.system_utterance,
                "turn_idx": turn.turn_no,
                "transcript": turn.transcription,
                "system_acts": []
            } for turn in dial.turns],
            "session-id": dial.id
        } for dix, dial in enumerate(self.data.dialogues)]

    def _vote_to_frame(self, vote_func, **kwargs):
        for slot in ["food", "area", "pricerange"]:
            lab_funcs = self.value_voter.get_labeling_functions_for_slot(slot)
            fixing_funcs = self.value_voter.get_fixing_functions_for_slot(slot)
            vote_func(slot, lab_funcs, fixing_funcs, **kwargs)

    def majority_vote(self):
        dials = self._file_format()
        self.cast_votes()
        turn_index = 0
        current_dial = ""
        dial = []
        for _, row in tqdm(self.dataframe.iterrows()):
            if row.dial != current_dial:
                current_dial = row.dial
                dial = [d for d in dials if d["session-id"] == row.dial][0]
                turn_index = 0
            food_cols = [col for col in self.dataframe.columns if "food_lf" in col]
            area_cols = [col for col in self.dataframe.columns if "area_lf" in col]
            price_cols = [col for col in self.dataframe.columns if "pricerange_lf" in col]

            food_maj = [col for col in row[food_cols] if col > -1]
            area_maj = [col for col in row[area_cols] if col > -1]
            price_maj = [col for col in row[price_cols] if col > -1]
            chosen_votes = []
            for slot_votes in [food_maj, area_maj, price_maj]:
                if len(slot_votes) == 0:
                    chosen_votes.append(-1)
                    continue
                vote_freqdist = FreqDist(slot_votes)
                votedist = vote_freqdist.most_common(2)
                if len(votedist) == 1:
                    chosen_votes.append(votedist[0][0])
                    continue
                if len(set([x[1] for x in votedist])) == 1:
                    # split vote, dont vote
                    chosen_votes.append(-1)
                    continue
                chosen_votes.append(vote_freqdist.max())

            food_vote, area_vote, price_vote = chosen_votes
            for attr_name, attr in zip(["food", "area", "pricerange"],
                    [food_vote, area_vote, price_vote]):
                if attr > -1:
                    vote = self.ontology[attr_name][int(attr)] if \
                            int(attr) < len(self.ontology[attr_name]) else "dontcare"
                    attr_name = attr_name if attr_name != "pricerange" else "price range"
                    dial["dialogue"][turn_index]["turn_label"].append([attr_name, vote])
            turn_index += 1
        json.dump(dials, open("dstc2_majority_en.json", "w"))
        return dials

    def apply_snorkel_model(self, slot):
        matrix = []
        relevant_cols = [col for col in self.dataframe.columns if f"{slot}_lf" in col]
        if slot == "pricerange":
            self.func_order.insert(len(self.func_order)-1, "budget_hit")
        relevant_cols.sort(
                key=lambda x: self.func_order.index(x.split(f"{slot}_lf_")[-1])
        )
        for _, row in self.dataframe.iterrows():
            matrix.append([self.dataframe.at[row.name, c] for c in relevant_cols])
        
        matrix = np.asarray(matrix)
        label_model = LabelModel(cardinality = len(self.ontology[slot]) + 102, verbose=True)
        label_model.fit(L_train=matrix, n_epochs=800, log_freq=100, seed=607)
        res = label_model.predict(matrix)
        for i, pred in enumerate(res):
            if pred > len(self.ontology[slot]):
                self.dataframe.at[i, slot] = -1
            else:
                self.dataframe.at[i, slot] = pred


    def snorkel_vote(self):
        dials = self._file_format()
        self.cast_votes()
        for slot in ["food", "area", "pricerange"]:
            self.apply_snorkel_model(slot)
        turn_index = 0
        current_dial = np.Inf
        dial = []
        for _, row in tqdm(self.dataframe.iterrows()):
            if row.dial != current_dial:
                current_dial = row.dial
                dial = [d for d in dials if d["session-id"] == row.dial][0]
                turn_index = 0
            for attr_name, attr in zip(["food", "area", "pricerange"],
                    [row.food, row.area, row.pricerange]):
                if attr > -1:
                    vote = self.ontology[attr_name][int(attr)] if \
                            int(attr) < len(self.ontology[attr_name]) else "dontcare"
                    attr_name = attr_name if attr_name != "pricerange" else "price range"
                    dial["dialogue"][turn_index]["turn_label"].append([attr_name, vote])
            turn_index += 1
        json.dump(dials, open("dstc2_snorkel_en.json", "w"))
        return dials


    def apply_votes(self):
        matrixes, functions = [], []
        for slot in ["food", "area", "pricerange"]:
            labeling_functions = self.value_voter.get_labeling_functions_for_slot(slot)
            matrix = self.generate_voting_matrix(slot, labeling_functions)
            matrixes.append(np.matrix(matrix))
            functions.append([labeling_function()(f) for f in labeling_functions])

        analyzes = []
        for matrix, lfs in zip(matrixes, functions):
            analyzed = LFAnalysis(L=matrix, lfs=lfs)
            analyzes.append(analyzed.lf_summary())
        return analyzes

    def apply_food_votes(self):
        all_functions = [
            self.value_voter.get_labeling_functions_for_slot(slot)
            for slot in ["food", "area", "pricerange"]
        ]
        frames = []
        analyses = []
        for f in all_functions:
            applier = PandasLFApplier(lfs=f)
            applied_res = applier.apply(df=self.dataframe)
            frames.append(applied_res)
            analyzed = LFAnalysis(L=applied_res, lfs=f).lf_summary()
            analyses.append(analyzed)

        print("votes -> vote matrix")
        print("analyses -> LFA results")
        return frames, analyses

    def vote_to_frame(self, slot, labeling_functions, 
            fixing_functions, **kwargs):
        # In order to fit a snorkel model, we need to create an
        # actual matrix. Probably best to add new rows to the frame 
        # originally.
        
        # keep a list of which fixing functions are to be applied
        # to each dial
        dials_to_fix = {}
        fixer = DialogueFixing(slot)

        for _, row in tqdm(self.dataframe.iterrows()):
            for lf in labeling_functions:
                col_name = f"{slot}_lf_{lf.__name__}"
                if col_name not in self.dataframe.columns:
                    self.dataframe[col_name] = -1
                vote = lf(row, **kwargs)
                self.dataframe.at[row.name, col_name] = vote

            # Fixing functions : Their vote depends on the vote of the 
            # labeling functions they are fixing, and will vote 
            # the opposite of them. For now, dialogue based.
            # Should fetch the previous votes in the same dialogue. Means 
            # that it should only trigger for the last utterance, correcting 
            # all previous turns.

            all_in_dial = self.dataframe.query(f"dial == '{row.dial}'")
            if row.dial in fixer.fixing_results:
                continue
            else:
                fixer.fixing_results[row.dial] = []

            for _, fix_row in all_in_dial.iterrows():
                for ff in fixing_functions:
                    res = ff(fix_row)
                    fixer.fixing_results[row.dial].append(res)
                        
        fixer.apply(self.dataframe)

    def generate_voting_matrix(self, slot, labeling_functions):
        """
        We want to modify votes from previous utterances in a dialogue. Therefore,
        the snorkel methods for applying labeling functions are not sufficient.
        This function serves to be a different way of creating a voting matrix.
        """
        matrix = []
        excludeds = []
        # get first utterance from each dialogue
        first_from_dial = self.dataframe.drop_duplicates(subset=["dial"])
        for _, row in first_from_dial.iterrows():
            all_utterances = self.dataframe.query(f"dial == '{row.dial}'")
            #slot_negated = self.value_voter.get_exclusion_for_dial(slot, all_utterances)
            for _, utterance_row in all_utterances.iterrows():
                votes = []
                for lf in labeling_functions:
                    vote = lf(utterance_row)
                    if utterance_row.dial == "voip-d225fad9df-20130328_204846":
                        print(utterance_row.name)
                        print(vote)
                        print(utterance_row.transcription)
                    #if vote > 0 and slot_negated:
                    #    excludeds.append(1)
                    #    votes.append(-1)
                    #else:
                    #    excludeds.append(0)
                    votes.append(vote)
                matrix.append(votes)
        return matrix, excludeds


class LabFuncMap:

    def __init__(self, func1, func2):
        self.func_1 = func1
        self.func_2 = func2
        self.overlap = []
        self.conflict = []

    def count_overlap(self):
        return len(self.overlap)

    def count_conflict(self):
        return len(self.conflict)


class LabFunc:

    def __init__(self, name):
        self.name = name


class AnalyzeLFOverview:

    def __init__(self):
        self.maps = []

    def get_map(self, func_1, func_2):
        candidates = [m for m in self.maps
                if func_1.name in (m.func_1.name,m.func_2.name)
                and func_2.name in (m.func_1.name,m.func_2.name)
        ]
        if len(candidates) > 1:
            raise ValueError("More than one map found for two functions")
        if len(candidates) == 1:
            return candidates[0]
        new_map = LabFuncMap(func_1, func_2)
        self.maps.append(new_map)
        return new_map

    def get_maps_for_lf(self, fn):
        return [m for m in self.maps if fn.name in (m.func_1.name, m.func_2.name)]

    def get_all_labfuncs(self):
        lfs = []
        for map in self.maps:
            if map.func_1 not in lfs:
                lfs.append(map.func_1)
            if map.func_2 not in lfs:
                lfs.append(map.func_2)
        return lfs


class FuncStats:

    def __init__(self, maps, lf, df):
        self.maps = maps
        self.lf_col = lf.name
        self.overlap_rate = 0
        self.conflict_rate = 0
        self.overlap_turn_count = 0
        self.conflict_turn_count = 0
        self.overlap_total_count = 0
        self.conflict_total_count = 0
        self.times_applied = len(df.query(f"{lf.name} > -1"))
        self.coverage = round((self.times_applied/len(df)), 3)
        self.parse()

    def parse(self):
        overlap_turns, conflict_turns = [], []
        for m in self.maps:
            overlap_turns.extend(m.overlap)
            conflict_turns.extend(m.conflict)
        overlap = len(set(overlap_turns))
        conflict = len(set(conflict_turns))
        if self.times_applied > 0:
            self.overlap_rate = round((overlap/self.times_applied), 3)
            self.conflict_rate = round((conflict/self.times_applied), 3)

        self.overlap_turn_count = overlap
        self.conflict_turn_count = conflict
        self.overlap_total_count = len(overlap_turns)
        self.conflict_total_count = len(conflict_turns)


class FuncStatsPandas:

    def __init__(self, fstats):
        filtered = []
        blacklist = []
        for fst in fstats:
            if fst.lf_col not in blacklist:
                filtered.append(fst)
                blacklist.append(fst.lf_col)
        self.fstats = filtered

    def create_pandas(self):
        shell = {k: [] for k in dir(self.fstats[0]) if ("conflict" in k or "overlap" in k
            or "coverage" in k)}
        for col, nums in shell.items():
            for fstat in self.fstats:
                nums.append(getattr(fstat, col))
        return pd.DataFrame(shell, index=[fstat.lf_col for fstat in self.fstats])


class Analyzer:

    def __init__(self, dataframe):
        self.df = dataframe

    def analyze(self, slot):
        relevant_columns = [col for col in self.df.columns if f"{slot}_lf_" in col]
        overview = AnalyzeLFOverview()
        for col in relevant_columns:
            lf = LabFunc(col)
            for other_col in [c for c in relevant_columns if c != col]:
                other_lf = LabFunc(other_col)
                mapping = overview.get_map(lf, other_lf)
                    
                subset = self.df.query(f"{col} > -1 and {other_col} > -1")
                for _, row in subset.iterrows():
                    if row[col] == row[other_col]:
                        mapping.overlap.append(row.name)
                    else:
                        mapping.conflict.append(row.name)
        return overview

    def get_stats_for_fn(self, lf: LabFunc, overview: AnalyzeLFOverview):
        maps = overview.get_maps_for_lf(lf)
        return FuncStats(maps, lf, self.df)
    
    def structure_analysis(self, overview):
        lfs = overview.get_all_labfuncs()
        fspd = FuncStatsPandas([self.get_stats_for_fn(lf, overview) for lf in lfs])
        res_df = fspd.create_pandas()
        return res_df
        

class DialogueFixing:

    def __init__(self, slot):
        ## list of fixingresults
        self.fixing_results = {}
        self.slot = slot

    def apply(self, dataframe):
        for dialogue, results in self.fixing_results.items():
            applied_results = [x.position for x in results if x.apply]
            affected_dialogue = dataframe.query(f"dial == '{dialogue}'")
            final_result = min(applied_results) if any(applied_results) else\
                min(affected_dialogue.index)
            min_idx = final_result - min(affected_dialogue.index)
            affected_dialogue = affected_dialogue[:min_idx]
            if len(affected_dialogue) == 0:
                for _, turn in affected_dialogue.iterrows():
                    for funcname in [x for y in results for x in y.affected]:
                        dataframe.at[turn.name, funcname] = -1
            for res in results:
                vote = -1
                for affected_func in res.affected:
                    col_name = f"{self.slot}_lf_{affected_func}"
                    fix_col_name = f"{col_name}_{res.ff.__name__}"
                    if fix_col_name not in dataframe:
                        dataframe[fix_col_name] = -1
                    for _, affected_utt in affected_dialogue.query(
                            f"{col_name} > -1"
                    ).iterrows():
                        vote = -1 
                        if res.apply:
                            vote = affected_utt[col_name]
                            if not res.positive:
                                vote += 100
                        dataframe.at[affected_utt.name, fix_col_name] = vote


class FormatEntry:
    """
    Object representation of a row in a dataframe, that is to be formatted
    with latex
    """

    def __init__(self, row, order, operations):
        self.row = row
        self.order = order
        self.operations = operations
        self.operation_map = {
            "name": lambda x: x,
            "perc": lambda x: round(x*100, 3) if x > 0 else 0,
            "rate": lambda x,y: round(x/y, 2) if x > 0 else 0
        }
        self.postfixes = {
            "name": "",
            "perc": "\%",
            "rate": ""
        }
        self.func_name_table = {
            "valintext": "VIT",
            "valinresponse": "VIRS",
            "valinrequest": "VIRQ",
            "dontcarevalue": "DCV",
            "votewhatabout": "VWA",
            "dontcarex": "DCX",
            "supportval": "SV",
            "giveparavote": "PV",
            "findexample": "FE",
            "keywordfound": "KWF",
            "isnegative": "NEG",
            "valintextlastvoteno": "VIT-LVN",
            "valinresponselastvoteno": "VIRS-LVN",
            "valinrequestlastvoteno": "VIRQ-LVN",
            "dontcarevaluelastvoteno": "DCV-LVN",
            "supportvallastvoteno": "SV-LVN",
            "giveparavotelastvoteno": "GPV-LVN",
            "findexamplelastvoteno": "FE-LVN",
            "keywordfoundlastvoteno": "KWF-LVN",
            "valintextvalfromsecond": "VIT-VFS",
            "valinresponsevalfromsecond": "VIRS-VFS",
            "valinrequestvalfromsecond": "VIRQ-VFS",
            "dontcarevaluevalfromsecond": "DCV-VFS",
            "supportvalvalfromsecond": "SV-VFS",
            "giveparavotevalfromsecond": "GPV-VFS",
            "findexamplevalfromsecond": "FE-VFS",
            "keywordfoundvalfromsecond": "KWF-VFS",

            "valintextexcludefunc": "VIT-REJ",
            "valinresponseexcludefunc": "VIRS-REJ",
            "valinrequestexcludefunc": "VIRQ-REJ",
            "dontcarevalueexcludefunc": "DCV-REJ",
            "votewhataboutexcludefunc": "VWA-REJ"
        }

    def generate_format_string(self):
        s = ""
        for column, op in zip(self.order, self.operations):
            # special case
            if column == "name":
                removed_first_part = re.sub(r"\w+_lf_", "", self.row.name)
                num = removed_first_part.replace("_", "")
                num = self.func_name_table[num]
            elif isinstance(column, str):
                num = self.operation_map[op](self.row[column])
            elif isinstance(column, tuple):
                nums = []
                for col in column:
                    nums.append(self.row[col])
                num = self.operation_map[op](*nums)

            s += str(num)
            s += self.postfixes[op]
            s += " & "
        s = s[:-2]
        s += "\\\\"
        return s

class LatexFormatter:

    ORDER = ["name", "coverage", "overlap_rate",
            ("overlap_total_count","overlap_turn_count"), "conflict_rate",
            ("conflict_total_count", "conflict_turn_count")]
    OPERATIONS = ["name", "perc", "perc", "rate", "perc", "rate"]

    def __init__(self, dataframe):
        self.dataframe = dataframe

    def format(self):
        table_rows = []
        for _, row in self.dataframe.iterrows():
            entry = FormatEntry(row, self.ORDER, self.OPERATIONS)
            table_rows.append(entry.generate_format_string())
        return table_rows



if __name__ == '__main__':
    sc = SnorkelCenter()
    sc.get_analyses()
