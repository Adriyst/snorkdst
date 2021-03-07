from snorkel.labeling import LFAnalysis, PandasLFApplier, labeling_function
from snorkel.labeling.model import LabelModel

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

        self.func_order = [
                "val_in_text", "val_in_response", "val_in_request",
                "woz_label", "dontcare_value"
        ]
        self.func_order.extend([f"{x}_exclude_func" for x in self.func_order])
 
    def _file_format(self):
        return [{
            "dialogue_idx": dix,
            "dialogue": [{
                "turn_label": [],
                "asr": [],
                "system_transcript": turn.system_utterance,
                "turn_idx": turn.turn_no,
                "transcript": turn.transcription,
                "system_acts": []
            } for turn in dial.turns],
            "session-id": dial.id
        } for dix, dial in enumerate(self.data.dialogues)]

    def _vote_to_frame(self, vote_func):
        for slot in ["food", "area", "pricerange"]:
            lab_funcs = self.value_voter.get_labeling_functions_for_slot(slot)
            fixing_funcs = self.value_voter.get_fixing_functions_for_slot(slot)
            vote_func(slot, lab_funcs, fixing_funcs)

    def majority_vote(self):
        dials = self._file_format()
        self._vote_to_frame(self.vote_to_frame)
        turn_index = 0
        current_dial = ""
        dial = []
        for _, row in self.dataframe.iterrows():
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
                    print(dial)
                    dial["dialogue"][turn_index]["turn_label"].append([attr_name, vote])
            turn_index += 1
        json.dump(dials, open("dstc2_majority_en.json", "w"))
        return dials

    def apply_snorkel_model(self, slot):
        matrix = []
        relevant_cols = [col for col in self.dataframe.columns if f"{slot}_lf" in col]
        relevant_cols.sort(
                key=lambda x: self.func_order.index(x.split(f"{slot}_lf_")[-1])
        )
        for _, row in self.dataframe.iterrows():
            matrix.append([self.dataframe.at[row.name, c] for c in relevant_cols])
        
        matrix = np.asarray(matrix)
        label_model = LabelModel(cardinality = len(self.ontology[slot]) + 101, verbose=True)
        label_model.fit(L_train=matrix, n_epochs=500, log_freq=100, seed=607)
        res = label_model.predict(matrix)
        for i, pred in enumerate(res):
            if pred > len(self.ontology[slot]):
                self.dataframe.at[i, slot] = -1
            else:
                self.dataframe.at[i, slot] = pred


    def snorkel_vote(self):
        dials = self._file_format()
        self._vote_to_frame(self.vote_to_frame)
        for slot in ["food", "area", "pricerange"]:
            self.apply_snorkel_model(slot)
        turn_index = 0
        current_dial = np.Inf
        dial = []
        for _, row in self.dataframe.iterrows():
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
            fixing_functions):
        # In order to fit a snorkel model, we need to create an
        # actual matrix. Probably best to add new rows to the frame 
        # originally.
        for _, row in self.dataframe.iterrows():
            for lf in labeling_functions:
                col_name = f"{slot}_lf_{lf.__name__}"
                if col_name not in self.dataframe.columns:
                    self.dataframe[col_name] = -1
                vote = lf(row)
                self.dataframe.at[row.name, col_name] = vote

            # Fixing functions : Their vote depends on the vote of the 
            # labeling functions they are fixing, and will vote 
            # the opposite of them. For now, dialogue based.
            # Should fetch the previous votes in the same dialogue. Means 
            # that it should only trigger for the last utterance, correcting 
            # all previous turns.

            all_in_dial = self.dataframe.query(f"dial == {row.dial}")
            if row.name != max(all_in_dial.index.tolist()):
                continue
            
            for _, fix_row in all_in_dial.iterrows():
                for ff in fixing_functions:
                    res, funcs = ff(fix_row)
                    if res == False:
                        continue
                    for affected_func in funcs:
                        col_name = f"{slot}_lf_{affected_func}"
                        fix_col = f"{col_name}_{ff.__name__}"
                        if fix_col not in self.dataframe.columns:
                            self.dataframe[fix_col] = -1
                        if (prev_vote := fix_row[col_name]) > -1:
                            vote = prev_vote + 100
                        else:
                            vote = -1
                        self.dataframe.at[fix_row.name, fix_col] = vote
                        

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

if __name__ == '__main__':
    sc = SnorkelCenter()
    sc.snorkel_vote()
