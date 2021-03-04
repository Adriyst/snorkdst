from snorkel.labeling import labeling_function
import numpy as np
import pandas as pd
import regex as re
import json

import get_data
import bertbst_model

from pylev import levenshtein as distance

VOTE = {
    "VOTE": 1,
    "ABSTAIN": -1,
    "NEGATIVE": 0
}

class ValueVoter:

    WOZ_PATHS = {
        "train": "./woz_train_votes.json"
    }

    def __init__(self, center):
        self.ontology = center.ontology if center else\
                get_data.DataFetcher.fetch_dstc_ontology()["informable"]
        self.alternatives = bertbst_model.SEMANTIC_DICT
        
        # request wording for each slot
        self.request_statements = {
            "food": "what kind of food would you like?",
            "area": "what part of town do you have in mind?",
            "pricerange": "would you like something in the cheap , "\
                    "moderate , or expensive price range?"
        }

        self.confirm_statements = {
            "food": re.compile(
                r"you are looking for a ([a-z\s]*)\s?restaurant (is that |serving any kind of food )?right?"
            ),
            "area": re.compile(
                r"did you say you are looking for a restaurant in the (\w+) of town?"
            ),
            "pricerange": re.compile(
                r"let me confirm , you are looking for a restaurant in the (\w+) price range right?"
            )
        }

        self.invalids = [
            "unintelligible",
            "sil",
            "static"
        ]

        self.labeling_functions = [
                self.vote_for_slot, self.response_vote, self.confirmation_vote,
                self.woz_label_func, self.dontcare_vote
        ]

        self.decline = re.compile(r"(^|\s)(now?|wrong)($|\s)")
        self.accept = re.compile(r"(^|\s)(right|yea|correct|yes|yeah)($|\s)")
        self.dontcare = re.compile(r"(^any$|dont care|do not care|dontcare|care|it doesnt"\
                    " matter|none)")

        self.suggest_option = re.compile(
                r"^(?:(?:\w+\s?){1,6}) is a (?:nice|great) restaurant"\
                        r"(?: in the (\w+) part of town)?"\
                        r"(?: serving ((?:\w+\s?){1,2}) food)?"\
                        r"(?: and it is in the (\w+) price range)?"
                )

        self.exceptions = self.slot_exceptions()
        self.woz_train_votes = self.get_woz_votes("train")


    def dontcare_vote(self, slot):
        phrases = {
            "what part of town do you have in mind?": "area"
        }

        def dontcare_value(x: pd.Series):
            if self.dontcare.search(x.transcription):
                search_phrase = slot if slot != "pricerange" else "(pricerange|price range)"
                if re.search(rf"{search_phrase}", x.system_transcription) or \
                        (x.system_transcription in phrases and\
                        phrases[x.system_transcription] == slot):
                    return len(self.ontology[slot])
            
            return -1

        return dontcare_value

    def woz_label_func(self, slot):
        woz = self.get_woz("train", slot)
        
        def woz_label(x: pd.Series):
            return woz[x.name]

        return woz_label
        

    def get_woz(self, mode, slot):
        if mode == "train":
            return self.woz_train_votes[slot]

        raise NotImplementedError

    def question_not_request(self, line):
        return re.search(r"is it ([a-z]+)", line) is not None

    def exclude_slot(self, slot: str, x: pd.Series) -> bool:
        if not (match := self.suggest_option.match(x.system_transcription)):
            return False

        area_type, food_type, price_type = match.groups()
        food_split = food_type.split() if food_type else []
        if food_type is not None and\
                len(food_split) > 2 and\
                food_split[0] in ("cheap", "moderate", "expensive"):
                    price_type = food_split[0]
        right_val = self._deduct_right_val(slot, [food_type, area_type, price_type])

        if self.accept.search(x.transcription):
            # user confirms that we have the right values
            return False
        
        if self.decline.search(x.transcription):
            # some goal label is wrong
            if len([x for x in [area_type, food_type, price_type] if x]) > 1:
                # multiple slot values, cannot deduct
                return False
            if right_val is not None:
                # negate slot
                return True

            return False

        if slot in x.transcription or \
                (slot == "pricerange" and "price range" in x.transcription):
                    if self.vote_for_slot(slot)(x) >= 0:
                        # a value was stated in the slot, no need to interfere
                        return False
                    # a value was specifically requested, return negative
                    return True
        return False

    def find_double_word_value(self, word, transcript, slot):
        double_words = [x.split() for x in self.ontology[slot]
                if len(x.split()) > 1]
        is_first = [x for x in double_words if word == x[0]]
        whole_transcript = transcript.split()
        possibilities = []
        if len(is_first) > 0:
            for candidate in is_first:
                if (w_idx := whole_transcript.index(word)) == len(whole_transcript) - 1:
                    # last word, no way
                    continue
                if (attempt := whole_transcript[w_idx:w_idx+1]) == candidate:
                    possibilities.append(" ".join(attempt))
                    continue

        is_second = [x for x in double_words if word == x[1]]
        if len(is_second) > 0:
            for candidate in is_second:
                if (w_idx := whole_transcript.index(word)) == 0:
                    continue
                if (attempt := whole_transcript[w_idx-1:w_idx]) == candidate:
                    possibilities.append(" ".join(attempt))
                    continue
        return possibilities

    def vote_for_slot(self, slot):

        def val_in_text(x: pd.Series):
            if x.transcription in self.invalids or \
                    self.question_not_request(x.transcription):
                        return -1

            words = x.transcription.split()
            for word_idx, word in enumerate(words):
                if word in self.ontology[slot]:
                    if word in self.exceptions[slot] and\
                            word_idx < len(words) - 1 and\
                            words[word_idx+1] in self.exceptions[slot][word]:
                            continue
                    vote = self.ontology[slot].index(word)
                    return vote
                for candidate in [
                        word, *self.find_double_word_value(word, x.transcription, slot)
                ]:
                    for val, alt_vals in self.alternatives.items():
                        if val in self.ontology[slot] and candidate in alt_vals:
                            vote = self.ontology[slot].index(val)
                            return vote 

            
            return -1

        return val_in_text

    def response_vote(self, slot):

        def val_in_response(x: pd.Series):
            if x.transcription in self.invalids or \
                    self.question_not_request(x.transcription):
                        return -1
            if x.system_transcription != self.request_statements[slot]:
                return -1
            if self.dontcare.search(x.transcription):
                return len(self.ontology[slot])
            if self.decline.search(x.transcription):
                return -1
            if (attempt := self.vote_for_slot(slot)(x)) > 0:
                return attempt
            return -1

        return val_in_response

    def confirmation_vote(self, slot):

        def val_in_request(x: pd.Series):
            if x.transcription in self.invalids or \
                    self.question_not_request(x.transcription):
                        return -1

            statement = self.confirm_statements[slot].search(x.system_transcription)
            if not statement or x.transcription in self.invalids:
                return -1
            match_groups = statement.groups()
            if match_groups[-1] is not None:
                if match_groups[-1].strip() == "serving any kind of food":
                    if self.accept.search(x.transcription):
                        vote = len(self.ontology[slot])
                        return vote
                    
                    return self.accept_or_decline_val(x, slot, "")
            match = match_groups[0].strip()
            if len(match) == 0:
                return -1
            return self.accept_or_decline_val(x, slot, match)
            

        return val_in_request

    def accept_or_decline_val(self, x, slot, match):
        if self.decline.search(x.transcription):
            if (other_val := self.vote_for_slot(slot)(x)) > 0 and other_val != match:
                return other_val
            return -1
        if self.accept.search(x.transcription):
            vote = self.ontology[slot].index(match)
            return vote
        return -1

    def slot_exceptions(self):
        return {
                "area": {
                    "north": [
                        "african",
                        "american",
                        "indian"
                    ],
                    "northern": [
                        "european"
                    ],
                    "middle": [
                        "eastern"
                    ],
                    "eastern": [
                        "european"
                    ],
                    "south": [
                        "african",
                        "indian"
                    ]
                },
                "food": {},
                "pricerange": {}
    }

    def get_labeling_functions_for_slot(self, slot):
        return [func(slot) for func in self.labeling_functions]

    def _deduct_right_val(self, slot, vals):
        return {
            "food": vals[0],
            "area": vals[1],
            "pricerange": vals[2]
        }[slot]

    def get_exclusion_for_dial(self, slot, frame):
        return any([
            self.exclude_slot(slot, row) for _, row in frame.iterrows()
        ])

    def get_woz_votes(self, mode):
        voteset = json.load(open(self.WOZ_PATHS[mode]))
        votes = {
            "food": [],
            "area": [],
            "pricerange": []
        }
        for turn in voteset:
            for slotname, slotval in turn.items():
                if slotname == "price range":
                    slotname = "pricerange"
                if slotval == "":
                    votes[slotname].append(-1)
                    continue
                if slotval == "dontcare":
                    votes[slotname].append(len(self.ontology[slotname]))
                    continue
                if slotval in self.ontology[slotname]:
                    votes[slotname].append(
                            self.ontology[slotname].index(slotval)
                    )
                    continue
                added = False
                for alt, alt_vals in self.alternatives.items():
                    if slotval in alt_vals and alt in self.ontology[slotname]:
                        added = True
                        votes[slotname].append(
                                self.ontology[slotname].index(alt)
                        )
                        break
                if not added:
                    votes[slotname].append(-1)
        return votes

