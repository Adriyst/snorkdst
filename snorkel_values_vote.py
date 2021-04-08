from snorkel.labeling import labeling_function
from nltk import FreqDist
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


        self.welcome_msg = "hello , welcome to the cambridge restaurant system? "\
        "you can ask for restaurants "\
        "by area , price range or food type . how may i help you?"
        
        # request wording for each slot
        self.request_statements = {
            "food": "what kind of food would you like?",
            "area": "what part of town do you have in mind?",
            "pricerange": "would you like something in the cheap , "\
                    "moderate , or expensive price range?"
        }

        self.sorry_statement = re.compile("^sorry would you like")

        self.confirm_statements = {
            "food": re.compile(
                r"you are looking for a ([a-z\s]*)\s?restaurant (is that |serving any kind of food )?right?"
            ),
            "area": re.compile(
                r"did you say you are looking for a restaurant in (?:the (\w+)|any part) of town?"
            ),
            "pricerange": re.compile(
                r"let me confirm , you are looking for a restaurant in the (\w+) price range right?"
            )
        }

        self.confirm_dontcare_statements = {
            "food": re.compile(
                r"dontmatchanythinghere"
            ),
            "area": re.compile(
                r"ok , a restaurant in any part of town is that right?"
            ),
            "pricerange": re.compile(
                r"let me confirm , you are looking for a restaurant and you dont care"\
                " about the price range right?"
            )
        }

        self.invalids = [
            "unintelligible",
            "sil",
            "static"
        ]

        self.labeling_functions = [
                self.vote_for_slot, self.response_vote, self.confirmation_vote,
                self.dontcare_vote, self.whatabout_vote, self.any_x
        ]
        self.fixing_functions = [
                self.exclude_slot
        ]

        self.decline = re.compile(r"(^|\s)(now?|wrong)($|\s)")
        self.accept = re.compile(r"(^|\s)(right|yea?|correct|yes|yeah)($|\s)")
        self.dontcare = lambda slot: re.compile(rf"(^any(?!thing)( ({slot}|kind|type))?|dont care|do not "\
                "care|dontcare|care|(it )?(doesnt|dont) matter|none|dont know|"\
                "anything(?! else)|what ?ever)")

        self.suggest_option = re.compile(
                r"^(?:(?:\w+\s?){1,6}) is a (?:nice|great) restaurant"\
                        r"(?: in the (\w+) part of town)?"\
                        r"(?: serving ((?:\w+\s?){1,2}) food)?"\
                        r"(?: and it is in the (\w+) price range)?"
                )

        self.exceptions = self.slot_exceptions()
        self.woz_train_votes = self.get_woz_votes("train")

    def any_x(self, slot):

        def dontcare_x(x: pd.Series):
            slot_x = {
                    "food": "(food)",
                    "area": "(area|part)",
                    "pricerange": "(price|price range|pricerange)"
            }[slot]
            return len(self.ontology[slot]) if re.search(f"any {slot_x}", x.transcription) else -1

        return dontcare_x

    def invalid_vote(self, slot):
        
        def vote_invalid(x: pd.Series):
            affected_funcs = [fn.__name__ for fn in
                    self.get_labeling_functions_for_slot(slot)]

            return_T = FixingReturn(True, False, affected_funcs, vote_invalid, x.name)
            return_F = FixingReturn(False, False, [], vote_invalid, x.name)

            return return_T if x.transcription in self.invalids else return_F
           
        return vote_invalid

    def whatabout_vote(self, slot):

        def vote_whatabout(x: pd.Series):

            whatabout_regexp = re.search(
                    r"((what|how) about|is it) \w+",
                    x.transcription
            )
            if not whatabout_regexp:
                return -1

            return self.vote_for_slot(slot)(x)

        return vote_whatabout


    def dontcare_vote(self, slot):
        phrases = {
            "what part of town do you have in mind?": "area"
        }

        def dontcare_value(x: pd.Series):
            dc_vote = len(self.ontology[slot])
            if self.vote_for_slot(slot)(x) > -1:
                return -1
            if self.dontcare(slot).search(x.transcription):
                if x.system_transcription == self.welcome_msg:
                    return dc_vote
                if self.sorry_statement.search(x.system_transcription):
                    if {
                        "food":"food",
                        "area":"of town",
                        "pricerange": "price range"
                    }[slot] in x.transcription:
                        return dc_vote
                    else:
                        return -1
                search_phrase = slot if slot != "pricerange" else "(pricerange|price range)"
                if re.search(rf"{search_phrase}", x.system_transcription) or \
                        (x.system_transcription in phrases and\
                        phrases[x.system_transcription] == slot):
                    return dc_vote
            
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

    def exclude_slot(self, slot: str):

        def exclude_func(x):

            aff_func_names = [
                    self.vote_for_slot(slot).__name__,
                    self.response_vote(slot).__name__,
                    self.confirmation_vote(slot).__name__,
                    self.whatabout_vote(slot).__name__,
                    self.woz_label_func(slot).__name__
            ]
            affected_funcs = [fn.__name__ for fn in 
                    self.get_labeling_functions_for_slot(slot)
                    if fn.__name__ in aff_func_names]

            
            return_T = FixingReturn(True, False, affected_funcs, exclude_func, x.name)
            return_F = FixingReturn(False, False, [], exclude_func, x.name)

            if not (match := self.suggest_option.match(x.system_transcription)):
                return return_F

            area_type, food_type, price_type = match.groups()
            food_split = food_type.split() if food_type else []
            if food_type is not None and\
                    len(food_split) > 2 and\
                    food_split[0] in ("cheap", "moderate", "expensive"):
                        price_type = food_split[0]
            right_val = self._deduct_right_val(slot, [food_type, area_type, price_type])

            if self.accept.search(x.transcription):
                # user confirms that we have the right values
                return return_F
            
            if self.decline.search(x.transcription):
                # some goal label is wrong
                if len([x for x in [area_type, food_type, price_type] if x]) > 1:
                    # multiple slot values, cannot deduct
                    return return_F
                if right_val is not None:
                    # negate slot
                    return return_T 

                return return_F

            if slot in x.transcription or \
                    (slot == "pricerange" and "price range" in x.transcription):
                        if self.vote_for_slot(slot)(x) >= 0:
                            # a value was stated in the slot, no need to interfere
                            return return_F 
                        # a value was specifically requested, return negative
                        # need check for 'type of food' or something

                        return return_T
            return return_F
        return exclude_func

    def find_double_word_value(self, word, transcript, slot):
        double_words = [x.split() for x in self.ontology[slot]
                if len(x.split()) > 1]
        double_words.extend([
            x.split() for y in self.alternatives.values() 
            for x in y if len(x.split()) > 1
        ])
        is_first = [x for x in double_words if word == x[0]]
        whole_transcript = transcript.split()
        possibilities = []
        if len(is_first) > 0:
            for candidate in is_first:
                if (w_idx := whole_transcript.index(word)) == len(whole_transcript) - 1:
                    # last word, no way
                    continue
                if (attempt := whole_transcript[w_idx:w_idx+2]) == candidate:
                    possibilities.append(" ".join(attempt))

        is_second = [x for x in double_words if word == x[1]]
        if len(is_second) > 0:
            for candidate in is_second:
                if (w_idx := whole_transcript.index(word)) == 0:
                    continue
                if (attempt := whole_transcript[w_idx-1:w_idx+1]) == candidate:
                    possibilities.append(" ".join(attempt))
        return possibilities

    def vote_for_slot(self, slot):

        def val_in_text(x: pd.Series):
            if x.transcription in self.invalids:
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
            if self.dontcare(slot).search(x.transcription):
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

            if self.confirm_dontcare_statements[slot].match(x.system_transcription):
                if self.accept.search(x.transcription):
                    return len(self.ontology[slot])

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
            if match in self.ontology[slot]:
                vote = self.ontology[slot].index(match)
            else:
                vote = len(self.ontology[slot])

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

    def get_fixing_functions_for_slot(self, slot):
        return [func(slot) for func in self.fixing_functions]

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

class FixingReturn:
    """
    Class for return type of fixing functions. Contains three attributes:

     - apply (bool): Whether the function was triggered
     - mode (bool): Whether it should indicate positive or negative correlation
                    with chosen functions
     - affected ([fn]): List of voting function that it should correlate with
    """

    def __init__(self, apply, mode, affected, ff, position):
        self.apply = apply
        self.positive = mode
        self.affected = affected
        self.ff = ff
        self.position = position



class SystemStateParser:

    RESTAURANT_STATES = "./restaurantstate.json"
    FIND_RES_NAME = re.compile(r"((?:\w+\s*)+) is a")
    SLOTS = ["food", "area", "pricerange"]

    def __init__(self, df: pd.DataFrame, val_voter: ValueVoter):
        self.df = df
        self.val_voter = val_voter
        self.restaurant_states = json.load(open(self.RESTAURANT_STATES))
        self.system_states = self._split_to_sections()

    def _find_restaurant(self, start=0):
        for turn_idx, turn in self.df.iloc[start:].iterrows():
            if not (hit := self.FIND_RES_NAME.search(turn.system_transcription)):
                continue
            if (rest := hit.groups()[0]) not in self.val_voter.ontology["name"]:
                print("COULD NOT FIND RESTAURANT %s" % rest)
                continue
            return turn_idx, self.restaurant_states[rest]
        return None, None

    def _get_next_start_idx(self, idx):
        return len(self.df) - (max(self.df.index) - idx)

    def _split_to_sections(self):
        first_idx, first_state = self._find_restaurant()
        if not first_idx:
            return {}
        states = {first_idx: first_state}
        start_idx = self._get_next_start_idx(first_idx)
        next_idx, next_state = self._find_restaurant(start=start_idx)
        while next_idx is not None:
            states[next_idx] = next_state
            start_idx = self._get_next_start_idx(next_idx)
            next_idx, next_state = self._find_restaurant(start=start_idx)
        return states

    def find_approx_state(self):
        for state_idx, state in self.system_states.items():
            approx = self._form_approx_state(state_idx - 1)
            for k,v in approx.items():
                if state[k] != v:
                    right_idx = self._get_next_start_idx(state_idx)
                    print(self.df.iloc[0].dial)
                    for _, turn in self.df.iloc[:right_idx].iterrows():
                        print(turn.transcription)
                        print(turn.system_transcription)
                        print()
                    print(state)
                    print(approx)
                    input()
                    


    def _form_approx_state(self, idx):
        end_idx = self._get_next_start_idx(idx)
        approx = {}
        for turn_idx, turn in self.df.iloc[:end_idx].iterrows():
            for slot in self.SLOTS:
                rel_cols = [col for col in self.df.columns if f"{slot}_lf" in col]
                vote_cols = [col for col in rel_cols if turn[col] > -1]
                if not any(vote_cols):
                    continue
                vote_max = FreqDist(turn[vote_cols]).max()
                if vote_max >= len(self.val_voter.ontology[slot]):
                    continue
                ont_item = self.val_voter.ontology[slot][vote_max]
                approx[slot] = ont_item
        return approx




