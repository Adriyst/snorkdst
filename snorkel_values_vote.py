from snorkel.labeling import labeling_function
from nltk import FreqDist
import numpy as np
import pandas as pd
import regex as re
import json

import get_data
import bertbst_model

from pylev import levenshtein as distance

ASR_THRESHOLD = .15

class ValueVoter:

    WOZ_PATHS = {
        "train": "./woz_train_votes.json"
    }

    def __init__(self, center):
        self.ontology = center.ontology if center else\
                get_data.DataFetcher.fetch_dstc_ontology()["informable"]
        self.alternatives = bertbst_model.SEMANTIC_DICT
        self.center = center


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
                self.dontcare_vote, self.whatabout_vote, self.any_x, self.slot_support
        ]
        self.fixing_functions = [
                #self.exclude_slot
        ]

        self.decline = re.compile(r"(^|\s)(now?|wrong)($|\s)")
        self.accept = re.compile(r"(^|\s)(right|yea?|correct|yes|yeah)($|\s)")
        self.dontcare = lambda slot: re.compile(rf"(^any(?!thing)( ({slot}|kind|type))?|dont care|do not "\
                "care|dontcare|care|(it )?(doesn'?t|dont) matter|none|dont know|"\
                "anything(?! else)|what ?ever)")

        self.suggest_option = re.compile(
                r"^(?:(?:\w+\s?){1,6}) is a (?:nice|great) restaurant"\
                        r"(?: in the (\w+) part of town)?"\
                        r"(?: serving ((?:\w+\s?){1,2}) food)?"\
                        r"(?: and it is in the (\w+) price range)?"
                )

        self.exceptions = self.slot_exceptions()
        self.woz_train_votes = self.get_woz_votes("train")

        self.support_regexes = {
            "food": re.compile(r"serving ((?:\w+\s?){1,2}) food"),
            "area": re.compile(r"(?:in )?(\w+) (?:of town|part)"),
            "pricerange": re.compile(r"(?:(\w+) restaurant|in (the)? (\w+) "
                "(?:price|pricerange|price range))")
        }


    def check_asr(self, x, func, real=False, **kwargs):
        if real:
            return -1
        for asr in x.asr[1:]:
            transc, score = asr
            if score < ASR_THRESHOLD:
                return -1
            xcop = x.copy()
            xcop.transcription = transc
            if (asr_try := func(xcop, **kwargs)) > -1:
                return asr_try
        return -1

    def slot_support(self, slot):

        def support_val(x: pd.Series, real=False):
            transc = x.transcription if not real else x.real_transcription
            if not (match := self.support_regexes[slot].search(transc)):
                return -1

            if (hit := match.groups()[0]) not in self.ontology[slot]:
                if not isinstance(hit, str):
                    return -1
                if self.dontcare(slot).match(hit):
                    return len(self.ontology[slot])
                return -1

            words = transc.split()
            for word_idx, word in enumerate(words):
                if word in self.ontology[slot]:
                    if word in self.exceptions[slot] and\
                            word_idx < len(words) - 1 and\
                            words[word_idx+1] in self.exceptions[slot][word]:
                            continue
                    vote = self.ontology[slot].index(word)
                    return vote
                for candidate in [
                        word, *self.find_double_word_value(word, transc, slot)
                ]:
                    for val, alt_vals in self.alternatives.items():
                        if val in self.ontology[slot] and candidate in alt_vals:
                            vote = self.ontology[slot].index(val)
                            return vote 

            return -1

        return support_val

    def any_x(self, slot):

        def dontcare_x(x: pd.Series, real=False, cont=True):
            transc = x.transcription if not real else x.real_transcription
            slot_x = {
                    "food": r"any (food)",
                    "area": r"any (area|part)",
                    "pricerange": r"any (price|price range|pricerange)"
            }[slot]
            if not re.search(slot_x, transc):
                if not cont:
                    return -1

                return self.check_asr(x, dontcare_x, real=real, cont=False)
            return len(self.ontology[slot])

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

        def vote_whatabout(x: pd.Series, real=False, cont=True):
            transc = x.transcription if not real else x.real_transcription

            whatabout_regexp = re.search(
                    r"((what|how) about|is it) \w+",
                    transc
            )
            if not whatabout_regexp:
                if not cont: 
                    return -1

                return self.check_asr(x, vote_whatabout, real=real, cont=False)


            return self.vote_for_slot(slot)(x, real=real)

        return vote_whatabout


    def dontcare_vote(self, slot):
        phrases = {
            "what part of town do you have in mind?": "area"
        }

        def dontcare_value(x: pd.Series, real=False, cont=True):
            transc = x.transcription if not real else x.real_transcription
            dc_vote = len(self.ontology[slot])
            if self.vote_for_slot(slot)(x, real=real) > -1:
                return -1
            if self.dontcare(slot).search(transc):
                if x.system_transcription == self.welcome_msg:
                    return dc_vote
                if self.sorry_statement.search(x.system_transcription):
                    if {
                        "food":"food",
                        "area":"of town",
                        "pricerange": "price range"
                    }[slot] in transc:
                        return dc_vote
                    else:
                        return -1
                search_phrase = slot if slot != "pricerange" else "(pricerange|price range)"
                if re.search(rf"{search_phrase}", x.system_transcription) or \
                        (x.system_transcription in phrases and\
                        phrases[x.system_transcription] == slot):
                    return dc_vote
            
            if not cont:
                return -1

            return self.check_asr(x, dontcare_value, real=real, cont=False)

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

        def exclude_func(x, real=False):

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
                        if self.vote_for_slot(slot)(x, real=real) >= 0:
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

    def vote_for_slot(self, slot, threshold=ASR_THRESHOLD):

        def val_in_text(x: pd.Series, real=False):

            def resolve(transc):
                if transc in self.invalids:
                    return -1

                words = transc.split()
                for word_idx, word in enumerate(words):
                    if word in self.ontology[slot]:
                        if word in self.exceptions[slot] and\
                                word_idx < len(words) - 1 and\
                                words[word_idx+1] in self.exceptions[slot][word]:
                                continue
                        vote = self.ontology[slot].index(word)
                        return vote
                    for candidate in [
                            word, *self.find_double_word_value(word, transc, slot)
                    ]:
                        for val, alt_vals in self.alternatives.items():
                            if val in self.ontology[slot] and candidate in alt_vals:
                                vote = self.ontology[slot].index(val)
                                return vote 

                return -1

            if real:
                return resolve(x.real_transcription)

            candidates = {}
            for asr in x.asr:
                transc, score = asr
                if score < threshold:
                    break
                result = resolve(transc)
                if (result := resolve(transc)) > -1:
                    if result not in candidates:
                        candidates[result] = 0
                    candidates[result] += score

            if len(candidates) == 0:
                return -1
            ordered = list(sorted(candidates.keys(), reverse=True,
                key=lambda x: candidates[x]))
            return ordered[0]
            


        return val_in_text

    def response_vote(self, slot):

        def val_in_response(x: pd.Series, real=False):

            def try_asr():
                if real:
                    return -1
                for asr in x.asr[1:5]:
                    xcop = x.copy()
                    xcop.transcription = asr[0]
                    if (asr_try := self.vote_for_slot(slot)(xcop)) > -1:
                        return asr_try
                return -1

            transc = x.transcription if not real else x.real_transcription
            if transc in self.invalids or \
                    self.question_not_request(transc):
                        return -1
            if x.system_transcription != self.request_statements[slot]:
                return -1
            if self.dontcare(slot).search(transc):
                return len(self.ontology[slot])
            if self.decline.search(transc):
                return -1
            if (attempt := self.vote_for_slot(slot)(x, real=real)) > 0:
                return attempt
            return try_asr()

        return val_in_response

    def confirmation_vote(self, slot):

        def val_in_request(x: pd.Series, real=False, cont=True):
            transc = x.transcription if not real else x.real_transcription
            if transc in self.invalids or \
                    self.question_not_request(transc):
                        return -1

            if self.confirm_dontcare_statements[slot].match(x.system_transcription):
                if self.accept.search(transc):
                    return len(self.ontology[slot])

            statement = self.confirm_statements[slot].search(x.system_transcription)
            if not statement or transc in self.invalids:
                return -1
            match_groups = statement.groups()
            if match_groups[-1] is not None:
                if match_groups[-1].strip() == "serving any kind of food":
                    if self.accept.search(transc):
                        vote = len(self.ontology[slot])
                        return vote
                    return self.accept_or_decline_val(x, slot, "")
            match = match_groups[0].strip()
            if len(match) == 0:
                if real or not cont:
                    return -1

                for asr in x.asr:
                    xcop = x.copy()
                    xcop.transcription = asr[0]
                    new_attempt = val_in_request(xcop, cont=False)
                    if new_attempt > -1:
                        return new_attempt
                return -1

            if real or not cont:
                return self.accept_or_decline_val(x, slot, match)


            for asr in x.asr:
                xcop = x.copy()
                xcop.transcription = asr[0]
                new_attempt = val_in_request(xcop, cont=False)
                if new_attempt > -1:
                    return new_attempt

            return self.accept_or_decline_val(x, slot, match)
            

        return val_in_request

    def accept_or_decline_val(self, x, slot, match, real=False):
        transc = x.transcription if not real else x.real_transcription
        if self.decline.search(transc):
            if (other_val := self.vote_for_slot(slot)(x, real=real)) > 0 and other_val != match:
                return other_val
            return -1
        if self.accept.search(transc):
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
            self.exclude_slot(slot)(row) for _, row in frame.iterrows()
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


class SnorkelDialogueVoter:

    threshold = ASR_THRESHOLD

    def __init__(self, center):
        self.center = center
        self.functions = [
            self.find_missing_val, self.keyword_found_no_val
        ]

        self.price_functions = [
            self.budget_hit
        ]
        self.function_map = {
            "food": [],
            "area": [],
            "pricerange": self.price_functions
        }

    def get_functions(self, slot):
        funcs = [fn(slot) for fn in self.functions]
        funcs.extend(self.function_map[slot])
        return funcs
    
    def budget_hit(self, x: pd.Series, real=False):
        if real:
            return -1
        slot = "pricerange"

        curr_preds = self.resolve(x, slot)

        def find_keyword(transc):
            if "budget" not in transc:
                return []
            
            dial_df = self.center.dataframe.query(f"dial == '{x.dial}'")
            row_idx = x.name - min(dial_df.index)
            candidates = []
            for turn_idx, turn in dial_df.iloc[row_idx+1:].iterrows():
                if len(candidates) > 0:
                    break
                if len((pred := self.resolve(turn, slot))) > 0:
                    majority = self.get_majority_from_candidates(pred, slot)
                    if len(majority) > 0:
                        candidates.append((turn_idx, majority[1]))

            return candidates

        for asr in x.asr:
            transc, score = asr
            if score < self.threshold:
                break
            if len(cand := find_keyword(transc)) > -1:
                return cand
        return []


    def keyword_found_no_val(self, slot, real=False):

        def keyword_found(x: pd.Series, real=False):
            found = {
                "food": re.compile(r"(serving \w+| \w+ food)"),
                "area": re.compile(r"(\w+ of town|\w+ area)"),
                "pricerange": re.compile(r"(in the )?\w+ price ?range")
                }[slot]

            curr_preds = self.resolve(x, slot)
            if len(curr_preds) > 0:
                return []

            def check_for_transc(transc):
                if not found.search(transc):
                    return []
                # find first occuring slot value for slot
                dial_df = self.center.dataframe.query(f"dial == '{x.dial}'")
                row_idx = x.name - min(dial_df.index)
                candidates = []
                for turn_idx, turn in dial_df.iloc[row_idx+1:].iterrows():
                    if len(candidates) > 0:
                        break
                    if len((pred := self.resolve(turn, slot))) > 0:
                        majority = self.get_majority_from_candidates(pred, slot)
                        if len(majority) > 0:
                            candidates.append((turn_idx, majority[1]))
                return candidates

            if real:
                return check_for_transc(x.real_transcription)

            for asr in x.asr:
                transc, score = asr
                if score < self.threshold:
                    break
                if len(cand := check_for_transc(transc)) > 0:
                    return cand

            return []

        return keyword_found
    
    def find_missing_val(self, slot):
        
        def find_example(x: pd.Series, real=False):
            if real:
                return []
            dial_df = self.center.dataframe.query(f"dial == '{x.dial}'")
            if x.name == min(dial_df.index):
                return []

            row_idx = x.name - min(dial_df.index)
            any_vote = self.resolve(x, slot)
            if len(any_vote) == 0  or (len(any_vote) > 1 and 
                    len(set([r[1] for r in any_vote])) > 1):
                return []
            if any_vote[0][1] == len(self.center.value_voter.ontology[slot]):
                return []

            returns = []
            for _, row in dial_df.iloc[:row_idx].iterrows():
                candidates = FreqDist()
                any_cast = self.resolve(row, slot)
                if len(any_cast) > 0:
                    # dont cast vote if another value has been cast
                    return []
                for asr in x.asr[1:]:
                    for w in asr[0].split():
                        if len(w) < 4:
                            continue
                        if w in row.transcription:
                            candidates[w] += 1
                if len(candidates) == 0:
                    continue
                returns.append((row.name, any_vote[0][1]))

            return returns

        return find_example 

    def resolve(self, x: pd.Series, slot: str):
        """
        For a turn, find out if it has any votes for a given slot
        """
        rel_cols = [col for col in self.center.dataframe if f"{slot}_lf" in col]
        return [(col, x[col]) for col in rel_cols if x[col] > -1]
    
    def get_majority_from_candidates(self, cands, slot):
        """
        cands should be of the form [(column_name, prediction_value)].

        returns:
            top (column_name, prediction_value) by count
        """
        if len(cands) == 1:
            return cands[0]
        two_most_com = sorted(cands, reverse=True, key=lambda x: x[1])
        vals = list(set([x[1] for x in two_most_com]))
        if len(vals) == 1:
            return two_most_com[0]
        elif vals[0] > vals[1]:
            return two_most_com[0]
        elif vals[0] > len(self.center.value_voter.ontology[slot]) or \
                vals[1] > len(self.center.value_voter.ontology[slot]):
            return []
        return two_most_com[0]



class SnorkelFixingVoter:

    def __init__(self, center):
        self.center = center
        self.fixing_functions = [
                self.vote_no_last
        ]

    def get_functions(self, slot):
        return [fn(slot) for fn in self.fixing_functions]

    def vote_no_last(self, slot):
        """
        Will vote against any votes cast in the final turn. Returns a list of tuples of
        type (COL_TO_VOTE_AGAINST, VOTE_VALUE)
        """

        def last_vote_no(x: pd.Series, real=False):
            if real:
                return -1
            dial_df = self.center.dataframe.query(f"dial == '{x.dial}'")
            if x.name == max(dial_df.index):
                rel_cols = [col for col in dial_df.columns if f"{slot}_lf_" in col
                        and "last_vote_no" not in col]
                for col in rel_cols:
                    if x[col] > -1:
                        return (col, x[col] + 100)
            return -1

        return last_vote_no





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




