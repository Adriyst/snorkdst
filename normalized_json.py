import json
import random
from copy import deepcopy

import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk import ngrams
import logging
import torch

import get_data


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_MASTER_DIR = "/usr/local/Development/master"
VECTORS = f"{BASE_MASTER_DIR}/analysis_scripts/model/model.txt"
DEVSET = f"{BASE_MASTER_DIR}/data/devfilenames.txt"

class NormalCorpusJson:

    def __init__(self, data, corpus, **kwargs):
        self.data = data
        self.size = len(data)
        self.corpus = corpus
        self.dialogues = [NormalDialogueJson(t, corpus, **kwargs) for t in data]
        self.ontology = get_data.DataFetcher.fetch_dstc_ontology() if corpus == "dstc" else None


    def to_pandas(self):
        full_data = []
        for dialogue in self.dialogues:
            dial_obj = {
                "dial": dialogue.id,
                "transcription": "",
                "system_transcription": "",
                "real_transcription": "",
                "asr": [],
                "food": -1,
                "area": -1,
                "pricerange": -1
            }
            for turn in dialogue.turns:
                turn_obj = dial_obj.copy()
                turn_obj["transcription"] = turn.asr[0][0]
                turn_obj["system_transcription"] = turn.system_utterance
                turn_obj["real_transcription"] = turn.transcription
                turn_obj["asr"] = turn.asr
                full_data.append(turn_obj)
        return pd.DataFrame(full_data)

    def dialogue_is_last(self, num):
        return num == len(self.dialogues) - 1

    def randomize(self, seed=607):
        random.seed(seed)
        random.shuffle(self.dialogues)

    def restructure_set(self, model):
        for dialogue in self.dialogues:
            for turn in dialogue.turns:
                for slot, val in turn.labels.items():
                    if slot in ("this", "name", "slot"):
                        continue
                    transformed_slot = model.slot_mapping[slot]
                    try:
                        transformed_value = model.ontology[slot].index(val)
                    except ValueError:
                        logger.info(f"valueerror with id {id(self)}:")
                        logger.info(f"for some slot {slot} and some value {val}")
                        transformed_value = model.ontology[slot].index("blank")
                    turn.transformed_labels[transformed_slot] = transformed_value
                for sl in ["food", "pricerange", "area"]:
                    if sl in turn.labels:
                        continue
                    transformed_slot = model.slot_mapping[sl]
                    turn.transformed_labels[transformed_slot] = model.ontology[sl].index("blank")
        logger.info(f"Ending for set {id(self)} with {len(self.dialogues)} dialogues")

    def to_ngram_dataset(self, model, vectors=VECTORS, n_grams=3):
        from gensim.models import KeyedVectors

        self.restructure_set(model)

        vecs = model._vecs
        lower_vocab = {}
        for k, v in vecs.vocab.items():
            if k != k.lower():
                lower_vocab[k.lower()] = v
        vecs.vocab.update(lower_vocab)
        dataset = []

        for dial in self.dialogues:
            new_dial = []
            for turn in dial.turns:
                new_turn = {}
                tokenized_utter = word_tokenize(turn.transcription)
                tokenized_sys = word_tokenize(turn.system_utterance)
                user_grams = []
                tags = []

                sys_idx = torch.LongTensor([
                    vecs.vocab[w].index if w in vecs.vocab else 110221
                    for w in tokenized_sys
                ])

                for idx, cls in turn.tagged_sent:
                    if cls == 0:
                        only_w = tokenized_utter[idx]
                        w_idx = vecs.vocab[only_w].index if only_w in vecs.vocab else 110221
                        tags.append(w_idx)
                    elif cls == 1:
                        tags.append(-1)
                    else:
                        tags.append(-2)

                for n in range(1, n_grams+1):
                    utter_n_grams = ngrams(tokenized_utter, n)
                    utter_gram_list = [[vecs.vocab[w].index if w in vecs.vocab else 110221
                                        for w in gram] for gram in utter_n_grams]

                    user_grams.append(utter_gram_list)

                new_turn["ngrams"] = user_grams
                new_turn["system_utter"] = sys_idx
                new_turn["transcription"] = turn.transcription
                new_turn["tags"] = tags
                sorted_trans = sorted(turn.transformed_labels.items(),
                                      key=lambda x: x[0])
                new_turn["labels"] = torch.LongTensor(
                    [x[1] for x in sorted_trans])
                new_dial.append(new_turn)
            dataset.append((new_dial, dial.id))

        return dataset

    def bootstrap_slots(self):
        fq, in_idx = self.get_label_count()
        tmp_thresh = {
            "pricerange": 500,
            "food": 50,
            "area": 300,
        }
        not_there = []
        
        change_needed = self.get_needed_updates(fq, tmp_thresh)
        check = lambda: sum([len(x) for v in change_needed["add"].values() for x in v])
        finished = False
        slots = list(tmp_thresh.keys())
        max_num = len(slots)-1

        while not finished:
            while check():
                slot_one = random.randint(0,max_num)
                while not change_needed["add"][slots[slot_one]]:
                    slot_one = random.randint(0,max_num)
                first_hit = slots[slot_one]
                first = change_needed["add"][first_hit].pop()
                non_empty_slots = [x for x in change_needed["add"].keys() if
                        change_needed["add"][x] and x != first_hit]
                second = None
                if non_empty_slots:
                    slot_two = random.randint(0,max_num)
                    while slot_two == slot_one or not change_needed["add"][slots[slot_two]]:
                        slot_two = random.randint(0,max_num)
                    second = slots[slot_two]
                    second = change_needed["add"][second].pop()

                
                copy_dialogues = self.dialogues.copy()
                random.shuffle(copy_dialogues)
                if second:
                    covered_dials = [x for x in copy_dialogues if first in
                            [y[1] for y in x.all_labels] and second in [y[1] for y in
                                x.all_labels]]
                    if not covered_dials:
                        covered_dials = [x for x in copy_dialogues if first in
                                [y[1] for y in x.all_labels] or second in [y[1] for y in
                                    x.all_labels]]
                    if not covered_dials:
                        not_there.append(first)
                        not_there.append(second)
                        continue
                else:
                    covered_dials = [x for x in copy_dialogues if first in
                            [y[1] for y in x.all_labels]]
                    if not covered_dials:
                        not_there.append(first)
                        continue
                dial = covered_dials[0]
                dial.id = "%032x" % random.getrandbits(128)
                self.dialogues.append(dial)

            fq, inv_idx = self.get_label_count()
            change_needed = self.get_needed_updates(fq, tmp_thresh)
            for k, val in change_needed["add"].items():
                new_val = val.copy()
                for sl in val:
                    if sl in not_there:
                        new_val.remove(sl)
                change_needed["add"][k] = new_val
            if not check():
                finished = True

        return True

    
    def normalize_classes(self, w=1):
        fq, in_idx = self.get_label_count()
        new_corpus = self.copy()
        tmp_thresh = {
            "pricerange": 500,
            "food": 50,
            "area": 300,
        }

        change_needed = self.get_needed_updates(fq, tmp_thresh)
        dialogue_structures = self.get_label_structures()

        check = lambda: sum([len(x) for v in change_needed["add"].values() for x in v])
        finished = False
        counts = {"nottup": 0, "longtup": 0, "onetup": 0}

        while not finished:
            while check():
                get_new_struct = True
                
                while get_new_struct:
                    random_struct_index = random.randint(0, len(dialogue_structures)-1)
                    struct_to_fill = dialogue_structures[random_struct_index]
                    values_to_use = []
                    duplicate_check = []
                    for struct in struct_to_fill:
                        if isinstance(struct, list):
                            for s in struct:
                                duplicate_check.append(s)
                        else:
                            duplicate_check.append(struct)

                    duplicate_check = nltk.FreqDist(duplicate_check)
                    if duplicate_check.most_common(1)[0][1] <= 2:
                        get_new_struct = False

                for struct in struct_to_fill:
                    if len(struct) == 1 and struct[0] not in tmp_thresh:
                        values_to_use.append(())
                        continue
                    
                    def deal_with_struct(single_struct):
                        struct_len = len(change_needed["add"][single_struct])
                        if single_struct not in change_needed["add"] or struct_len == 0:
                            return None
                        return change_needed["add"][single_struct].pop()

                    if len(struct) > 1:
                        mult_slots = []
                        for s in struct:
                            dealt_with = deal_with_struct(s)
                            if dealt_with:
                                mult_slots.append((s, dealt_with))
                        values_to_use.append(mult_slots)
                    else:
                        struct = struct[0]
                        dealt_with = deal_with_struct(struct)
                        slot_tuple = (struct, dealt_with) if dealt_with else ()
                        values_to_use.append(slot_tuple)


                if values_to_use:
                    new_dialogue = NormalDialogueJson.copy_format(self.corpus)
                    for turn_no,tup in enumerate(values_to_use):
                        if not tup:
                            counts["nottup"] += 1
                            utterance = self.fetch_empty_utterance(turn_no)
                        elif isinstance(tup, list) and len(tup) > 1:
                            counts["longtup"] += 1
                            utterance = self.fetch_utterance_by_multiple_slots(tup)
                            if not utterance:
                                utterance = self.fetch_empty_utterance(turn_no)
                                tup = ()
                        else:
                            if isinstance(tup, list):
                                tup = tup[0]
                            counts["onetup"] += 1
                            utterance = self.fetch_utterance_by_single_slot(in_idx, tup)
                        nfj = NormalFormatJson.copy_format("dstc")
                        tup = [tup] if tup and isinstance(tup, tuple) else ()
                        nfj.labels = dict(tup)
                        nfj.transcription = utterance
                        new_dialogue.turns.append(nfj)
                    new_dialogue.update_labels()
                    self.dialogues.append(new_dialogue)
            fq, in_idx = self.get_label_count()
            change_needed = self.get_needed_updates(fq, tmp_thresh)
            if not check():
                finished = True
        print(counts)
        return fq

    def fetch_utterance_by_single_slot(self, inv_idx, slot):
        # first check if there is a match for the single required slot
        # if not, look for one randomly
        slot_name, slot_value = slot
        possible_idx = inv_idx[slot_value]
        random.shuffle(possible_idx)
        for idx in possible_idx:
            try_dial = self.dialogues[idx]
            for t in try_dial.turns:
                if slot_name in t.labels and t.labels[slot_name] == slot_value and \
                        len(list(t.labels.keys())) == 1:
                    return t.transcription
        dialogue_list = self.dialogues.copy()
        random.shuffle(dialogue_list)
        for dial in dialogue_list:
            for t in dial.turns:
                if slot_name in t.labels and t.labels[slot_name] == slot_value and \
                        len(list(t.labels.keys())) == 1:
                    return t.transcription
        return None


    def fetch_utterance_by_multiple_slots(self, slots):
        slot_names, slot_values = zip(slots)
        dialogue_list = self.dialogues.copy()
        random.shuffle(dialogue_list)
        for dial in dialogue_list:
            for t in dial.turns:
                if sum([1 if s in t.labels else 0 for s in slot_names]) == len(slots) and\
                        len(slots == len(list(t.labels.keys()))) and \
                        sum([1 if t.labels[s] == v else 0 for s,v in slots]) == len(slots):
                    return t.transcription
        return None

    def fetch_empty_utterance(self, turn_no, max_tries=100):
        utterance = ""
        tries = 0
        while not utterance:
            if tries >= max_tries:
                raise ValueError("exceeded max tries")
            rand_idx = random.randint(0, len(self.dialogues)-1)
            rand_dialogue = self.dialogues[rand_idx]
            if len(rand_dialogue.turns) <= turn_no:
                continue
            rand_turn = rand_dialogue.turns[turn_no]
            if not rand_turn.labels:
                return rand_turn.transcription
            tries += 1

    def get_needed_updates(self, fq, tmp_thresh):
        change_needed = {
            "add": {
                "food": [],
                "area": [],
                "pricerange": [],
            },
            "remove": {
                "food": [],
                "area": [],
                "pricerange": [],
            }
        }

        for slotname, subfq in fq.items():
            if slotname not in tmp_thresh:
                continue
            
            sfd = self.clean_slotfreq(subfq, slotname)
            if not sfd:
                continue
            lims = [
                int(tmp_thresh[slotname] * 0.9),
                int(tmp_thresh[slotname] * 1.1)
            ]

            for k,v in sfd.items():
                # k = slot value name 
                # v = slot value count
                if v < lims[0]:
                    change_needed["add"][slotname].append(k)
                elif v > lims[1]:
                    change_needed["remove"][slotname].append(k)
        return change_needed


    def clean_slotfreq(self, slotfreq, slotname):
        if slotname in ("pricerange", "area", "food"):
            del(slotfreq["dontcare"])
            return slotfreq
        return None


    def get_label_structures(self):
        structures = []
        avg_len = sum([len(d) for d in self.dialogues])//len(self.dialogues)
        len_range = [avg_len * .66, avg_len * 1.33]
        for dialogue in self.dialogues:
            if len_range[0] < len(dialogue) < len_range[1]:
                if len(dialogue.turns) < 5:
                    logger.info(len(dialogue))
                slotnames = []
                for turn in dialogue.turns:
                    labs = turn.labels.keys()
                    
                    if labs:
                        labs = list(filter(lambda x: x not in ("this", "name", "slot"), labs))
                        if labs:
                            slotnames.append(labs)
                        else:
                            slotnames.append(["%032x" % random.getrandbits(128)])
                    else:
                        slotnames.append(["%032x" % random.getrandbits(128)])

                structures.append(slotnames)

        return structures


    def get_label_count(self):
        """
        Gets a count for each label value dependent on each slotname, and an inverse
        index that keeps track of which documents a label occurs in.
        """
        fq = nltk.ConditionalFreqDist()
        inverse_index = {}
        for i,t in [(i, turn) for (i, dialogue) in enumerate(self.dialogues) for turn in dialogue.turns]:
            for k,v in t.labels.items():
                if v in inverse_index:
                    if i not in inverse_index[v]:
                        inverse_index[v].append(i)
                else:
                    inverse_index[v] = [i]
                fq[k][v] += 1
        return fq, inverse_index
        

    def copy(self):
        """
        Performs a deep copy on the current dataset.
        """
        copied_corpus = NormalCorpusJson.copy_format(self.corpus)
        for dialogue in self.dialogues:
            copy_list = list(filter(lambda x: x != "turns" and "__" not in x, dir(dialogue)))
            cp_dialogue = NormalDialogueJson.copy_format(self.corpus)
            for dial_attr in copy_list:
                dial_attr_value = getattr(dialogue, dial_attr)
                if not callable(dial_attr_value):
                    setattr(cp_dialogue, dial_attr, dial_attr_value)
            for turn in dialogue.turns:
                cp_turn = NormalFormatJson.copy_format(self.corpus)
                for attribute in dir(turn):
                    attr_value = getattr(turn, attribute)
                    if not callable(attr_value) and "__" not in attribute:
                        setattr(cp_turn, attribute, attr_value)
                cp_dialogue.turns.append(cp_turn)
            copied_corpus.dialogues.append(cp_dialogue)
        copied_corpus.size = len(copied_corpus.dialogues)
        return copied_corpus

    @staticmethod
    def copy_format(corpus):
        return NormalCorpusJson([], corpus)

    def __len__(self):
        return len(self.dialogues)

class NormalDialogueJson:

    def __init__(self, data, *args, **kwargs):
        self.id = data["session-id"] if "session-id" in data else data["dialogue_idx"]
        turn_data = data["turns"] if "turns" in data else data["dialogue"]
        self.length = len(turn_data)
        self.turns = []
        for t in turn_data:
            self.turns.append(NormalFormatJson(t, self, *args, **kwargs))
        self.all_labels = []
        self.goal_labels = {}
        #self.feedback = data["task-information"]["feedback"]


    def update_labels(self):
        for nfj in self.turns: self.all_labels.extend(nfj.labels.items())
        for slot, val in self.all_labels:
            if slot in ["food", "area", "pricerange"]:
                if slot in self.goal_labels:
                    if self.goal_labels[slot] and not val:
                        continue
                self.goal_labels[slot] = val

    def turn_is_last(self, num):
        return num == self.turns[-1].turn_no


    def __len__(self):
        return len(self.turns)


    def copy(self):
        copy = NormalDialogueJson.copy_format("dstc")
        for attr in dir(self):
            if callable(attr) or "__" in attr:
                continue
            setattr(copy, attr, getattr(self, attr))
        return copy

    @staticmethod
    def copy_format(corpus):
        return NormalDialogueJson({
                "session-id": "",
                "turns": [],
                "task-information": {"feedback":""}
            })


class NormalFormatJson:

    def __init__(self, data, dialogue, corpus, parse_labels=False, clean=False):
        self.turn_no = 0
        self.dialogue = dialogue
        self.transcription = ""
        self.labels = {}
        self.previous_labels = {}
        self.conversational_act = ""
        self.extra = {}
        self.id = ""
        self.system_utterance = ""
        self.transformed_labels = {}
        self.tagged_sent = []
        self.system_acts = []
        self.requested_slots = []
        self.confirmed_slots = []
        self.goal_labels = {}
        self.asr = []

        self.parse_labels = parse_labels

        if corpus == "dstc":
            if clean:
                self._clean_dstc(data)
            else:
                self.parse_dstc(data)
        elif corpus == "frames":
            self.parse_frames(data)

    def _clean_dstc(self, dstc_dict):
        self.turn_no = dstc_dict["turn_idx"]
        self.asr = dstc_dict["asr"]
        self.transcription = dstc_dict["transcript"]
        self.goal_labels = {"food": "none", "area": "none", "pricerange": "none"}
        if self.parse_labels:
            if self.turn_no > 0:
                self.goal_labels.update(self.dialogue.turns[self.turn_no-1].goal_labels)
            slotties = dstc_dict["turn_label"]
            for slotnam, slotval in slotties:
                if slotnam not in ("food", "area", "price range"):
                    continue
                if slotnam == "price range":
                    slotnam = "pricerange"
                self.labels[slotnam] = slotval
            self.goal_labels.update(self.labels)
        self.system_utterance = dstc_dict["system_transcript"].lower()


    def parse_dstc(self, dstc_dict):
        self.id = dstc_dict["audio-file"]
        self.turn_no = int(dstc_dict["turn-index"]) if dstc_dict["turn-index"] else 0
        self.transcription = dstc_dict["transcription"]
        tags = []
        if self.parse_labels:
            tokenized = word_tokenize(self.transcription)
            json_part = dstc_dict["semantics"]["json"]
            goal_vals = dstc_dict["goal-labels"]
            for pot_val in ("food", "area", "pricerange"):
                if pot_val not in goal_vals:
                    goal_vals[pot_val] = "none"
            self.goal_labels = goal_vals
            for jsobj in json_part:
                for slot in jsobj["slots"]:
                    sname, sval = slot[0], slot[1]
                    if jsobj["act"] == "inform"\
                       and sname in ("pricerange", "food", "area"):
                        self.requested_slots.append(sname)
                    if sname in tokenized:
                        tags.append((tokenized.index(sname), 1))
                    if sval in tokenized:
                        tags.append((tokenized.index(sval), 2))
                    self.labels[sname] = sval
            tags.extend([(i, 0) for i in range(len(tokenized)) if i not in [x[0] for x in
                tags]])
            self.tagged_sent = sorted(tags, key=lambda x: x[0])
            for dial_act in dstc_dict["system_acts"]:
                for slot in dial_act["slots"]:
                    if dial_act["act"] == "request":
                        self.confirmed_slots.append(("request", slot[1]))
        self.conversational_act = dstc_dict["semantics"]["cam"]\
            if "cam" in dstc_dict["semantics"] else ""
        self.system_utterance = dstc_dict["system_transcription"]
        # TODO:
        # add hypotheses if valuable?


    def parse_frames(self, frames_dict):
        self.turn_no = frames_dict["turn_no"]
        self.transcription = frames_dict["text"]
        lab_obj = frames_dict["labels"]
        active_frame = lab_obj["active_frame"]
        current_frame = list(filter(
            lambda x: x["frame_id"] == active_frame, lab_obj["frames"]))[0]
        slot_list = current_frame["info"].items()
        for slotname, slotval in slot_list:
            assert len(slotval) == 1
            entryval = slotval[0]
            self.labels[slotname] = entryval["val"] if not entryval["negated"] else f"neg|{entryval['val']}"
        self.conversational_act = lab_obj["acts"][0]["name"]
        # TODO
        # extras ? 

    def to_dict(self):
        return {
            "turn_no": self.turn_no,
            "transcription": self.transcription,
            "labels": self.labels,
            "conversational_act": self.conversational_act,
            "extra": self.extra
        }

    def to_json(self):
        return json.dumps(self.to_dict())

    def _str_of_labels(self):
        endlist = []
        for k, v in self.labels.items():
            endlist.append(f"{k}:{v}")
        return "\n".join(endlist)

    def __str__(self):
        return f"Turn number: {self.turn_no}\n"\
                f"Transcription: {self.transcription}\n"\
                "labels:\n"\
                f"{self._str_of_labels()}\n"\
                f"conversational_act: {self.conversational_act}"

    



    @staticmethod
    def copy_format(corpus):
        return NormalFormatJson({
                "audio-file": "",
                "turn-index": "",
                "transcription": "",
                "semantics": {},
                "system_transcription": ""
            }, "dstc")

