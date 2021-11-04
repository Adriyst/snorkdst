import torch
from torch import nn
from torch.optim import Adam
from torch.cuda import is_available
from transformers import BertTokenizer, BertModel

from nltk import FreqDist, ConditionalFreqDist
import random
import math
from copy import deepcopy
import logging
import json
import os 
import numpy as np
from tqdm import tqdm
from frozendict import frozendict
import re

import dataset_dstc2
import tokenization
from dst_util import (
        tokenize_text_and_label, get_token_label_ids, get_start_end_pos,
        get_bert_input
)
from get_data import DataFetcher
from config import CONFIG
from metric_bert_dst import get_joint_slot_correctness

logger = logging.getLogger(__name__)

random.seed(609)

SEMANTIC_DICT = {
  'centre': ['center', 'downtown', 'central', 'down town', 'middle'],
  'south': ['southern', 'southside'],
  'north': ['northern', 'uptown', 'northside'],
  'west': ['western', 'westside'],
  'east': ['eastern', 'eastside'],

  'cheap': ['low price', 'inexpensive', 'cheaper', 'low priced', 'affordable',
            'nothing too expensive', 'without costing a fortune', 'cheapest',
            'good deals', 'low prices', 'afford', 'on a budget', 'fair prices',
            'less expensive', 'cheapeast', 'not cost an arm and a leg'],
  'moderate': ['moderately', 'medium priced', 'medium price', 'fair price',
               'fair prices', 'reasonable', 'reasonably priced', 'mid price',
               'fairly priced', 'not outrageous','not too expensive',
               'on a budget', 'mid range', 'reasonable priced', 'less expensive',
               'not too pricey', 'nothing too expensive', 'nothing cheap',
               'not overpriced', 'medium', 'inexpensive'],
  'expensive': ['high priced', 'high end', 'high class', 'high quality',
                'fancy', 'upscale', 'nice', 'fine dining', 'expensively priced'],

  'afghan': ['afghanistan'],
  'african': ['africa'],
  'asian oriental': ['asian', 'oriental'],
  'australasian': ['australian asian', 'austral asian'],
  'australian': ['aussie'],
  'barbeque': ['barbecue', 'bbq'],
  'basque': ['bask'],
  'belgian': ['belgium'],
  'british': ['cotto'],
  'canapes': ['canopy', 'canape', 'canap'],
  'catalan': ['catalonian'],
  'corsican': ['corsica'],
  'crossover': ['cross over', 'over'],
  'eritrean': ['eirtrean'],
  'gastropub': ['gastro pub', 'gastro', 'gastropubs'],
  'hungarian': ['goulash'],
  'indian': ['india', 'indians', 'nirala'],
  'international': ['all types of food'],
  'italian': ['prezzo'],
  'jamaican': ['jamaica'],
  'japanese': ['sushi', 'beni hana'],
  'korean': ['korea'],
  'lebanese': ['lebanse'],
  'north american': ['american', 'hamburger'],
  'portuguese': ['portugese'],
  'seafood': ['sea food', 'shellfish', 'fish'],
  'singaporean': ['singapore'],
  'steakhouse': ['steak house', 'steak'],
  'thai': ['thailand', 'bangkok'],
  'traditional': ['old fashioned', 'plain'],
  'turkish': ['turkey'],
  'unusual': ['unique and strange'],
  'venetian': ['vanessa'],
  'vietnamese': ['vietnam', 'thanh binh'],
}

class PrePrepped:

    SLOTS = ["area", "food", "price range"]
    
    def __init__(self):
        self.train_set = None
        self.validate_set = None
        self.test_set = None
        self.majority_set = None
        self.majority_pattern_matching_set = None
        self.majority_test_set = None
        self.majority_validate_set = None
        self.snorkel_set = None
        self.snorkel_pattern_matching_set = None
        self.snorkel_test_set = None
        self.snorkel_validate_set = None

    def compare_sets(*args) -> ({str:[int]}, [int]):
        return BertNet.report_predict(*args)

    def generate_comparison(self, version) -> {str: [int]}:
        """
        Compare a labeled set to the gold labels.
        version should be some iteration of x_y, where
        x is the aggregation method - "test", "majority", "snorkel"
        y is the set - "train", "validation", "snorkel"
        """
        gold_version = "test" if "test" in version else "validate"
        gold_set = self.fetch_set(gold_version, use_asr_hyp=1, exclude_unpointable=False)
        comp_set = self.fetch_set(version, use_asr_hyp=1, exclude_unpointable=False)
        preds = {"food": [], "area": [], "price range": []}
        for dial in tqdm(set(["-".join(x.guid.split("-")[:-1]) for x in gold_set])):
            for comp_turn, gold_turn in zip(
                self.get_turns_for_guid(dial, comp_set),
                self.get_turns_for_guid(dial, gold_set)
            ):
                for slot in preds:
                    preds[slot].append(
                            self.get_pred_of_turn(gold_turn, comp_turn, slot)
                    )
        return preds

    @staticmethod
    def get_pred_of_turn(gold_turn, comp_turn, slot):
        return {
            "guid": gold_turn.guid,
            "slot": slot,
            "class_label_id": BertNet._dst_slot_map()[gold_turn.class_label[slot]],
            "class_prediction": PrePrepped.resolve_slotval(comp_turn, slot),
            "start_pos": PrePrepped.resolve_pos(
                min, gold_turn.text_a_label[slot], gold_turn.text_b_label[slot]
            ),
            "start_prediction": PrePrepped.resolve_pos(
                min, comp_turn.text_a_label[slot], comp_turn.text_b_label[slot]
            ),
            "end_pos": PrePrepped.resolve_pos(
                max, gold_turn.text_a_label[slot], gold_turn.text_b_label[slot]
            ),
            "end_prediction": PrePrepped.resolve_pos(
                max, comp_turn.text_a_label[slot], comp_turn.text_b_label[slot]
            )
        }

    @staticmethod
    def resolve_pos(fn, *args):
        try:
            return fn(PrePrepped._resolve_pos(*args))
        except ValueError:
            return 0

    @staticmethod
    def _resolve_pos(a_preds: [int], b_preds: [int]) -> int:
        to_use = a_preds if sum(a_preds) > 0 else b_preds
        to_use = np.asarray(to_use)
        return np.where(to_use == 1)[0]

    @staticmethod
    def resolve_slotval(turn, slot):
        pred = BertNet._dst_slot_map()[turn.class_label[slot]]
        return pred if pred < 3 else 0


    @staticmethod
    def get_turns_for_guid(guid, turns):
        """
        guid: x-n where x denotes set, n denotes dial idx
        turns: set from which turns should be got
        """
        return [t for t in turns if "-".join(t.guid.split("-")[:-1]) == guid]

    def fetch_set(self, version, **kwargs):
        loaded_map = {
            "train": self.train_set,
            "validate": self.validate_set,
            "test": self.test_set,
            "majority": self.majority_set,
            "majority_pattern_matching": self.majority_pattern_matching_set,
            "majority_test": self.majority_test_set,
            "majority_validate": self.majority_validate_set,
            "snorkel": self.snorkel_set,
            "snorkel_pattern_matching": self.snorkel_pattern_matching_set,
            "snorkel_test": self.snorkel_test_set,
            "snorkel_validate": self.snorkel_validate_set
        }
        relevant_set = loaded_map[version]
        if relevant_set:
            return relevant_set

        mode_map = {
            "train": "train",
            "validate": "dev",
            "test": "test",
            "majority_test": "test",
            "snorkel_test": "test",
            "majority_validate": "dev",
            "snorkel_validate": "dev"
        }
        version_type = mode_map[version] if version in mode_map else mode_map["train"]
        loaded_set = dataset_dstc2.create_examples(
            os.path.join(CONFIG["MODEL"]["DATASETPATH"], f"dstc2_{version}_en.json"),
                self.SLOTS, version_type, **kwargs
        )
        grouped_set = self._group_set(loaded_set)
        loaded_map[version] = grouped_set
        return grouped_set

    def _group_set(self, example_set):
        """
        Group the dataset into representations of multiple ASR in single turns.
        """
        grouping = {}
        all_ex = []
        curr_guid = ""
        for ex in example_set:
            if ex.guid not in grouping:
                grouping[ex.guid] = []
                if len(curr_guid) > 0:
                    all_ex.append(GroupedFeatures(grouping[curr_guid]))
                curr_guid = ex.guid
            grouping[ex.guid].append(ex)
        return all_ex 
        

class GroupedFeatures:

    def __init__(self, group):
        self.group = sorted(group, reverse=True, key=lambda x: x.asr_score)
        self.valid = len(self.group) > 0
        if self.valid:
            self.guid = self.group[0].guid
            self.text_a = self.group[0].text_a
            self.text_a_label = self.group[0].text_a_label
            self.class_label = self.group[0].class_label
            self.asrs = [x.asr_score for x in self.group]
            self.text_b, self.text_b_label, self.text_idx = self.find_pointed_asr(self.asrs)
            self.all_texts = [x.text_b for x in self.group]
            self.session_id = self.group[0].session_id


    def find_pointed_asr(self, asrs):
        for i, ex in enumerate(self.group):
            if asrs[i] < CONFIG["MODEL"]["ASR_THRESHOLD"] or i > CONFIG["MODEL"]["ASR_HYPS"]:
                break
            for idx_labeling in (ex.text_b_label, ex.text_a_label):
                if any(idx_labeling):
                    return ex.text_b, ex.text_b_label, i
        return (self.group[0].text_b, self.group[0].text_b_label, 0) \
                if len(self.group) > 0 else ([], [], 0)



class DataLoader:

    UNK_TOKEN = "[UNK]"

    def __init__(self):
        self.tokenizer = tokenization.FullTokenizer(
                vocab_file = CONFIG["MODEL"]["BERT_VOCAB_LOC"],
                do_lower_case=True
        )

    def fetch_batch(self, dataset, bsize, *args, **kwargs):
        for batch in range(0, len(dataset), bsize):
            if batch + bsize > len(dataset):
                yield self.produce_batchset(dataset[batch:], *args, **kwargs)
            else:
                yield self.produce_batchset(dataset[batch:batch+bsize], *args, **kwargs)
        

    def get_tokens_and_labels(self, turn, slot, skip_vals=False, **kwargs):
        tokens_a, token_labels_a = tokenize_text_and_label(
                turn.text_a, turn.text_a_label, slot, self.tokenizer, **kwargs
        )
        tokens_b, token_labels_b = tokenize_text_and_label(
                turn.text_b, turn.text_b_label, slot, self.tokenizer, **kwargs
        )
        token_label_ids = get_token_label_ids(
                token_labels_a, token_labels_b, CONFIG["MODEL"]["SENT_MAX_LEN"]
        )

        if len(tokens_b) == 0:
            tokens_b.append("[UNK]")

        if skip_vals:
            return (tokens_a, tokens_b)
    
        startvals, endvals = get_start_end_pos(
            turn.class_label[slot], token_label_ids, CONFIG["MODEL"]["SENT_MAX_LEN"]
        )

        
        return (tokens_a, tokens_b, startvals, endvals)


    def produce_batchset(self, turns, slots, bert_tokenizer, pred=False, **kwargs):
        input_ids, masks, types, labels = [], [], [], []
        food_labels, area_labels, price_labels = [], [], []
        text_indexes = []
        starts = {k: [] for k in slots}
        ends = deepcopy(starts)
        class_types = {
                "none": 0,
                "dontcare": 1,
                "copy_value": 2,
                "unpointable": 3
        }
        turn_based_bert = {}
        for turn in turns:
            class_label_id_dict = {}
            for slot in ["food", "area", "price range"]:
                class_label_id_dict[slot] = class_types[turn.class_label[slot]]
                tokens_a, tokens_b, startvals, endvals = self.get_tokens_and_labels(turn,
                        slot, **kwargs)
                if slot == "price range":
                    slot = "pricerange"
                starts[slot].append(startvals)
                ends[slot].append(endvals)
            
            word_idxs, inp_ids, mask, seg_ids = get_bert_input(tokens_a, tokens_b, 
                    CONFIG["MODEL"]["SENT_MAX_LEN"], self.tokenizer)
            input_ids.append(inp_ids)
            masks.append(mask)
            types.append(seg_ids)

            text_indexes.append(turn.text_idx)

            for lab, lablist in zip(class_label_id_dict.values(), 
                    [food_labels, area_labels, price_labels]):
                lablist.append(lab)

            labels.append(turn.class_label)
            turn_based_bert[turn.guid] = self._sort_asr(turn, **kwargs)

        return FeatureBatchSet(input_ids, masks, types, starts, ends, food_labels,
                               area_labels, price_labels, turn_based_bert, text_indexes, pred=pred)

    def _sort_asr(self, turn, **kwargs):
        all_feats = {"input_ids": [], "attention_mask": [], "token_type_ids": []}
        all_asrs = []
        real_text_b = turn.text_b
        for txt, asr in zip(turn.all_texts, turn.asrs):
            if len(txt) == 0:
                txt.append(self.UNK_TOKEN)
            turn.text_b = txt
            tokens_a, tokens_b = self.get_tokens_and_labels(
                turn, "food", skip_vals=True, **kwargs)
            _, inp_ids, mask, seg_ids = get_bert_input(tokens_a, tokens_b, 
                                                       CONFIG["MODEL"]["SENT_MAX_LEN"],
                                                       self.tokenizer)
            all_feats["input_ids"].append(inp_ids)
            all_feats["attention_mask"].append(mask)
            all_feats["token_type_ids"].append(seg_ids)
            all_asrs.append(asr)
        turn.text_b = real_text_b
        return all_feats, all_asrs

    
class BertIndex:

    PATH = "./all_embs.npy"

    def __init__(self):
        self.embeddings = np.load(BertIndex.PATH).item()

    def __call__(self, words: [int]):
        return check_cuda_stack_float([self[idx] for idx in words])

    def __getitem__(self, idx):
        return check_cuda_float(torch.from_numpy(self.embeddings[idx]))

class BertIndex_OLD:

    TRAIN_SIZE = 1611

    def __init__(self):
        self.curr_perc = 1
        self.curr_mode = "train"
        self.load_vectors("train")

    def get_features_from_batch(self, guids: [str]):
        return [self.get_features(guid) for guid in guids]

    def get_features(self, guid: str):

        try:
            return self.vectors[f"{guid}_seq"], self.vectors[f"{guid}_comb"]
        except KeyError:
            self.curr_perc += 1
            self.load_vectors(guid.split("-")[0])
        return self.vectors[f"{guid}_seq"], self.vectors[f"{guid}_comb"]

    def load_vectors(self, mode):
        mode = mode if mode != "dev" else "validate"
        if mode != self.curr_mode:
            self.curr_perc = 1
            self.curr_mode = mode
        path = f"{CONFIG['VECTORS']}{mode}/{self.curr_perc}/features.npz"
        self.vectors = np.load(path)

class BertNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.preprep = PrePrepped()
        self.loader = DataLoader()
        self.device = "cuda" if is_available() else "cpu"
        self.is_cuda = self.device == "cuda"
        self._setup_bert()
        emb_dim = self.bert.get_input_embeddings().embedding_dim
        self.emb_dim = emb_dim
        self._setup_layers()
        self._setup_loss_activation()
        #self.bert_index = BertIndex()
        self.cache = {}

        self.slots = ["food", "area", "pricerange"]
        self.ontology = DataFetcher.fetch_dstc_ontology()["informable"]

    def get_relevant_results(self, batch, predset, slot_name):
        """
        Position loss should only be calculated for those indexes where 
        both the prediction and the ground truth is 2, i.e. slot detected.
        """
        true_slot_idx = batch.idx_for_cat(slot_name)
        pred_slot_idx = predset.pred_idx(slot_name, true_slot_idx)
        num_slot = pred_slot_idx.size(0)
        num_slot = 1 if num_slot == 0 else num_slot
        first, second = predset.pos_preds(slot_name, pred_slot_idx)
        return num_slot, first, second, pred_slot_idx

    def get_pos_loss(self, predset, batch, slot_name):
        """
        Calculate the position loss for some predictions for a batch given a slot name.
        Should amount to 20% of the loss, 10% for each position.
        """
        num_slot, first_preds, second_preds, pred_slot_idx = \
            self.get_relevant_results(batch, predset, slot_name)
        
        first_preds, second_preds = predset.pos_preds(slot_name,
                pred_slot_idx)

        first_position_loss = self.pos_loss_fn(
                first_preds, batch.cat_map[slot_name]["first"][pred_slot_idx]
        ) * .1
        first_position_loss[torch.isnan(first_position_loss)] = 0
        second_position_loss = self.pos_loss_fn(
                second_preds, batch.cat_map[slot_name]["second"][pred_slot_idx]
        ) * .1
        second_position_loss[torch.isnan(second_position_loss)] = 0
        return first_position_loss, second_position_loss, num_slot

    def get_cls_loss(self, predset, batch, slot_name):
        """
        Class loss should account for 80% of the loss. Returns 0 loss if there 
        are no entries in the batch with class = 2.
        """
        try:
            class_loss = self.cl_loss_fn(
                    predset.cat_map[slot_name]["class"],
                    batch.cat_map[slot_name]["class"]
            ) * .8
        except ValueError:
            class_loss = check_cuda_float([0])
        return class_loss

    def fit(self, mode="train", continue_from=0, watch_param=False, **kwargs):

        best_acc = 0

        dataset = self.preprep.fetch_set(mode, 
                use_asr_hyp=CONFIG["MODEL"]["ASR_HYPS"]
        )
        bsize = CONFIG["MODEL"]["TRAIN_BATCH_SIZE"]
        
        for epoch in range(continue_from, CONFIG["MODEL"]["NUM_EPOCHS"]):
            batch_generator = self.loader.fetch_batch(
                    dataset,
                    bsize, self.slots, self.tokenizer, slot_value_dropout=0
            )
            self.train()
            self.bert.train()
            logger.info("Starting epoch # %s" % (epoch + 1))
            epoch_class_loss, epoch_pos_loss = 0, 0
            partition = 0
            prevlisted = 10
            for batch in batch_generator:
                partition += round((bsize/len(dataset))*100, 3)
                if partition > prevlisted:
                    logger.info(f"{prevlisted}% of dataset processed")
                    prevlisted += 10 
                self.zero_grad()
                class_logits, pos_logits = self(batch, **kwargs)
                predset = PredictionSet(*class_logits, *pos_logits)
                loss = 0
                for slot_name in predset.categories:
                    first_pos_loss, second_pos_loss, num_slot = self.get_pos_loss(
                            predset, batch, slot_name)
                    epoch_pos_loss += (first_pos_loss.item() / num_slot)
                    epoch_pos_loss += (second_pos_loss.item() / num_slot)
                    class_loss = self.get_cls_loss(predset, batch, slot_name)

                    epoch_class_loss += class_loss.item()
                    loss += (first_pos_loss * .1) + (second_pos_loss * .1) + \
                            (class_loss * .8)

                loss.backward()
                self.optim.step()
            logger.info("Position loss for epoch: %s" % epoch_pos_loss)
            logger.info("Class loss for epoch: %s" % epoch_class_loss)
            logger.info("Dev loss:")
            self.predict(watch_param=watch_param, **kwargs)
            torch.save(self.state_dict(),
                    open(f"{CONFIG['MODEL']['MODEL_DIR']}bertdst-light-{mode}-{epoch}.pt", "wb")
            )
            torch.save(self.bert.state_dict(),
                    open(f"{CONFIG['MODEL']['MODEL_DIR']}light-bertstate-{mode}-{epoch}.pt", "wb")
            )
            logger.info("#" * 30)


    @staticmethod
    def _dst_pred_format(guid, cls_gt, slot):
        return {
            "guid": guid,
            "slot": slot,
            "class_label_id": cls_gt,
            "class_prediction": 0,
            "start_pos": 0,
            "start_prediction": 0,
            "end_pos": 0,
            "end_prediction": 0
        }

    @staticmethod
    def _dst_slot_map():
        return {
            "none": 0,
            "dontcare": 1,
            "copy_value": 2,
            "unpointable": 3
        }

    @staticmethod
    def _dst_first_last_index(entry):
        first = entry.index(1)
        for i in [x for x in range(len(entry)) if x != first]:
            if entry[i] == 1:
                return first, i
        return first, first

    def get_prediction_format(self, predset, **kwargs):
        batch_generator = self.loader.fetch_batch(
                predset, 1, self.slots, self.tokenizer, pred=True, slot_value_dropout=0
        )

        all_pred_info = {}
        all_preds = []
        for p in predset:
            all_pred_info[p.guid] = {}
            for slot in [x if x != "pricerange" else "price range" for x in self.slots]:
                cls = self._dst_slot_map()[p.class_label[slot]]
                pinfo_slot = slot if slot != "price range" else "pricerange"
                all_pred_info[p.guid][pinfo_slot] = {}
                all_pred_info[p.guid][pinfo_slot]["gt_class"] = cls
                pred_dict = self._dst_pred_format(p.guid, cls, slot)
                if cls in (0, 1, 3):
                    all_preds.append(pred_dict)
                    continue
                idxs = p.text_a_label[slot].copy()
                idxs.extend(p.text_b_label[slot])
                first, last = self._dst_first_last_index(idxs)
                pred_dict["start_pos"] = first
                pred_dict["end_pos"] = last
                all_pred_info[p.guid][pinfo_slot]["gt_first"] = first
                all_pred_info[p.guid][pinfo_slot]["gt_second"] = last

                all_preds.append(pred_dict)
        
        slot_preds = {
            "food": [],
            "area": [],
            "pricerange": []
        }
        for batch in tqdm(batch_generator):
            class_logits, pos_logits = self(batch, train=False, **kwargs)
            preds = PredictionSet(*class_logits, *pos_logits)
            guid = batch.guids[0]
            relevant_p = [x for x in all_preds if x["guid"] == guid]
            for p in relevant_p:
                slot = p["slot"] if p["slot"] != "price range" else "pricerange"
                pred_map = preds.cat_map[slot]
                cls = pred_map["class"].argmax(-1).item()
                first = pred_map["first"].argmax(-1).item()
                second = pred_map["second"].argmax(-1).item()
                p["class_prediction"] = cls
                p["start_prediction"] = first
                p["end_prediction"] = second
                all_pred_info[guid][slot]["class"] = cls
                all_pred_info[guid][slot]["first"] = first
                all_pred_info[guid][slot]["second"] = second
                slot_preds[slot].append(p)

        return slot_preds, all_pred_info, all_preds

    @staticmethod
    def _calc_fscore(prec, recall):
        return 2 * ((prec * recall) / (prec + recall))

    @staticmethod
    def _get_recall(tp, fn):
        return tp / (tp + fn)

    @staticmethod
    def _get_prec(tp, fp):
        return tp / (tp + fp)

    def get_prediction_dump(self, slot_preds):
        """
        Need to keep track of the following:
            1) At which turn in the dialogue does the model fail
            2) Which slot fails
            3) Is it a false positive, wrong prediction, or false negative
            4) 3, for each slot

        input is [
            {
                gucci
                guid
                slot
                class_label_id
                class_prediction
                start_pos
                start_prediction
                end_pos
                end_prediction
            }
        ]
        """
        dump = PredictionDump()
        for turn_idx in range(len(slot_preds["food"])):
            for slot in self.slots:
                dump.assign_turn(slot_preds[slot][turn_idx])
        print(len(dump.dialogues))
        return dump


    @staticmethod
    def _turn_iscurr(turn, curr):
        return int(turn["guid"].split("-")[1]) == curr

    def get_distance_measures(self, mode="validate", **kwargs):
        self.eval()
        self.bert.eval()
        predset = self.preprep.fetch_set(mode, 
                use_asr_hyp=CONFIG["MODEL"]["ASR_HYPS"], 
                exclude_unpointable=False
        )
        slot_preds, all_pred_info, all_preds = \
                self.get_prediction_format(predset, **kwargs)
        counts = {s: {"first":[],"second":[]} for s in self.slots}
        for slot, turns in slot_preds.items():
            for t in turns:
                if t["class_label_id"] != 2 or t["class_prediction"] != 2:
                    continue
                counts[slot]["first"].append(
                        abs(t["start_pos"] - t["start_prediction"])
                )
                counts[slot]["second"].append(
                        abs(t["end_pos"] - t["end_prediction"])
                )
        for slot, nums in counts.items():
            first_miss_rate = \
                    len([x for x in nums["first"] if x > 0]) / len(nums["first"])
            second_miss_rate = \
                    len([x for x in nums["second"] if x > 0]) / len(nums["second"])
            print(f"For slot {slot}")
            print("Mean of first errors: %s" % np.median(nums["first"]))
            print("Mean of second errors: %s" % np.median(nums["second"]))
            print("Mean of first errors, errors only: %s" % np.median(
                [x for x in nums["first"] if x > 0]))
            print("Mean of second errors, errors only: %s" % np.median(
                [x for x in nums["second"] if x > 0]))
            print("Number of misses, first: %s" % len(
                [x for x in nums["first"] if x > 0]))
            print("Number of misses, second: %s" % len(
                [x for x in nums["second"] if x > 0]))
            print("Miss rate, first: %s" % first_miss_rate)
            print("Miss rate, second: %s" % second_miss_rate)

    def get_cer(self, mode="validate", no_blanks=False, **kwargs):
        self.eval()
        self.bert.eval()
        predset = self.preprep.fetch_set(mode, 
                use_asr_hyp=CONFIG["MODEL"]["ASR_HYPS"], 
                exclude_unpointable=False
        )
        slot_preds, all_pred_info, all_preds = \
                self.get_prediction_format(predset, **kwargs)
        total_score = 0
        blank_count = 0
        num_turns = len(slot_preds["food"])
        for turn_idx in range(num_turns):
            score = 1
            turn_isblank = 0
            for slot in self.slots:
                turn_pred = slot_preds[slot][turn_idx]
                if turn_pred["class_label_id"] != turn_pred["class_prediction"]:
                    score -= .33
                elif turn_pred["class_label_id"] == 0 and no_blanks:
                    turn_isblank += 1
            if (no_blanks and turn_isblank < 3) or not no_blanks:
                total_score += score
            else:
                blank_count += 1
        if no_blanks:
            noblank_score = total_score / (num_turns - blank_count)
            print("CER without blanks: %s" % noblank_score)
        else:
            print("Combined concept error rate: %s" % (total_score / num_turns))

    def get_average_perplexity(self, mode="validate", **kwargs):
        self.eval()
        self.bert.eval()
        predset = self.preprep.fetch_set(mode, 
                use_asr_hyp=CONFIG["MODEL"]["ASR_HYPS"], 
                exclude_unpointable=False
        )
        batch_generator = self.loader.fetch_batch(
                predset, 1, self.slots, self.tokenizer, pred=True, slot_value_dropout=0
        )
        class_perplexity = {s: [] for s in self.slots}
        avg_class_perplexity = 0
        avg_pos_perplexity = 0
        for batch in tqdm(batch_generator):
            class_logits, pos_logits = self(batch, train=False, **kwargs)
            for slot, slot_logits in zip(self.slots, class_logits):
                logval = np.log2(
                            np.max(slot_logits[0].cpu().detach().numpy())
                        )
                logval = logval if not np.isnan(logval) else 0
                class_perplexity[slot].append(logval)

        for slot, vals in class_perplexity.items():
            print("perplexity for %s: %s" % (slot, 2**(-(sum(vals) * (1/len(vals))))))

        return class_perplexity

    def get_f_score(self, mode="validate", **kwargs):
        self.eval()
        self.bert.eval()
        predset = self.preprep.fetch_set(mode, 
                use_asr_hyp=CONFIG["MODEL"]["ASR_HYPS"], 
                exclude_unpointable=False
        )
        slot_preds, all_pred_info, all_preds = \
                self.get_prediction_format(predset, **kwargs)

        counts = ConditionalFreqDist()
        for slot, turns in slot_preds.items():
            for t in turns:
                if t["class_label_id"] == t["class_prediction"] and t["class_prediction"] == 0:
                    continue
                if t["class_label_id"] == t["class_prediction"]:
                    counts[slot]["tp"] += 1
                elif t["class_prediction"] != t["class_label_id"] and t["class_label_id"] == 0:
                    counts[slot]["fp"] += 1
                else:
                    counts[slot]["fn"] += 1
        print("results:")
        print(counts.tabulate())
        for slot, res in counts.items():
            print(f"for slot {slot}:")
            prec = self._get_prec(res["tp"], res["fp"])
            recall = self._get_recall(res["tp"], res["fn"])
            f_score = self._calc_fscore(prec, recall)
            print("Precision: %s\t recall: %s\t f-score: %s" % (prec, recall, f_score))

    def predict(self, mode="validate", watch_param=False, **kwargs):
        self.eval()
        self.bert.eval()
        predset = self.preprep.fetch_set(mode, 
                use_asr_hyp=CONFIG["MODEL"]["ASR_HYPS"], 
                exclude_unpointable=False
        )

        slot_preds, all_pred_info, all_preds = \
                self.get_prediction_format(predset, **kwargs)
        tot_corrs, joint_acc = self.report_predict(slot_preds)
        if watch_param:
            where_wrong = np.where(joint_acc == 0)[0]
            for num_idx, wrong_idx in enumerate(where_wrong):
                turn = predset[wrong_idx]
                guid = turn.guid
                logger.info("For turn with guid %s" % guid)
                logger.info(" ".join(turn.text_a))
                logger.info(" ".join(turn.text_b))
                logger.info("_"*30)
                logger.info("Predictions:")
                for slot in self.slots:
                    gt_slot = "price range" if slot == "pricerange" else slot
                    gt_cls = self._dst_slot_map()[turn.class_label[gt_slot]]
                    pred_cls = all_pred_info[guid][slot]["class"]
                    logger.info("¤"*30)
                    logger.info("For slot %s with pred class %s and ground truth %s" % (slot, pred_cls, gt_cls))
                    logger.info("With result %s" % tot_corrs[slot][wrong_idx])
                    if gt_cls != pred_cls:
                        logger.info("Wrong class prediction, no idx needed")
                        continue
                    if gt_cls != 2:
                        prev_wrong_idx = where_wrong[num_idx-1]
                        prev_wrong_guid = predset[prev_wrong_idx].guid
                        if prev_wrong_guid == guid and tot_corrs[slot][prev_wrong_idx] == 0:
                            logger.info("Carried over bad label from previous turn.")
                            continue
                            
                        logger.info("No idx, got it right")
                        continue
                    logger.info("real idx:", all_pred_info[guid][slot]["gt_first"], 
                            all_pred_info[guid][slot]["gt_second"])
                    logger.info("pred_idxs:", all_pred_info[guid][slot]["first"],
                            all_pred_info[guid][slot]["second"])
                    logger.info("_"*30)
                logger.info("#"*30)
                stop_cont = input()
                if stop_cont == "stop":
                    break
        joint_acc = np.mean(joint_acc)
        logger.info("Joint accuracy: %s" % joint_acc)


    @staticmethod
    def report_predict(pred_dict: {str: [int]}):
        joint_acc = 1
        tot_corrs = {}
        for slot, preds in pred_dict.items():
            logger.info("For slot %s" % slot)
            tot_corr, class_corr, pos_corr = get_joint_slot_correctness(preds,
                    ignore_file=True)
            tot_corrs[slot] = tot_corr
            joint_acc *= tot_corr
            logger.info("total correct: %s" % np.mean(tot_corr))
            logger.info("class correct: %s" % np.mean(class_corr))
            logger.info("pos correct %s" % np.mean(pos_corr))
        return tot_corrs, joint_acc

    @staticmethod
    def filter_bert_dict(bert_dict, asrs):
        """
        It is best to filter out the utterances under the elected 
        ASR threshold before running it through the BERT model, to reduce
        impact on resources.
        """
        new_dict = {k: [] for k in bert_dict.keys()}
        for input_type, values in bert_dict.items():
            for num_iter, (asr, value) in enumerate(zip(asrs, values)):
                if num_iter == 0 or asr > CONFIG["MODEL"]["ASR_THRESHOLD"]:
                    new_dict[input_type].append(value)
        
        return {k: check_cuda_stack_long(v) for k,v in new_dict.items()}
            
    
    def weighted_calc(self, X, bsize):
        """
        Calculate the weighted sum of the ASR vectors.
        """
        comb = torch.zeros((bsize, self.emb_dim), requires_grad=True).to(self.device)
        seq = torch.zeros((bsize, CONFIG["MODEL"]["SENT_MAX_LEN"], self.emb_dim), requires_grad=True).to(self.device)
        dial_idx = -1
        for dial, feats in X.asr_feats.items():
            dial_idx += 1
            bert_dict, asr = feats
            bert_dict = self.filter_bert_dict(bert_dict, asr)
            new_seq , new_comb = self.bert(**bert_dict).to_tuple()
            if (torch.sum(bert_dict["input_ids"][X.text_indexes[dial_idx]])
                !=
                torch.sum(X.to_bert_format()["input_ids"][dial_idx])
            ):
                print(dial_idx)
                print(bert_dict["input_ids"])
                print(X.to_bert_format()["input_ids"][dial_idx])
                print(dial)
                input()
            for new_entry in range(new_comb.size(0)):
                comb[dial_idx] += (new_comb[new_entry] * asr[new_entry])
            seq[dial_idx] = new_seq[X.text_indexes[dial_idx]]
        return seq, comb
        
    def _get_output(self, X, bsize, weighted):
        return self.weighted_calc(X, bsize) if weighted else \
                self.bert(**X.to_bert_format()).to_tuple()
        

    def forward(self, X, train=True, weighted=False):

        bsize = CONFIG["MODEL"]["TRAIN_BATCH_SIZE"] if train else 1
        
        seq, comb = self._get_output(X, bsize, weighted)
        if train:
            comb = self.dropout(comb)
            seq = self.dropout(seq)

        a_food = self.food_a(comb)
        a_area = self.area_a(comb)
        a_price = self.price_a(comb)

        food_ab = self.alphabeta_food(seq)
        area_ab = self.alphabeta_area(seq)
        price_ab = self.alphabeta_price(seq)

        get_alphabeta = lambda x: (x[:, :, 0], x[:, :, 1])

        return ((a_food, a_area, a_price),
                (get_alphabeta(food_ab),
                get_alphabeta(area_ab),
                get_alphabeta(price_ab)))

    def _setup_loss_activation(self):
        self.activation = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=CONFIG["MODEL"]["DROPOUT"])
        self.cl_loss_fn = nn.CrossEntropyLoss()
        self.pos_loss_fn = nn.CrossEntropyLoss()
        if self.is_cuda:
            self.bert.cuda()
            self.softmax.cuda()
            self.cl_loss_fn.cuda()
            self.pos_loss_fn.cuda()

        self.optim = Adam(self.parameters(), lr=CONFIG["MODEL"]["LEARN_RATE"])


    def _setup_bert(self):
        self.tokenizer = BertTokenizer.from_pretrained(CONFIG["MODEL"]["BERT_VERSION"])
        #self.tokenizer = self.loader.tokenizer
        self.bert = BertModel.from_pretrained(CONFIG["MODEL"]["BERT_VERSION"])
        if self.is_cuda:
            self.bert.cuda()


    def _setup_layers(self):
        self.food_a = nn.Linear(self.emb_dim, 3)
        self.area_a = nn.Linear(self.emb_dim, 3)
        self.price_a = nn.Linear(self.emb_dim, 3)

        self.alphabeta_food = nn.Linear(self.emb_dim, 2)
        self.alphabeta_area = nn.Linear(self.emb_dim, 2)
        self.alphabeta_price = nn.Linear(self.emb_dim, 2)

        if self.is_cuda:
            self.food_a.cuda()
            self.area_a.cuda()
            self.price_a.cuda()
            self.alphabeta_food.cuda()
            self.alphabeta_area.cuda()
            self.alphabeta_price.cuda()




def check_cuda_float(list_like):
    to_ret = torch.FloatTensor(list_like)
    if is_available():
        to_ret = to_ret.cuda()
    return to_ret

def check_cuda_long(list_like):
    to_ret = torch.LongTensor(list_like)
    if is_available():
        to_ret = to_ret.cuda()
    return to_ret

def _check_cuda_stack(list_like):
    to_ret = torch.stack(list_like)
    if is_available():
        to_ret = to_ret.cuda()
    return to_ret

def check_cuda_stack_long(list_like):
    converted = _check_cuda_stack_both(list_like, check_cuda_long)
    return _check_cuda_stack(converted)

def check_cuda_stack_float(list_like):
    converted = _check_cuda_stack_both(list_like, check_cuda_float)
    return _check_cuda_stack(converted)

def _check_cuda_stack_both(list_like, function):
    if len(list_like) == 0:
        return torch.tensor([])

    if isinstance(list_like[0], list) or isinstance(list_like[0], np.ndarray):
        converted = [function(x) for x in list_like]
    elif isinstance(list_like[0], int):
        converted = [function(list_like)]
    else:
        converted = list_like
    return converted


class PredictionSet:

    def __init__(self, food_class, area_class, price_class, food_pos, area_pos,
            price_pos):
        self.food_class = food_class
        self.area_class = area_class
        self.price_class = price_class

        self.food_pred_first = food_pos[0]
        self.food_pred_second = food_pos[1]
        self.area_pred_first = area_pos[0]
        self.area_pred_second = area_pos[1]
        self.price_pred_first = price_pos[0]
        self.price_pred_second = price_pos[1]

        self.categories = ["food", "area", "pricerange"]
        self.cat_map = {
                "food": {
                    "class": self.food_class,
                    "first": self.food_pred_first,
                    "second": self.food_pred_second
                },
                "area": {
                    "class": self.area_class,
                    "first": self.area_pred_first,
                    "second": self.area_pred_second
                },
                "pricerange": {
                    "class": self.price_class,
                    "first": self.price_pred_first,
                    "second": self.price_pred_second
                }
        }

    def eval_format(self, slot):
        max_len = 80
        vals = [0] * max_len
        first_idx = self.cat_map[slot]["first"].argmax(-1).item()
        sec_idx = self.cat_map[slot]["second"].argmax(-1).item()
        #if sec_idx < first_idx:
        #    sec_idx = first_idx
        #elif sec_idx > first_idx + 1:
        #    sec_idx = first_idx + 1 
        vals[first_idx] = 1
        try:
            vals[sec_idx] = 1
        except IndexError:
            logger.info("wtf")
            logger.info(sec_idx)
        return vals

    def pred_idx(self, cat, idx):
        if idx[0].size(0) == 0:
            return check_cuda_long([])
        pred_idx = torch.where(self.cat_map[cat]["class"].argmax(-1) == 2)
        if pred_idx[0].size(0) == 0:
            return check_cuda_long([])
        pred_idx = pred_idx[0]
        idx = idx[0]
        if pred_idx.size(0) != idx.size(0):
            pred_idx, idx = self.pad_idxes(pred_idx, idx)
        return pred_idx[torch.where(idx == pred_idx)]

    def pad_idxes(self, idx_1, idx_2):

        def solve(shorter, longer, shortlen, longlen):
            new_short = list(range(longlen))
            new_short[:shortlen] = shorter.tolist()
            new_short[shortlen:] = [999] * (longlen-shortlen)
            return check_cuda_long(new_short)

        first = idx_1.size(0)
        second = idx_2.size(0)
        if first > second:
            new_short = solve(idx_2, idx_1, second, first)
            new_long = idx_1
        else:
            new_short = solve(idx_1, idx_2, first, second)
            new_long = idx_2
        return new_short, new_long

    def pos_preds(self, cat, idx):
        first_part = self.cat_map[cat]["first"][idx]
        second_part = self.cat_map[cat]["second"][idx]
        return first_part, second_part



class FeatureBatchSet:

    def __init__(self, inputs, masks, types, starts, ends, food_labels, area_labels,
                 price_labels, all_feats, text_indexes, pred=False):
        
        self.inputs = check_cuda_stack_long(inputs)
        self.masks = check_cuda_stack_long(masks)
        self.types = check_cuda_stack_long(types)

        self.food_start = check_cuda_long(starts["food"])
        self.food_end = check_cuda_long(ends["food"])
        self.area_start = check_cuda_long(starts["area"])
        self.area_end = check_cuda_long(ends["area"])
        self.price_start = check_cuda_long(starts["pricerange"])
        self.price_end = check_cuda_long(ends["pricerange"])

        self.food_class = check_cuda_long(food_labels)
        self.area_class = check_cuda_long(area_labels)
        self.price_class = check_cuda_long(price_labels)

        self.text_indexes = text_indexes
        
        self.cat_map = {
                "food": {
                    "class": self.food_class,
                    "first": self.food_start,
                    "second": self.food_end
                },
                "area": {
                    "class": self.area_class,
                    "first": self.area_start,
                    "second": self.area_end
                },
                "pricerange": {
                    "class": self.price_class,
                    "first": self.price_start,
                    "second": self.price_end
                }
        }
        self.guids = list(all_feats.keys())
        self.asr_feats = self._format_all_feats(all_feats)

    def idx_for_cat(self, cat):
        return torch.where(self.cat_map[cat]["class"] == 2)

    def to_bert_format(self):
        return {
            "input_ids": self.inputs,
            "attention_mask": self.masks,
            "token_type_ids": self.types
        }

    def _format_all_feats(self, f):
        ## input: dict {turn-guid: (bert_dict, [asr])}
        dicty = {}
        for turn_guid, berts in f.items():
            for k,v in berts[0].items():
                berts[0][k] = check_cuda_stack_long(v)
            dicty[turn_guid] = berts
        return dicty


class PredictionDump:

    def __init__(self):
        self.dialogues: [PredictionDialogue] = []
        self.current_dial = None

    def assign_turn(self, turn):
        turn_guid = turn["guid"]
        if self.got_dial(turn_guid):
            self.current_dial.add_turn(turn)
        else:
            dial = PredictionDialogue()
            self.dialogues.append(dial)
            self.current_dial = dial
            dial.add_turn(turn)

    def got_dial(self, guid):
        if not self.current_dial:
            return False
        return self.current_dial.is_dial(guid)

    def to_json(self):
        return [d.to_json() for d in self.dialogues]




class PredictionDialogue:

    def __init__(self):
        self.turns: [PredictionTurn] = []
        self.dial_idx = None
        self.is_perfect = True
        self.current_turn = None

    def add_turn(self, turn: {str: int}):
        if self.got_turn(turn["guid"]):
            self.current_turn.update(
                turn["slot"], 
                turn["class_label_id"], 
                turn["class_prediction"]
            )
        else:
            t = PredictionTurn(turn)
            self.turns.append(t)
            self.current_turn = t
            self.dial_idx = int(turn["guid"].split("-")[1])
            t.update(
                turn["slot"], 
                turn["class_label_id"], 
                turn["class_prediction"]
            )

    def find_first_fail(self):
        """
        Return first turn that 
        """
        goods = list(sorted(
            filter(lambda t: not t.is_good(), self.turns), key=lambda t: t.turn_idx 
        ))
        return goods[0] if len(goods) else \
                list(sorted(self.turns, key=lambda t: t.turn_idx))[-1]

    def got_turn(self, guid):
        if not self.current_turn:
            return False
        return self.current_turn.is_turn(guid)

    def is_dial(self, guid):
        return int(guid.split("-")[1]) == self.dial_idx

    def to_json(self):
        d = { "dial_idx": self.dial_idx }
        any_failed = [t for t in self.turns if t.is_good()]
        if len(any_failed) == len(self.turns):
            d["fail"] = False
            d["fail_idx"] = -1
        else:
            first_failed = self.find_first_fail()
            d["fail"] = True
            d["fail_idx"] = first_failed.turn_idx
            d["fail_slots"] = [k for k,v in first_failed.slots.items() if not v]
        d["turns"] = [t.to_json() for t in self.turns]
        return d



class PredictionTurn:
    
    def __init__(self, turn_dict):
        self.guid = turn_dict["guid"]
        self.turn_idx = int(self.guid.split("-")[-1])
        self.slots = {
            "food": True,
            "area": True,
            "pricerange": True
        }

    def is_good(self):
        return all(self.slots.values())

    def update(self, slot, class_label, class_pred):
        self.slots[slot] = (class_label == class_pred)

    def is_turn(self, guid):
        return int(guid.split("-")[-1]) == self.turn_idx

    def to_json(self):
        d = { "guid": self.guid }
        d.update(self.slots)
        return d


def pad_to_max_len(tokens, max_len, pad_token):
    while len(tokens) != max_len:
        tokens.append(pad_token)
    return tokens


def load_model_state(model, dataset, number):
    model.load_state_dict(torch.load(f"{CONFIG['MODEL']['MODEL_DIR']}bertdst-light-{dataset}-{number}.pt"))
    model.bert.load_state_dict(torch.load(f"{CONFIG['MODEL']['MODEL_DIR']}light-bertstate-{dataset}-{number}.pt"))


if __name__ == '__main__':
    from argparse import ArgumentParser
    import sys
    import os

    parser = ArgumentParser()
    parser.add_argument("mode", type=str)
    parser.add_argument("--set", type=str, required=False, default="train")
    parser.add_argument("--log", action='store_true')
    parser.add_argument("--single", type=str, required=False)
    parser.add_argument("--continue_from", type=int, required=False, default=-1)
    parser.add_argument("--weighted", action='store_true')

    args = parser.parse_args()

    mode = args.mode
    dataset = args.set
    bn = BertNet()
    if mode == "train":
        cont_from = max(args.continue_from, -1)
        if cont_from >= 0:
            load_model_state(bn, dataset, cont_from)
        bn.fit(mode=dataset, continue_from=cont_from+1, weighted=args.weighted,
                watch_param=args.log)
    elif mode in ("test", "validate"):
        if args.single:
            load_model_state(bn, dataset, args.single)
            bn.predict(mode=mode, watch_param=args.log, weighted=args.weighted)
        else:
            for ep in range(CONFIG["MODEL"]["NUM_EPOCHS"]):
                if not\
                    os.path.exists(f'{CONFIG["MODEL"]["MODEL_DIR"]}bertdst-light-{dataset}-{ep}.pt'):
                        logger.info("FINISHED AT EPOCH %s" % ep)
                        sys.exit(0)

                logger.info("For model # %s" % ep)
                load_model_state(bn, dataset, ep)
                bn.predict(mode=mode, watch_param=args.log, weighted=args.weighted)
    else:
        logger.info("Mode must be either train, validate, or test")
        sys.exit(1)

