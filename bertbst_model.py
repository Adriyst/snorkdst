import torch
from torch import nn
from torch.optim import Adam
from torch.cuda import is_available
from transformers import BertTokenizer, BertModel

from nltk import FreqDist, ConditionalFreqDist, bigrams
import random
from copy import deepcopy
import logging
import pickle
import json
import os 
import numpy as np
from tqdm import tqdm
import numpy as np
from pylev import levenshtein as distance

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
        self.snorkel_set = None
        self.snorkel_pattern_matching_set = None

    def fetch_set(self, version, **kwargs):
        loaded_map = {
            "train": self.train_set,
            "validate": self.validate_set,
            "test": self.test_set,
            "majority": self.majority_set,
            "majority_pattern_matching": self.majority_pattern_matching_set,
            "snorkel": self.snorkel_set,
            "snorkel_pattern_matching": self.snorkel_pattern_matching_set
        }
        relevant_set = loaded_map[version]
        if relevant_set:
            return relevant_set

        mode_map = {
            "train": "train",
            "validate": "dev",
            "test": "test"
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
            self.text_b, self.text_b_label = self.find_pointed_asr()
            self.class_label = self.group[0].class_label
            self.asrs = [x.asr_score for x in self.group]
            self.all_texts = [x.text_b for x in self.group]
            self.session_id = self.group[0].session_id


    def find_pointed_asr(self):
        for ex in self.group:
            for idx_labeling in (ex.text_b_label, ex.text_a_label):
                if any(idx_labeling):
                    return ex.text_b, ex.text_b_label
        return (self.group[0].text_b, self.group[0].text_b_label) \
                if len(self.group) > 0 else ([], [])



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
        

    def get_tokens_and_labels(self, turn, slot, **kwargs):
        tokens_a, token_labels_a = tokenize_text_and_label(
                turn.text_a, turn.text_a_label, slot, self.tokenizer, **kwargs
        )
        tokens_b, token_labels_b = tokenize_text_and_label(
                turn.text_b, turn.text_b_label, slot, self.tokenizer, **kwargs
        )
        token_label_ids = get_token_label_ids(
                token_labels_a, token_labels_b, CONFIG["MODEL"]["SENT_MAX_LEN"]
        )

        startvals, endvals = get_start_end_pos(
            turn.class_label[slot], token_label_ids, CONFIG["MODEL"]["SENT_MAX_LEN"]
        )

        if len(tokens_b) == 0:
            tokens_b.append("[UNK]")
        
        return (tokens_a, tokens_b, startvals, endvals)


    def produce_batchset(self, turns, slots, bert_tokenizer, pred=False, **kwargs):
        input_ids, masks, types, labels = [], [], [], []
        food_labels, area_labels, price_labels = [], [], []
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
            #print(turn.guid)
            #print(word_idxs)
            #input()
            input_ids.append(inp_ids)
            masks.append(mask)
            types.append(seg_ids)


            for lab, lablist in zip(class_label_id_dict.values(), 
                    [food_labels, area_labels, price_labels]):
                lablist.append(lab)

            labels.append(turn.class_label)
            turn_based_bert[turn.guid] = self._sort_asr(turn, bert_tokenizer) 

        return FeatureBatchSet(input_ids, masks, types, starts, ends, food_labels,
                area_labels, price_labels, turn_based_bert, pred=pred)

    def _sort_asr(self, turn, bt):
        all_feats = {"input_ids": [], "attention_mask": [], "token_type_ids": []}
        all_asrs = []
        for txt, asr in zip(turn.all_texts, turn.asrs):
            if len(txt) == 0:
                txt.append(self.UNK_TOKEN)
            _, inp_ids, mask, seg_ids = get_bert_input(turn.text_a, txt, 
                    CONFIG["MODEL"]["SENT_MAX_LEN"], bt)
            all_feats["input_ids"].append(inp_ids)
            all_feats["attention_mask"].append(mask)
            all_feats["token_type_ids"].append(seg_ids)
            all_asrs.append(asr)
        return all_feats, all_asrs


class BertNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.preprep = PrePrepped()
        self.loader = DataLoader()
        self.is_cuda = is_available()
        self._setup_bert()
        emb_dim = self.bert.get_input_embeddings().embedding_dim
        self.emb_dim = emb_dim
        self._setup_layers()
        self._setup_loss_activation()

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

    def fit(self, mode="train", continue_from=0, **kwargs):

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
            self.predict(**kwargs)
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

    def predict(self, mode="validate", watch_param=False, **kwargs):
        self.eval()
        self.bert.eval()
        all_preds = []
        predset = self.preprep.fetch_set(mode, use_asr_hyp=1, exclude_unpointable=False)
        batch_generator = self.loader.fetch_batch(
                predset, 1, self.slots, self.tokenizer, pred=True, slot_value_dropout=0
        )

        all_pred_info = {}
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
            guid = batch.guid
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

        joint_acc = 1
        tot_corrs = {}

        for slot, preds in slot_preds.items():
            print("For slot", slot)
            tot_corr, class_corr, pos_corr = get_joint_slot_correctness(preds,
                    ignore_file=True)
            tot_corrs[slot] = tot_corr
            joint_acc *= tot_corr
            print("total correct:", np.mean(tot_corr))
            print("class correct:", np.mean(class_corr))
            print("pos correct", np.mean(pos_corr))
        
        if watch_param:
            where_wrong = np.where(joint_acc == 0)[0]
            for num_idx, wrong_idx in enumerate(where_wrong):
                turn = predset[wrong_idx]
                guid = turn.guid
                print("For turn with guid", guid)
                print(" ".join(turn.text_a))
                print(" ".join(turn.text_b))
                print("_"*30)
                print("Predictions:")
                for slot in self.slots:
                    gt_slot = "price range" if slot == "pricerange" else slot
                    gt_cls = self._dst_slot_map()[turn.class_label[gt_slot]]
                    pred_cls = all_pred_info[guid][slot]["class"]
                    print("¤"*30)
                    print("For slot", slot, "with pred class",
                            pred_cls, "and ground truth", gt_cls)
                    print("With result", tot_corrs[slot][wrong_idx])
                    if gt_cls != pred_cls:
                        print("Wrong class prediction, no idx needed")
                        continue
                    if gt_cls != 2:
                        prev_wrong_idx = where_wrong[num_idx-1]
                        prev_wrong_guid = predset[prev_wrong_idx].guid
                        if prev_wrong_guid == guid and tot_corrs[slot][prev_wrong_idx] == 0:
                            print("Carried over bad label from previous turn.")
                            continue
                            
                        print("No idx, got it right")
                        continue
                    print("real idx:", all_pred_info[guid][slot]["gt_first"], 
                            all_pred_info[guid][slot]["gt_second"])
                    print("pred_idxs:", all_pred_info[guid][slot]["first"],
                            all_pred_info[guid][slot]["second"])
                    print("_"*30)
                print("#"*30)
                input()
        joint_acc = np.mean(joint_acc)
        print("Joint accuracy:", joint_acc)


    
    def weighted_calc(self, X, bsize):
        """
        Calculate the weighted sum of the ASR vectors.
        """
        comb = check_cuda_float(torch.zeros((bsize, self.emb_dim)))
        dial_idx = -1
        curr_dial = ""
        for dial, feats in X.asr_feats.items():
            bert_dict, asr = feats
            if curr_dial != dial:
                curr_dial = dial
                dial_idx += 1
            _, new_comb = self.bert(**bert_dict).to_tuple()
            for ns in range(len(asr)):
                comb[dial_idx] += (new_comb[ns] * asr[ns])
        return comb
        

    def forward(self, X, train=True, feats_only=None, weighted=False):

        bsize = CONFIG["MODEL"]["TRAIN_BATCH_SIZE"] if train else 1
        
        if weighted:
            comb = self.weighted_calc(X, bsize)
            seq, _ = self.bert(**X.to_bert_format()).to_tuple()
        else:
            seq, comb = self.bert(**X.to_bert_format()).to_tuple() 

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

    if isinstance(list_like[0], list):
        converted = [function(x) for x in list_like]
    elif isinstance(list_like[0], int):
        converted = [function(list_like)]
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
            print("wtf")
            print(sec_idx)
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
            price_labels, all_feats, pred=False):
        
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
        if pred:
            self.guid = list(all_feats.keys())[0]
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
    parser.add_argument("--log", type=bool, default=False)
    parser.add_argument("--single", type=str, required=False)
    parser.add_argument("--continue_from", type=int, required=False)
    parser.add_argument("--weighted", type=bool, default=False, required=False)

    args = parser.parse_args()

    mode = args.mode
    dataset = args.set
    bn = BertNet()
    if mode == "train":
        cont_from = args.continue_from if args.continue_from else 0
        if cont_from:
            load_model_state(bn, dataset, cont_from)
        bn.fit(mode=dataset, continue_from=cont_from, weighted=args.weighted)
    elif mode in ("test", "validate"):
        if args.single:
            load_model_state(bn, dataset, args.single)
            bn.predict(mode=mode, watch_param=args.log, weighted=args.weighted)
        else:
            for ep in range(CONFIG["MODEL"]["NUM_EPOCHS"]):
                if not\
                    os.path.exists(f'{CONFIG["MODEL"]["MODEL_DIR"]}bertdst-light-{dataset}-{ep}.pt'):
                        print("FINISHED AT EPOCH ", ep)
                        sys.exit(0)

                print("For model # %s" % ep)
                load_model_state(bn, dataset, ep)
                bn.predict(mode=mode, watch_param=args.log, weighted=args.weighted)
    else:
        print("Mode must be either train, validate, or test")
        sys.exit(1)

