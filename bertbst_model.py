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
#from Levenshtein import distance
# need pure python package for saga
from pylev import levenshtein
distance = levenshtein

import dataset_dstc2
import tokenization
from dst_util import tokenize_text_and_label, get_token_label_ids, get_start_end_pos
from get_data import DataFetcher

BERT_VOCAB_LOC = "/usr/local/Development/bert/vocab.txt"

logger = logging.getLogger(__name__)

random.seed(609)

SEMANTIC_DICT = {
  'centre': ['center', 'downtown', 'central', 'down town', 'middle'],
  'south': ['southern', 'southside'],
  'north': ['northern', 'uptown', 'northside'],
  'west': ['western', 'westside'],
  'east': ['eastern', 'eastside'],
  'east side': ['eastern', 'eastside'],

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

    BASE_PATH = "/usr/local/Development/master/data/clean/"
    SLOTS = ["area", "food", "price range"]
    
    def __init__(self):
        self.train_set = None
        self.validate_set = None
        self.test_set = None
        self.majority_set = None
        self.snorkel_set = None

    def fetch_set(self, version):
        loaded_map = {
            "train": self.train_set,
            "validate": self.validate_set,
            "test": self.test_set,
            "majority": self.majority_set,
            "snorkel": self.snorkel_set
        }
        if (relevant_set := loaded_map[version]):
            return relevant_set

        mode_map = {
            "train": "train",
            "validate": "dev",
            "test": "test",
            "majority": "train",
            "snorkel": "train"
        }
        version_type = mode_map[version]
        loaded_set = dataset_dstc2.create_examples(
            os.path.join(self.BASE_PATH, f"dstc2_{version}_en.json"),
            self.SLOTS, version_type
        )
        loaded_map[version] = loaded_set
        return loaded_set


class DataLoader:

    def __init__(self):
        self.tokenizer = tokenization.FullTokenizer(
                vocab_file="/usr/local/Development/bert/vocab.txt",
                do_lower_case=True
        )

    def fetch_batch(self, dataset, bsize, *args):
        for batch in range(0, len(dataset), bsize):
            if batch + bsize > len(dataset):
                yield self.produce_batchset(dataset[batch:], *args)
            else:
                yield self.produce_batchset(dataset[batch:batch+bsize], *args)
        

    def produce_batchset(self, turns, slots, bert_tokenizer):
        max_len = 80
        input_ids, masks, types, labels = [], [], [], []
        food_labels, area_labels, price_labels = [], [], []
        starts = {k: [] for k in slots}
        ends = deepcopy(starts)
        class_types = ["none", "dontcare", "copy_value"]
        for turn in turns:
            class_label_id_dict = {}
            for slot in ["food", "area", "price range"]:
                tokens_a, token_labels_a = tokenize_text_and_label(
                        turn.text_a, turn.text_a_label, slot, self.tokenizer 
                )
                tokens_b, token_labels_b = tokenize_text_and_label(
                        turn.text_b, turn.text_b_label, slot, self.tokenizer 
                )
                token_label_ids = get_token_label_ids(
                        token_labels_a, token_labels_b, max_len
                )
                class_label_id_dict[slot] = class_types.index(turn.class_label[slot])

                startvals, endvals = get_start_end_pos(
                    turn.class_label[slot], token_label_ids, max_len
                )
                bert_tok = bert_tokenizer.encode_plus(tokens_a, tokens_b,
                        padding='max_length', max_length=80)
                
                if slot == "price range":
                    slot = "pricerange"
                starts[slot].append(startvals)
                ends[slot].append(endvals)

            input_ids.append(bert_tok["input_ids"])
            masks.append(bert_tok["attention_mask"])
            types.append(bert_tok["token_type_ids"])


            for lab, lablist in zip(class_label_id_dict.values(), 
                    [food_labels, area_labels, price_labels]):
                lablist.append(lab)

            labels.append(turn.class_label)
        return FeatureBatchSet(input_ids, masks, types, starts, ends, food_labels,
                area_labels, price_labels)


class BertNet(nn.Module):

    BERT_VERSION = "bert-base-uncased"

    def __init__(self):
        super().__init__()
        self.preprep = PrePrepped()
        self.loader = DataLoader()
        self.tokenizer = BertTokenizer.from_pretrained(self.BERT_VERSION)
        self.bert = BertModel.from_pretrained(self.BERT_VERSION)
        emb_dim = self.bert.get_input_embeddings().embedding_dim
        self.food_a = nn.Linear(emb_dim, 3)
        self.area_a = nn.Linear(emb_dim, 3)
        self.price_a = nn.Linear(emb_dim, 3)

        self.alphabeta_food = nn.Linear(emb_dim, 2)
        self.alphabeta_area = nn.Linear(emb_dim, 2)
        self.alphabeta_price = nn.Linear(emb_dim, 2)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=0.3)
        self.loss_fn = nn.CrossEntropyLoss()

        if is_available():
            self.bert.cuda()
            self.food_a.cuda()
            self.area_a.cuda()
            self.price_a.cuda()
            self.alphabeta_food.cuda()
            self.alphabeta_area.cuda()
            self.alphabeta_price.cuda()
            self.softmax.cuda()
            self.loss_fn.cuda()

        self.optim = Adam(self.parameters(), lr=2e-5)

        self.epochs = 30
        self.train_bsize = 8

        self.slots = ["food", "area", "pricerange"]
        self.ontology = DataFetcher.fetch_dstc_ontology()["informable"]
        
    def get_relevant_results(self, batch, predset, slot_name):
        true_slot_idx = batch.idx_for_cat(slot_name)
        pred_slot_idx = predset.pred_idx(slot_name, true_slot_idx)
        num_slot = pred_slot_idx.size(0)
        num_slot = 1 if num_slot == 0 else num_slot
        first, second = predset.pos_preds(slot_name, pred_slot_idx)
        return num_slot, first, second, pred_slot_idx

    def fit(self, mode="train"):

        best_acc = 0
        
        for epoch in range(self.epochs):
            batch_generator = self.loader.fetch_batch(
                    self.preprep.fetch_set(mode),
                    self.train_bsize, self.slots, self.tokenizer
            )
            self.train()
            self.bert.train()
            logger.info("Starting epoch # %s" % (epoch + 1))
            epoch_class_loss, epoch_pos_loss = 0, 0
            tralse = True

            for batch in batch_generator:
                self.zero_grad()
                class_logits, pos_logits = self(batch.to_bert_format())
                predset = PredictionSet(*class_logits, *pos_logits)

                loss = 0
                for slot_name in predset.categories:
                    num_slot, first_preds, second_preds, pred_slot_idx = \
                            self.get_relevant_results(batch, predset, slot_name)
                    
                    first_preds, second_preds = predset.pos_preds(slot_name,
                            pred_slot_idx)
                    
                    first_position_loss = self.loss_fn(
                            first_preds, batch.cat_map[slot_name]["first"][pred_slot_idx]
                    ) * .1
                    first_position_loss[torch.isnan(first_position_loss)] = 0
                    second_position_loss = self.loss_fn(
                            second_preds, batch.cat_map[slot_name]["second"][pred_slot_idx]
                    ) * .1
                    second_position_loss[torch.isnan(second_position_loss)] = 0
                    epoch_pos_loss += (first_position_loss.item() / num_slot)
                    epoch_pos_loss += (second_position_loss.item() / num_slot)

                    class_loss = self.loss_fn(
                            predset.cat_map[slot_name]["class"],
                            batch.cat_map[slot_name]["class"]
                    ) * .8

                    epoch_class_loss += class_loss.item()
                    loss += first_position_loss + second_position_loss + class_loss

                loss.backward()
                self.optim.step()
            logger.info("Position loss for epoch: %s" % epoch_pos_loss)
            logger.info("Class loss for epoch: %s" % epoch_class_loss)
            logger.info("Dev loss:")
            devacc = self.predict()
            if devacc > best_acc and devacc > .5:
                best_acc = devacc
                torch.save(self.state_dict(), open(f"./bertdst-light-{mode}.pt", "wb"))
                torch.save(self.bert.state_dict(), open(f"./light-bertstate-{mode}.pt", "wb"))
            logger.info("#" * 30)
            
    def predict(self, mode="validate"):
        self.eval()
        self.bert.eval()
        good, bad = 0, 0
        featureset = self.preprep.fetch_set(mode)
        dial_struct = []
        current_id = ""
        current_dial = []
        for turn in featureset:
            _, dial_no, turn_no = turn.guid.split("-")
            if dial_no != current_id:
                current_id = dial_no
                if len(current_dial) > 0:
                    dial_struct.append(sorted(current_dial, key=lambda x: turn_no))
                    current_dial = []
                current_dial.append(turn)
            else:
                current_dial.append(turn)
        else:
            dial_struct.append(sorted(current_dial, key= lambda x: turn_no))

        cls_error_fasit = 0
        cls_error_pred = 0
        pos_error = 0
        dontcare_error = 0
        tralse = True
        slot_misses = FreqDist()

        for dial in dial_struct:
            nully = True
            pos_correct = {k:True for k in self.slots}
            for turn in dial:
                with torch.no_grad():
                    class_logits, pos_logits = self(
                            { k: check_cuda_long(v) for k, v in  
                                self.tokenizer.encode_plus(
                                    turn.text_a, turn.text_b, return_tensors="pt"
                                ).items()
                                }, train=False
                    )
                predset = PredictionSet(*class_logits, *pos_logits)
                
                for slot in ["food", "area", "price range"]:
                    p_slot = slot if slot != "price range" else "pricerange"
                    pred_cls = predset.cat_map[p_slot]["class"].argmax(-1).item()
                    pred_pos = predset.eval_format(p_slot)

                    #if tralse:
                    #    print(p_slot)
                    #    print(predset.cat_map[p_slot]["class"])
                    #    print(turn.class_label[slot])
                    #    print(turn.session_id)
                    #    print(turn.guid)
                    #    q = input()
                    #    if q == "q":
                    #        tralse = False

                    if turn.class_label[slot] == "copy_value":
                        nully = False
                        if pred_cls != 2:
                            pos_correct[p_slot] = False
                            cls_error_fasit += 1
                            slot_misses[slot] += 1
                            continue

                        labs = turn.text_a_label[slot].copy()
                        labs.extend(turn.text_b_label[slot])
                        while len(labs) < len(pred_pos):
                            labs.append(0)

                        if labs != pred_pos: 
                            pos_correct[p_slot] = False
                            pos_error += 1
                        else:
                            pos_correct[p_slot] = True

                    elif turn.class_label[slot] == "dontcare":
                        if pred_cls != 1:
                            pos_correct[p_slot] = False
                            nully = False
                            dontcare_error += 1
                            slot_misses[slot] += 1
                        else:
                            pos_correct[p_slot] = True
                    else:
                        if pred_cls != 0:
                            slot_misses[slot] += 1
                            pos_correct[slot] = False
                            nully = False
                            cls_error_pred += 1
                            continue
                if nully:
                    continue
                if all(pos_correct.values()):
                    good += 1
                else:
                    bad += 1
        print("#"*30)
        print("Slots missed:")
        for k,v in slot_misses.most_common():
            print(f"{k}: {v} times")
        print("class fasit error:", cls_error_fasit)
        print("class pred error:", cls_error_pred)
        print("dontcare error:", dontcare_error)
        print("pos error:", pos_error)
        print("good cnt:", good)
        print("bad cnt:", bad)
        print(good/(good+bad))
        print("#"*30)
        return good/(good+bad)

    def forward(self, X, train=True):

        seq, comb = self.bert(**X).to_tuple()

        if train:
            comb = self.dropout(comb)
            seq = self.dropout(seq)

        a_food = self.softmax(self.food_a(comb))
        a_area = self.softmax(self.area_a(comb))
        a_price = self.softmax(self.price_a(comb))

        food_ab = self.alphabeta_food(seq)
        area_ab = self.alphabeta_area(seq)
        price_ab = self.alphabeta_price(seq)

        def get_alphabeta(tensor):
            alphamax = self.softmax(tensor[:, :, 0])
            betamax = self.softmax(tensor[:, :, 1])
            return alphamax, betamax

        foodstart, foodend = get_alphabeta(food_ab)
        areastart, areaend = get_alphabeta(area_ab)
        pricestart, priceend = get_alphabeta(price_ab)

        return ((a_food, a_area, a_price),
                ((foodstart, foodend),
                (areastart, areaend),
                (pricestart, priceend)))

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
            price_labels):
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

    def idx_for_cat(self, cat):
        return torch.where(self.cat_map[cat]["class"] == 2)

    def to_bert_format(self):
        return {
            "input_ids": self.inputs,
            "attention_mask": self.masks,
            "token_type_ids": self.types
        }


def pad_to_max_len(tokens, max_len, pad_token):
    while len(tokens) != max_len:
        tokens.append(pad_token)
    return tokens

if __name__ == '__main__':
    import sys

    arg = sys.argv[1]
    bn = BertNet()
    if arg == "train":
        bn.fit()
    elif arg == "majority":
        bn.fit(mode="majority")
    elif arg == "snorkel":
        bn.fit(mode="snorkel")
    elif arg == "validate":
        bn.load_state_dict(torch.load("./bertdst-light.pt"))
        bn.bert.load_state_dict(torch.load("light-bertstate.pt"))
        bn.predict()
    elif arg == "test":
        bn.load_state_dict(torch.load("./bertdst-light-train.pt"))
        bn.bert.load_state_dict(torch.load("light-bertstate-train.pt"))
        bn.predict(mode="test")
    elif arg == "dev":
        bn.load_state_dict(torch.load("./bertdst-light.pt"))
        bn.bert.load_state_dict(torch.load("light-bertstate.pt"))
        bn.predict()
    elif arg == "majority_test":
        bn.load_state_dict(torch.load("./bertdst-light-majority.pt"))
        bn.bert.load_state_dict(torch.load("light-bertstate-majority.pt"))
        bn.predict(mode="test")
    elif arg == "snorkel_test":
        bn.load_state_dict(torch.load("./bertdst-light-snorkel.pt"))
        bn.bert.load_state_dict(torch.load("light-bertstate-snorkel.pt"))
        bn.predict(mode="test")

