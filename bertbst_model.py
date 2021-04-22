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
import time
#from Levenshtein import distance
# need pure python package for saga
from pylev import levenshtein
distance = levenshtein

import dataset_dstc2
import tokenization
from dst_util import tokenize_text_and_label, get_token_label_ids, get_start_end_pos
from get_data import DataFetcher

BERT_VOCAB_LOC = "/usr/local/Development/bert/vocab.txt"
MODEL_DIR = "/run/media/adriantysnes/HDD/models/"

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
            os.path.join(self.BASE_PATH, f"dstc2_{version}_en.json"),
            self.SLOTS, version_type, **kwargs
        )
        grouped_set = self._group_set(loaded_set)
        loaded_map[version] = grouped_set
        return grouped_set

    def _group_set(self, example_set):
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

    def __init__(self):
        self.tokenizer = tokenization.FullTokenizer(
                vocab_file=BERT_VOCAB_LOC,
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
        turn_based_bert = {}
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
                if len(tokens_b) == 0:
                    tokens_b.append("[UNK]")

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
            turn_based_bert[turn.guid] = self._sort_asr(turn, bert_tokenizer) 

        return FeatureBatchSet(input_ids, masks, types, starts, ends, food_labels,
                area_labels, price_labels, turn_based_bert)

    def _sort_asr(self, turn, bt):
        all_feats = {"input_ids": [], "attention_mask": [], "token_type_ids": []}
        all_asrs = []
        for txt, asr in zip(turn.all_texts, turn.asrs):
            if len(txt) == 0:
                txt.append("[UNK]")
            berted = bt.encode_plus(turn.text_a, txt,
                    padding='max_length', max_length=80)
            for k,v in berted.items():
                all_feats[k].append(v)
            all_asrs.append(asr)
        return all_feats, all_asrs


class BertNet(nn.Module):

    #BERT_VERSION = "bert-base-uncased"
    BERT_VERSION = "prajjwal1/bert-medium"

    def __init__(self):
        super().__init__()
        self.preprep = PrePrepped()
        self.loader = DataLoader()
        self.tokenizer = BertTokenizer.from_pretrained(self.BERT_VERSION)
        self.bert = BertModel.from_pretrained(self.BERT_VERSION)
        emb_dim = self.bert.get_input_embeddings().embedding_dim
        self.emb_dim = emb_dim
        self.food_a = nn.Linear(emb_dim, 3)
        self.area_a = nn.Linear(emb_dim, 3)
        self.price_a = nn.Linear(emb_dim, 3)

        self.alphabeta_food = nn.Linear(emb_dim, 2)
        self.alphabeta_area = nn.Linear(emb_dim, 2)
        self.alphabeta_price = nn.Linear(emb_dim, 2)

        self.activation = nn.Sigmoid()
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

        self.optim = Adam(self.parameters(), lr=1e-5)

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

        dataset = self.preprep.fetch_set(mode, use_asr_hyp=5)
        
        for epoch in range(self.epochs):
            batch_generator = self.loader.fetch_batch(
                    dataset,
                    self.train_bsize, self.slots, self.tokenizer
            )
            self.train()
            self.bert.train()
            logger.info("Starting epoch # %s" % (epoch + 1))
            epoch_class_loss, epoch_pos_loss = 0, 0
            tralse = True
            partition = 0
            prevlisted = 10
            for batch in batch_generator:
                partition += round((self.train_bsize/len(dataset))*100, 3)
                if partition > prevlisted:
                    logger.info(f"{prevlisted}% of dataset processed")
                    prevlisted += 10 
                self.zero_grad()
                class_logits, pos_logits = self(batch)
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

                    try:
                        class_loss = self.loss_fn(
                                predset.cat_map[slot_name]["class"],
                                batch.cat_map[slot_name]["class"]
                        ) * .8
                    except ValueError:
                        class_loss = check_cuda_float([0])

                    epoch_class_loss += class_loss.item()
                    loss += first_position_loss + second_position_loss + class_loss

                loss.backward()
                self.optim.step()
            logger.info("Position loss for epoch: %s" % epoch_pos_loss)
            logger.info("Class loss for epoch: %s" % epoch_class_loss)
            logger.info("Dev loss:")
            devacc = self.predict()
            torch.save(self.state_dict(),
                    open(f"{MODEL_DIR}bertdst-light-{mode}-{epoch}.pt", "wb")
            )
            torch.save(self.bert.state_dict(),
                    open(f"{MODEL_DIR}light-bertstate-{mode}-{epoch}.pt", "wb")
            )
            #if devacc > best_acc and devacc > .5:
            #    best_acc = devacc
            #    torch.save(self.state_dict(), open(f"./bertdst-light-{mode}.pt", "wb"))
            #    torch.save(self.bert.state_dict(), open(f"./light-bertstate-{mode}.pt", "wb"))
            logger.info("#" * 30)


    def develop(self, mode="validate"):
        self.eval()
        self.bert.eval()
        dial_struct = self._fetch_predictset(mode)
        random.shuffle(dial_struct)

        for dial in dial_struct:
            goal_labs = {k:True for k in self.slots}
            for turn in dial:
                predset = self._get_predset(turn)
                for slot in ["food", "area", "price range"]:
                    p_slot = slot if slot != "price range" else "pricerange"
                    pred_cls = predset.cat_map[p_slot]["class"].argmax(-1).item()
                    pred_pos = predset.eval_format(p_slot)

                    if pred_cls == 2:
                        if turn.class_label[slot] != "copy_value":
                            goal_labs[p_slot] = False
                            print("Wrong class prediction for dial", turn.session_id)
                            self._turn_error_print(turn, pred_cls, pred_pos, slot)
                            print("¤"*30)
                            print()
                            input()
                        else:
                            print(f"Right class prediction for slot {slot}")
                            self._turn_error_print(turn, pred_cls, pred_pos, slot)
                            print("¤"*40)
                            print()


    def _turn_error_print(self, turn: dataset_dstc2.InputExample,
                                predicted_class: str,
                                predicted_idx: int,
                                slot: str):
        pred_pos = np.where(np.array(predicted_idx) == 1)[0]
        pred_value = turn.text_a + turn.text_b
        pred_slotval = []
        for p in pred_pos:
            pred_slotval.append(pred_value[p])

        hit_index = np.array(turn.text_a_label[slot] + turn.text_b_label[slot])
        hit_goal = np.where(hit_index == 1)[0]

        inp_class_labs = ["none", "dontcare", "copy_value"]
            
        print("Predicted class:", predicted_class)
        print("Actual class:", inp_class_labs.index(turn.class_label[slot]))
        print("Predicted indexes:", pred_pos)
        print("Actual indexes:", hit_goal)
        print("¤"*30)
        print()
        print("For slot", slot)
        print("Turn text_a:", turn.text_a)
        print("Turn text_b:", turn.text_b)
        print("Turn text_a_label:", turn.text_a_label)
        print("Turn text_b_label:", turn.text_b_label)
        print("For predicted value", " ".join(pred_slotval))
        print("¤"*30)
        print()
        print("Class label:", turn.class_label[slot])
        print("Indexes to hit:", hit_goal)


    def predict_v2(self, mode="validate"):
        self.eval()
        self.bert.eval()
        good, bad = 0, 0
        dials = self._fetch_predictset(mode)
        real_set = DataFetcher.fetch_clean_dstc(mode, parse_labels=True)
        for dial in dials:
            nully = True
            corr_slot = {k:True for k in self.slots}
            predz = {k: "none" for k in self.slots}
            for turn in dial:
                slot_plus_idx = {k: 0 for k in self.slots}
                predset = self._get_predset(turn)
                predstring = ""
                for slot in ["food", "area", "price range"]:
                    p_slot = slot if slot != "price range" else "pricerange"
                    pred_cls = predset.cat_map[p_slot]["class"].argmax(-1).item()
                    pred_pos = predset.eval_format(p_slot)
                    if turn.class_label[slot] == "copy_value":
                        nully = False
                        if pred_cls != 2:
                            predstring += ",bad"
                            corr_slot[p_slot] = False
                            continue

                        labs = turn.text_a_label[slot].copy() +\
                            turn.text_b_label[slot].copy()
                        while len(labs) < len(pred_pos):
                            labs.append(0)

                        for xlab, xtext in zip([turn.text_a_label, turn.text_b_label],
                                    [turn.text_a, turn.text_b]):
                                if sum(xlab[slot]) > 0:
                                    rel = np.array(xlab[slot])
                                    num = np.where(rel == 1)[0]
                                    tru_idx = []
                                    for idx in num:
                                        tru_idx.append(xtext[idx])
                                    predz[p_slot] = " ".join(tru_idx)

                        if labs != pred_pos:
                            corr_slot[p_slot] = False
                            predstring += ",bad"
                        else:
                            corr_slot[p_slot] = True
                            slot_plus_idx[p_slot] = labs
                            predstring += ",good"

                    elif turn.class_label[slot] == "dontcare":
                        nully = False
                        if pred_cls != 1:
                            predstring += ",bad"
                            corr_slot[p_slot] = False
                        else:
                            corr_slot[p_slot] = True
                            slot_plus_idx[p_slot] = labs
                            predstring += ",good"
                            predz[p_slot] = "dontcare"
                    else:
                        if pred_cls != 0:
                            nully = False
                            predstring += ",bad"
                            corr_slot[p_slot] = False
                        else:
                            predstring += ",good"
                            
                if nully:
                    continue
                if predstring == ",good,good,good" and all(corr_slot.values()):
                    good += 1
                    real_goals = [dial for dial in real_set.dialogues
                            if dial.id == turn.session_id]
                    real_turn = real_goals[0].turns[int(turn.guid.split("-")[-1])]
                    real_labs = real_turn.goal_labels
                    print(turn.session_id)
                    print("-"*30)
                    print(turn.text_a)
                    print(turn.text_b)
                    print(predz)
                    print(real_labs)
                    print("wtf")
                    input()

                    if predz != real_turn.goal_labels:
                        for k,v in predz.items():
                            if k not in real_labs:
                                print(turn.text_a)
                                print(turn.text_b)
                                print(predz)
                                print(real_labs)
                                print("wtf")
                                input()
                            if real_labs[k] != predz[k]:
                                if real_labs[k] in SEMANTIC_DICT and v in \
                                        SEMANTIC_DICT[real_labs[k]]:
                                            continue

                                print(real_labs)
                                print(predz)
                                print(turn.session_id)
                                input()

                else:
                    bad += 1
        print("goods:", good)
        print("bads:", bad)
        print("acc:", good/(good + bad))


    def predict(self, mode="validate"):
        self.eval()
        self.bert.eval()
        good, bad = 0, 0
        
        cls_error_fasit = 0
        cls_error_pred = 0
        pos_error = 0
        dontcare_error = 0
        slot_misses = FreqDist()
        dial_struct = self._fetch_predictset(mode)
        currid = ""
        lastid = ""
        eval_json = {}

        for dial_idx, dial in enumerate(dial_struct):
            eval_json[dial[0].session_id] = []
            nully = True
            true_gold_labs = {}
            preddies =  {}
            pos_correct = {k:True for k in self.slots}
            for turn in dial:
                turn_no = int(turn.guid.split("-")[-1])
                eval_json[turn.session_id].append({})
                for k,v in turn.class_label.items():
                    if v != "none":
                        true_gold_labs[k] = v
                    true_gold_labs
                if currid != turn.session_id:
                    currid = turn.session_id

                predset = None
                if len(turn.text_b) > 0:
                    with torch.no_grad():
                        txts, asr = self.loader._sort_asr(turn, self.tokenizer)
                        fbs = FeatureBatchSet([], [], [], [], 
                                [], [], [], [], {turn.guid: (txts, asr)}, pred=True)
                        class_logits, pos_logits = self(fbs, train=False)
                    predset = PredictionSet(*class_logits, *pos_logits)

                for slot in ["food", "area", "price range"]:
                    p_slot = slot if slot != "price range" else "pricerange"
                    if predset:
                        pred_cls = predset.cat_map[p_slot]["class"].argmax(-1).item()
                        pred_pos = predset.eval_format(p_slot)
                        if pred_cls == 2:
                            eval_json[turn.session_id][-1]["pred_pos"] = \
                                    [pred_pos.index(l) for l in pred_pos if l > 0]

                    else:
                        pred_cls = 0

                    if pred_cls > 0:
                        preddies[slot] = pred_cls
                    if turn.class_label[slot] == "copy_value":
                        nully = False
                        labs = turn.text_a_label[slot].copy()
                        labs.extend(turn.text_b_label[slot])
                        while len(labs) < len(pred_pos):
                            labs.append(0)

                        eval_json[turn.session_id][-1]["goal_pos"] = \
                                [labs.index(l) for l in labs if l > 0]

                        if pred_cls != 2:
                            pos_correct[p_slot] = False
                            cls_error_fasit += 1
                            slot_misses[slot] += 1
                            continue



                        if labs != pred_pos:
                            pos_correct[p_slot] = False
                            pos_error += 1
                        else:
                            pos_correct[p_slot] = True

                    elif turn.class_label[slot] == "dontcare":
                        nully = False
                        if pred_cls != 1:
                            pos_correct[p_slot] = False
                            dontcare_error += 1
                            slot_misses[slot] += 1
                        else:
                            pos_correct[p_slot] = True
                    elif turn.class_label[slot]:
                        if pred_cls != 0:
                            slot_misses[slot] += 1
                            pos_correct[slot] = False
                            nully = False
                            cls_error_pred += 1
                            continue
                eval_json[turn.session_id][-1]["transcript"] = turn.text_b
                for k,v in preddies.items():
                    eval_json[turn.session_id][-1][f"pred_{k}"] = v
                map = {
                    "copy_value": 2,
                    "none": 0,
                    "dontcare": 1,
                    "unpointable": 3
                }
                for k,v in true_gold_labs.items():
                    eval_json[turn.session_id][-1][f"real_{k}"] = map[v]
                if nully:
                    eval_json[turn.session_id][-1]["status"] = "null"
                    continue
                if all(pos_correct.values()):
                    good += 1
                    eval_json[turn.session_id][-1]["status"] = "good"
                else:
                    eval_json[turn.session_id][-1]["status"] = "bad"
                    bad += 1
        if mode == "test":
            json.dump(eval_json, open("./evaljson.json", "w"))
        logger.info("#"*30)
        logger.info("Slots missed:")
        for k,v in slot_misses.most_common():
            logger.info(f"{k}: {v} times")
        logger.info("class fasit error:", cls_error_fasit)
        logger.info("class pred error:", cls_error_pred)
        logger.info("dontcare error:", dontcare_error)
        logger.info("pos error:", pos_error)
        logger.info("good cnt:", good)
        logger.info("bad cnt:", bad)
        logger.info(good/(good+bad))
        logger.info("#"*30)
        return good/(good+bad)

    def forward(self, X, train=True):

        bsize = self.train_bsize if train else 1
        seq = check_cuda_float(torch.zeros((bsize, 80, self.emb_dim)))
        comb = check_cuda_float(torch.zeros((bsize, self.emb_dim)))
        dial_idx = -1
        curr_dial = ""
        for dial, feats in X.asr_feats.items():
            bert_dict, asr = feats
            if curr_dial != dial:
                curr_dial = dial
                dial_idx += 1
            new_seq, new_comb = self.bert(**bert_dict).to_tuple()
            for ns in range(len(asr)):
                seq[dial_idx] += (new_seq[ns] * asr[ns])
                comb[dial_idx] += (new_comb[ns] * asr[ns])
            
        #seq, comb = self.bert(**X).to_tuple()

        if train:
            comb = self.dropout(comb)
            seq = self.dropout(seq)

        a_food = self.food_a(comb)
        a_area = self.area_a(comb)
        a_price = self.price_a(comb)

        food_ab = self.alphabeta_food(seq)
        area_ab = self.alphabeta_area(seq)
        price_ab = self.alphabeta_price(seq)

        def get_alphabeta(tensor):
            alphamax = tensor[:, :, 0]
            betamax = tensor[:, :, 1]
            return alphamax, betamax

        foodstart, foodend = get_alphabeta(food_ab)
        areastart, areaend = get_alphabeta(area_ab)
        pricestart, priceend = get_alphabeta(price_ab)

        return ((a_food, a_area, a_price),
                ((foodstart, foodend),
                (areastart, areaend),
                (pricestart, priceend)))

    def _get_predset(self, turn, train=False):
        with torch.no_grad():
            class_logits, pos_logits = self(
                { k: check_cuda_long(v) for k, v in  
                    self.tokenizer.encode_plus(
                        turn.text_a, turn.text_b, return_tensors="pt"
                    ).items()
                }, train=train
            )
            return PredictionSet(*class_logits, *pos_logits)

    def _fetch_predictset(self, mode):
        """
        Helper function to generate a featureset for the predict and develop functions.
        """
        featureset = self.preprep.fetch_set(mode, use_asr_hyp=5, exclude_unpointable=False)
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
        return dial_struct


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
        
        if not pred:
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
        self.asr_feats = self._format_all_feats(all_feats)

    def idx_for_cat(self, cat):
        return torch.where(self.cat_map[cat]["class"] == 2)

    def to_bert_format(self):
        return {
                "main": {
                    "input_ids": self.inputs,
                    "attention_mask": self.masks,
                    "token_type_ids": self.types
                },
                "asr": self.asr_feats
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


if __name__ == '__main__':
    import sys

    arg = sys.argv[1]
    bn = BertNet()
    if arg in ("train", "majority", "snorkel", "majority_pattern_matching",
            "snorkel_pattern_matching"):
        bn.fit(mode=arg)
    elif arg in ("validate", "test", "majority_test", "majority_pattern_matching_test",
            "snorkel_test", "snorkel_pattern_matching_test"):
        predmode = "test" if arg != "validate" else arg
        if arg in ("test", "validate"):
            arg = "train"
        if len(arg.split("_")) == 2:
            arg = arg.split("_")[0]
        arg = arg if arg != "test" else "train"
        for ep in range(bn.epochs):
            print("For model # %s" % ep)
            bn.load_state_dict(torch.load(f"{MODEL_DIR}bertdst-light-{arg}-{ep}.pt"))
            bn.bert.load_state_dict(torch.load(f"{MODEL_DIR}light-bertstate-{arg}-{ep}.pt"))
            bn.predict(mode=predmode)

