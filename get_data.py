import os
import json
import normalized_json

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

class DataFetcher:

    base_path = "/usr/local/Development/master/data/"

    @staticmethod
    def fetch_data_by_corpus_name(name, subdirs=None, **kwargs):
        if name == "dstc":
            data = []
            subdirs = subdirs if subdirs else  ["Mar13_S0A0/", "Mar13_S0A1/", "Mar13_S1A0/", "Mar13_S1A1/"]
            for sd in subdirs:
                for logdir in os.listdir(DataFetcher.base_path + sd):
                    filename = DataFetcher.base_path + sd + logdir
                    label_obj = json.load(open(filename + "/label.json"))
                    dialogue_obj = json.load(open(filename + "/log.json"))
                    assert len(label_obj["turns"]) == len(dialogue_obj["turns"])
                    for i in range(len(dialogue_obj["turns"])):
                        label_obj["turns"][i]["system_transcription"] = \
                            dialogue_obj["turns"][i]["output"]["transcript"]
                        label_obj["turns"][i]["system_acts"] = \
                            dialogue_obj["turns"][i]["output"]["dialog-acts"]
                    data.append(label_obj)
        elif name == "frames":
            data = json.load(open(DataFetcher.base_path + "frames.json"))
        return normalized_json.NormalCorpusJson(data, name, **kwargs)
            
    @staticmethod
    def fetch_dstc_ontology():
        return json.load(open(DataFetcher.base_path + "dstc_ontology.json"))

    @staticmethod
    def fetch_dstc_test():
        return DataFetcher.fetch_data_by_corpus_name("dstc", subdirs=["testset/"], parse_labels=True)

    @staticmethod
    def fetch_clean_dstc(part, **kwargs):
        base_path = DataFetcher.base_path + "clean/"
        loaded = json.load(open(base_path + f"dstc2_{part}_en.json"))
        return normalized_json.NormalCorpusJson(loaded, "dstc", clean=True, **kwargs)

    @staticmethod
    def fetch_dstc_test(**kwargs):
        return DataFetcher.fetch_data_by_corpus_name("dstc", subdirs=["testset/"], **kwargs)

    def fetch_some_dialogues(self, name):
        if name != "dstc":
            raise NotImplementedError
        
        return DataFetcher.fetch_data_by_corpus_name(name, subdirs=["devset/"], parse_labels=True)

