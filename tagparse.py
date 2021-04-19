# coding: utf-8
from snorkel_values_vote import ParaphrasingTagger
from snorkel_center import SnorkelCenter
sc = SnorkelCenter()
pt = ParaphrasingTagger(sc.dataframe, sc.value_voter)
states = pt.parse()
