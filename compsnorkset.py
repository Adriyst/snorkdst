# coding: utf-8
import get_data
import snorkel_center
testset = get_data.DataFetcher.fetch_clean_dstc("test")
sc_2 = snorkel_center.SnorkelCenter(data=testset)
sc_2.majority_vote()
