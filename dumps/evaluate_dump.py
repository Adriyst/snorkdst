import json
import numpy as np
from nltk import FreqDist, ConditionalFreqDist
import matplotlib.pyplot as plt

PATH = "./%s_%s_dump.json"
BAR_WIDTH = .45


class Bar:

    def __init__(self, data, guid):
        self.data: FreqDist = data
        self.guid = guid

    def values(self, labels):
        return [self.data[l] for l in labels]

    def value(self, label):
        return self.data[label]

class BarPlotter:

    BAR_MIN_LIMIT = 0

    def __init__(self):
        self.datas = []
        self.labels = []

    def add(self, bar_data: FreqDist, *args):
        b = Bar(bar_data, *args)
        self.datas.append(b)
        self.labels.extend(bar_data.keys())
        self.labels = list(sorted(set(self.labels)))


    def fill_datas(self):
        """
        When a count is not seen for a dataset, use 0.
        """
        for d in self.datas:
            for l in self.labels:
                if l not in d.data:
                    d.data[l] = 0


    def filter_datas(self):
        rem_labs = []
        for lab in self.labels:
            if sum([d.value(lab) for d in self.datas]) < self.BAR_MIN_LIMIT:
                rem_labs.append(lab)
        self.labels = [lab for lab in self.labels if lab not in rem_labs]

    def get_bar_range(self):
        """
        Create a linear space to format the bar chart. 
        """
        linrate = BAR_WIDTH / len(self.datas)
        linstart = -linrate * (len(self.datas) / 2)
        linend = linstart * -1
        return np.linspace(linstart, linend, num=len(self.datas))

    def plot(self):
        self.fill_datas()
        self.filter_datas()
        x = np.arange(len(self.labels))
        fig, ax = plt.subplots()
        rects = []
        barwidths = self.get_bar_range()
        for data_idx, data in enumerate(self.datas):
            rect = ax.bar(
                    x + barwidths[data_idx],
                    data.values(self.labels),
                    BAR_WIDTH/len(self.datas),
                    label=data.guid
            )
            rects.append(rect)
        ax.set_ylabel("Counts")
        ax.set_title("Fail count per turn in dialogue")
        ax.set_xticks(x)
        ax.set_xticklabels(self.labels)
        ax.legend()

        for rect in rects:
            ax.bar_label(rect, padding=3)
        
        fig.tight_layout()
        plt.show()



def avg_fail_turn(dials):
    return np.mean([d["fail_idx"] for d in dials if d["fail"]])

def median_fail_turn(dials):
    return np.median([d["fail_idx"] for d in dials if d["fail"]])

def fail_counts(dials):
    return FreqDist([d["fail_idx"] for d in dials])

def dials_with_no_fail(dials):
    return sum([0 if d["fail"] else 1 for d in dials])


def general_stats(data):
    avg_fail = avg_fail_turn(data)
    nofail = dials_with_no_fail(data)
    medianfail = median_fail_turn(data)
    failcnt = fail_counts(data)
    print(avg_fail)
    print(nofail)
    print(medianfail)
    print(failcnt.most_common())

def dials_only_gold_good(gold, maj, snork):
    right_dials = []
    for g_t, m_t, s_t in zip(gold, maj, snork):
        if not g_t["fail"] and m_t["fail"] and s_t["fail"]:
            right_dials.append((g_t, m_t, s_t))
    return right_dials

def plot_fails():
    plotter = BarPlotter()
    for agg in ("train", "majority", "snorkel"):
        for dataset in ("validate", "test"):
            data = json.load(open(PATH % (agg, dataset), "r"))
            fails = fail_counts(data)
            plotter.add(fails, f"{agg}-{dataset}")
    plotter.plot()

def plot_majsnorkfail(failed_dials):
    slotcnts = ConditionalFreqDist()
    cfd = ConditionalFreqDist()
    for g, m, s in failed_dials:
        for fs in m["fail_slots"]:
            cfd["majority"][fs] += 1
        for fs in s["fail_slots"]:
            cfd["snorkel"][fs] += 1
        slotcnts["majority"][m["fail_idx"]] += 1
        slotcnts["snorkel"][s["fail_idx"]] += 1


    plotter = BarPlotter()
    plotter.add(slotcnts["majority"], "majority-test")
    plotter.add(slotcnts["snorkel"], "snorkel-test")
    plotter.plot()


def main(f):
    gold = json.load(open(PATH % ("train", "test")))
    majority = json.load(open(PATH % ("majority", "test")))
    snorkel = json.load(open(PATH % ("snorkel", "test")))
    failed_dials = dials_only_gold_good(gold, majority, snorkel)
    cfd = ConditionalFreqDist()
    for _, maj, snork in failed_dials:
        for fs in maj["fail_slots"]:
            cfd["majority"][fs] += 1
        for fs in snork["fail_slots"]:
            cfd["snorkel"][fs] += 1

    maj_tot = sum(cfd["majority"].values())
    snork_tot = sum(cfd["snorkel"].values())
    tots = { "majority": maj_tot, "snorkel": snork_tot }

    print("Fail numbers:")
    for mode in ("majority", "snorkel"):
        for slot in ("food", "area", "price range"):
            print(f"{mode} - {slot}: {cfd[mode][slot] / tots[mode]}")

    print("all numbers:")
    real_cfd = ConditionalFreqDist()
    for dial in [d for d in majority if d["fail"] == True]:
        for fs in dial["fail_slots"]:
            real_cfd["majority"][fs] += 1
    for dial in [d for d in snorkel if d["fail"] == True]:
        for fs in dial["fail_slots"]:
            real_cfd["snorkel"][fs] += 1

    real_tots = { 
        "majority": sum(real_cfd["majority"].values()),
        "snorkel": sum(real_cfd["snorkel"].values())
    } 
    for mode in ("majority", "snorkel"):
        for slot in ("food", "area", "price range"):
            print(f"{mode} - {slot}: {real_cfd[mode][slot] / real_tots[mode]}")


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("file", type=str, 
            help="file to review, example majority_validate")
    args = parser.parse_args()
    main(args.file)
