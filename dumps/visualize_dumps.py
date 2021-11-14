import json
import matplotlib.pyplot as plt
from evaluate_dump import BarPlotter, BAR_WIDTH
import numpy as np

def plot(datas):
    labels = ["food", "area", "pricerange"]
    x = np.arange(len(labels))
    fig, ax = plt.subplots()
    linrate = BAR_WIDTH / len(labels)
    linstart = -linrate * (len(labels) / 2)
    linend = linstart * -1
    spacey = np.linspace(linstart, linend, num=3)
    rects = []
    for data_idx, data in enumerate(datas):
        # { slot: [3] }
        print(data)
        print(spacey)
        rects.append(ax.bar(
            x + spacey[data_idx],
            np.asarray(list(data.values())[0]),
            BAR_WIDTH / 3,
            label = list(data.keys())[0]
        ))
    ax.set_ylabel("Rate of entire dataset")
    ax.set_title("Rate of occurrence of slot")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(["gold", "majority", "SNORKEL"])
    for rect in rects:
        ax.bar_label(rect, padding=3)
    fig.tight_layout()
    plt.show()
    

def plot_bar():
    data = json.load(open("./label_dist.json", "r"))
    # need to have { dataset: [food, area, price] }
    newdict = {0: [], 1: [], 2: []}
    for i in range(3):
        for slot in ("food", "area", "price"):
            newdict[i].append(data[f"{slot}_rel"][i])
    plot([dict({k:v}) for k,v in newdict.items()])

def plot_lines():
    data = json.load(open("./slot_overview.json", "r"))
    datas = {}
    for slot in ("food", "area", "price range"):
        datas[slot] = {}
        for dist in ("train", "majority", "snorkel"):
            datas[slot][f"y_{dist}"] = [x[1] for x in data[dist][slot]]
        padlen = max(map(len, datas[slot].values()))
        for dist in ("train", "majority", "snorkel"):
            while len(datas[slot][f"y_{dist}"]) < padlen:
                datas[slot][f"y_{dist}"].append(0)
        datas[slot]["x"] = range(1,padlen+1)
    for slot in ("food", "area", "price range"):
        plt.figure(num = 3, figsize=(8, 5))
        for dist in ("train", "majority", "snorkel"):
            plt.plot(datas[slot]["x"], datas[slot][f"y_{dist}"])
        plt.legend(["gold", "majority", "SNORKEL"])
        plt.show()

def main():
    plot_lines()
    plot_bar()

if __name__ == '__main__':
    main()
