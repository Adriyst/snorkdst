import matplotlib.pyplot as plt
import numpy as np

DATA = {
    "gold_validate": { "class": 0.267, "position": 0.010 },
    "gold_test": {"class": 0.241, "position": 0.007},
    "majority_validate": { "class": 0.656, "position": 0.147 },
    "majority_test": { "class": 0.772, "position": 0.134 },
    "snorkel_validate": { "class": 0.852, "position": 0.131 },
    "snorkel_test": { "class": 1.077, "position": 0.010 }
}

CLASS_DATA = np.asarray([x["class"] for x in DATA.values()])
POSITION_DATA = np.asarray([x["position"] for x in DATA.values()])

grouped_data = {
        "gold": { "class": [0.267, 0.241], "position": [0.010, 0.007] },
        "majority": { "class": [0.656, 0.772], "position": [0.015, 0.013] },
        "snorkel": { "class": [0.852, 1.077], "position": [0.013, 0.010] }
}

width = 0.35
x = np.arange(2)
fig, ax = plt.subplots()

linrate = width / 3
linstart = -linrate * (3/ 2)
linend = linstart * -1
widths =  np.linspace(linstart, linend, num=3)

rects = []
for idx, (lab, vals) in enumerate(grouped_data.items()):
    rects.append(
        ax.bar(
            x - widths[idx],
            np.asarray(vals["position"]),
            width/2,
            label=lab
        )
    )

ax.set_ylabel("Loss")
ax.set_title("Loss for the different sets")
ax.set_xticks(x)
ax.set_xticklabels(["validate", "test"])
ax.legend()

for rect in rects:
    ax.bar_label(rect, padding=3)

fig.tight_layout()
plt.show()

