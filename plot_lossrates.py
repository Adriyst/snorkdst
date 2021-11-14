import json
import matplotlib.pyplot as plt

dataset = "validate"
#dataset = "test"
data = json.load(open("./lossrates_woz.json"))
num_datapoints = int(len(data[f"train_{dataset}"]["class"]) * 0.03)

for m in ("train", "majority", "snorkel"):
    plt.plot(range(num_datapoints), data[f"{m}_{dataset}"]["class"][:num_datapoints])

plt.xlabel("Turn")
plt.ylabel("Loss")
plt.legend(["gold", "majority", "snorkel"])
plt.title(f"Loss rates for the {num_datapoints} most lossy turns for the {dataset} set")
plt.show()
