import pandas as pd
import numpy as np

#data = pd.read_csv("./data/test/behaviors.csv", sep="\t")
# print(len(data))
#print(np.sum(data["clicked"] == 1))
a = np.load("./data/train/context_embedding.npy")
a[np.isnan(a)] = 0
print(np.any(np.isnan(a)))
np.save("./data/train/context_embedding.npy", a)
