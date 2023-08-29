import numpy as np
import pickle
from annoy import AnnoyIndex

item_vector = 2048
feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))
annoy_index = AnnoyIndex(item_vector,"angular")
for index,value in enumerate(feature_list):
    annoy_index.add_item(index,value)
    print(index)

annoy_index.build(15)
annoy_index.save("feature_annoy.ann")