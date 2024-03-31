import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
mention_linked_entity_score_list = [[48.80], [48.78], [48.81], [48.77], [48.79]]
mMscaler = MinMaxScaler()
mMscaler.fit(mention_linked_entity_score_list)
mMscaled_data = mMscaler.transform(mention_linked_entity_score_list)
mMscaled_data = pd.DataFrame(mMscaled_data)
print(mMscaled_data)
# normalized_scores = np.array(mention_linked_entity_score_list) / np.sum(mention_linked_entity_score_list)
# print(normalized_scores)

#mentino단위로 normalize하여서 threshold seleciton
#ColBERT 