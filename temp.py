import numpy as np

features = np.load("/workdir/data/3dgs_attribute_merge_pdis0001_knn5/train/scene0010_00/coord.npy", allow_pickle=True)
print(type(features))
print(features.shape)
print(features.dtype)
