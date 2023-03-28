import torch
import numpy
import data_loader

#hyperparameters
BATCH_SIZE = 5

image_loader_train, image_loader_validation = data_loader.load_data(BATCH_SIZE)

dataiter = iter(image_loader_train)
print(len(image_loader_train))
mini_batch = next(dataiter)

print(mini_batch["image"].shape)
print(mini_batch["target_labels"].shape)

