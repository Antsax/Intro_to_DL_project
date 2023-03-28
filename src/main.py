import torch
import numpy
import data_loader

#hyperparameters
BATCH_SIZE = 5

image_dataloader = data_loader.load_data(BATCH_SIZE)

dataiter = iter(image_dataloader)
mini_batch = next(dataiter)

print(mini_batch["image"].shape)
print(mini_batch["target_labels"].shape)

