import torch
import numpy
import load_data

#hyperparameters
BATCH_SIZE = 1

image_dataloader = load_data.load_data(BATCH_SIZE)

dataiter = iter(image_dataloader)
mini_batch = next(dataiter)

print(mini_batch["image"].shape)
print(mini_batch["target_labels"].shape)

