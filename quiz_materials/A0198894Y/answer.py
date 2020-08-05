import torch
import torch.nn as nn
import numpy as np
import time
import foolbox as fb

from classifier import Net as Model

device = 'cuda' #CHANGE THIS LINE TO 'cpu' IF NO GPU TO USE
model = Model().to(device)
model.load_state_dict(torch.load('../model/Classifier.pt',
                                 map_location = torch.device(device)))
model.eval()
for p in model.parameters():
    p.requires_grad = False
    
dataset = 0 #CHANGE THIS LINE. RUNS FROM 0 TO 9
centers = np.load('../data/centers{}.npy'.format(dataset))

epsilon = 1/(2 ** 10)

# Getting a Batch
batchSize = 32
idx = np.random.randint(0, len(centers), (batchSize,))
c = torch.tensor(centers[idx], dtype = torch.float32,
                 device = device, requires_grad = True)

# Forward Propagation
logits = model(c) # Pass batch through model
softmaxPrediction = torch.softmax(logits, dim = -1) # Get Softmax Prediction
classPrediction = softmaxPrediction.argmax(dim = -1) # Get Class Prediction

# Backward Propagation
loss_fn = torch.nn.CrossEntropyLoss() # Create Loss Function
loss = loss_fn(logits, classPrediction) # Calculate Loss
loss.backward() # Backpropagate Loss

# Printing gradient of loss function at x
print(c.grad)
