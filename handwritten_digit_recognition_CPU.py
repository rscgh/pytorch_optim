# Import necessary packages


import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time

import os

from torch import optim
from torch import nn



from torchvision import datasets, transforms


'''

source: https://github.com/amitrajitbose/handwritten-digit-recognition
https://towardsdatascience.com/handwritten-digit-mnist-pytorch-977b5338e627?gi=8182445af9f6

more MNIST:
https://github.com/pytorch/examples/blob/master/mnist/main.py

https://teaching.pages.centralesupelec.fr/deeplearning-lectures-build/00-pytorch-fashionMnist.html
https://medium.com/swlh/pytorch-real-step-by-step-implementation-of-cnn-on-mnist-304b7140605a
https://nextjournal.com/gkoehler/pytorch-mnist


optimizer ~ adam:
https://github.com/pytorch/pytorch/blob/master/torch/optim/adam.py
https://github.com/pytorch/pytorch/blob/52f2db752d2b29267da356a06ca91e10cd732dbc/torch/optim/functional.py#L53

optimizer ~ SGD:
https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py

'''

## What visualizations should be displayed?

do_show_sample_img = False
do_example_pass = True
do_view_single_result = True;

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

# Download and load the training data
trainset = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
valset = datasets.MNIST('MNIST_data/', download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)



######################################################################
# Look at Sample images

if do_show_sample_img: 
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    print(type(images))
    print(images.shape)
    print(labels.shape)

    #plt.imshow(images[0].numpy().squeeze(), cmap='gray_r');
    #plt.show()

    figure = plt.figure()
    num_of_images = 60
    for index in range(1, num_of_images + 1):
        plt.subplot(6, 10, index)
        plt.axis('off')
        plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')
    plt.show()

######################################################################



# Layer details for the neural network
input_size = 784
hidden_sizes = [128, 64]
output_size = 10

# Build a feed-forward network
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))


model = nn.Sequential(nn.Linear(input_size, 100),nn.ReLU(), nn.Linear(100, output_size), nn.LogSoftmax(dim=1));
model = nn.Sequential(nn.Linear(input_size, 100), nn.Linear(100, output_size), nn.LogSoftmax(dim=1));
print(model)


criterion = nn.NLLLoss()

# Optimizers require the parameters to optimize and a learning rate


from DLR import DLR

#optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
optimizer = DLR(model.parameters(), lr=0.01, alpha=40)
#optimizer = DLR(model.parameters(), lr=0.01, alpha=20)
#optimizer = DLR(model.parameters(), lr=0.01, alpha=200)
#optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)




######################################################################
# Do one Example run
if do_example_pass:

    images, labels = next(iter(trainloader))
    images = images.view(images.shape[0], -1)

    logps = model(images)
    loss = criterion(logps, labels)

    print('Before backward pass: \n', model[0].weight.grad)

    loss.backward()

    print('After backward pass: \n', model[0].weight.grad)

    print('Initial weights - ', model[0].weight)

    images, labels = next(iter(trainloader))
    images.resize_(64, 784)

    # Clear the gradients, do this because gradients are accumulated
    optimizer.zero_grad()

    # Forward pass, then backward pass, then update weights
    output = model(images)
    loss = criterion(output, labels)
    loss.backward()
    print('Gradient -', model[0].weight.grad)

    # Take an update step and few the new weights
    optimizer.step()
    print('Updated weights - ', model[0].weight)


print(model[0].weight.abs().sum(axis=1))

def prepare_full_network_weight_grid(model,grad=False, without_recurr = True):

    layer_names = []
    populations = [layer for layer in model if "activation" not in str(layer.__class__)]
    weight_mats = [[x for x in layer.named_parameters()] for layer in populations]

    num_neur_p_layer = []
    layer_start_ind = [int(0)]
    last_out_feat= None

    for layer, n in zip(populations, range(len(populations))): 

        layer_name = str(layer).split('(')[0] + str(n);                       
        layer_names = layer_names +[layer_name]

        num_neur_p_layer =  num_neur_p_layer + [layer.in_features]
        last_out_feat = layer.out_features

        layer_start_ind = layer_start_ind + [np.array(num_neur_p_layer).sum().astype(int)]


    num_neur_comb = np.array(num_neur_p_layer).sum() + last_out_feat;
    full_weight_grid = np.zeros((num_neur_comb, num_neur_comb ));

    for layer, n in zip(populations, range(len(populations))):

        weights = None
        for p in layer.named_parameters():
          if p[0] == 'weight': 
            citem = p[1] if grad==False else p[1].grad;
            weights = citem.detach()

        x =  layer_start_ind[n+1]
        y =  layer_start_ind[n]
        xt = x + layer.out_features
        yt = y + layer.in_features

        #print(x,":",xt, " | ",y,":", yt)

        full_weight_grid[y:yt , x :xt] = weights.T


    return full_weight_grid[:-last_out_feat,num_neur_p_layer[0]:] if without_recurr else full_weight_grid;


print(model)
full_weight_grid = prepare_full_network_weight_grid(model)
fig, ax = plt.subplots(figsize=(6, 6))
mx = np.absolute(full_weight_grid).max()
im = ax.imshow(full_weight_grid, cmap="PuOr_r", vmax = mx, vmin = -mx)
fig.tight_layout()
plt.show()

######################################################################


time0 = time()
epochs = 15
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
    
        # Training pass
        optimizer.zero_grad()
        
        output = model(images)
        loss = criterion(output, labels)
        
        #This is where the model learns by backpropagating
        loss.backward()
        
        #And optimizes its weights here
        optimizer.step()
        
        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))
print("\nTraining Time (in minutes) =",(time()-time0)/60)






######################################################################
# View Results:

def view_classify(img, ps):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()

    plt.show()

if do_view_single_result:
    images, labels = next(iter(valloader))
    img = images[0].view(1, 784)

    # Turn off gradients to speed up this part
    with torch.no_grad():
      logps = model(img)

    # Output of the network are log-probabilities, need to take exponential for probabilities
    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    print("Predicted Digit =", probab.index(max(probab)))
    view_classify(img.view(1, 28, 28), ps)



######################################################################
# Do the test dataset

correct_count, all_count = 0, 0
for images,labels in valloader:
  for i in range(len(labels)):
    img = images[i].view(1, 784)
    # Turn off gradients to speed up this part
    with torch.no_grad():
        logps = model(img)

    # Output of the network are log-probabilities, need to take exponential for probabilities
    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    pred_label = probab.index(max(probab))
    true_label = labels.numpy()[i]
    if(true_label == pred_label):
      correct_count += 1
    all_count += 1

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count))



######################################################################
# Plot the weights





full_weight_grid = prepare_full_network_weight_grid(model)
fig, ax = plt.subplots(figsize=(6, 6))
mx = np.absolute(full_weight_grid).max()
im = ax.imshow(full_weight_grid, cmap="PuOr_r", vmax = mx, vmin = -mx)
fig.tight_layout()
plt.show()
