import glob
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pylab as plt
import torch
import tqdm
from model_unet import FoInternNet
from preprocess import tensorize_image, tensorize_mask,image_mask_check
import shutil

######### PARAMETERS ##########
valid_size = 0.3
test_size  = 0.1
batch_size = 4
epochs = 30
cuda = True
input_shape = (224, 224)
n_classes = 2
###############################

# PREPARE IMAGE AND MASK LISTS
SRC_DIR = os.getcwd()
ROOT_DIR = os.path.join(SRC_DIR, '..')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
MASK_DIR = os.path.join(DATA_DIR, 'masks')
IMAGE_DIR = os.path.join(DATA_DIR, 'images')
AUG_IMAGE=os.path.join(DATA_DIR,'images_augmentation')
AUG_MASK=os.path.join(DATA_DIR,'masks_augmentation')
###############################


# PREPARE IMAGE AND MASK LISTS
image_path_list = glob.glob(os.path.join(IMAGE_DIR, '*'))
image_path_list.sort()

mask_path_list = glob.glob(os.path.join(MASK_DIR, '*'))
mask_path_list.sort()

# PREPARE IMAGE AND MASK LISTS (AUGMENTATITIONS)
aug_path_list = glob.glob(os.path.join(AUG_IMAGE, '*'))
aug_path_list.sort()
aug_mask_path_list = glob.glob(os.path.join(AUG_MASK, '*'))
aug_mask_path_list.sort()

# DATA CHECK
image_mask_check(image_path_list, mask_path_list)


# SHUFFLE INDICES
indices = np.random.permutation(len(image_path_list))

# DEFINE TEST AND VALID INDICES
test_ind  = int(len(indices) * test_size)
valid_ind = int(test_ind + len(indices) * valid_size)

# SLICE TEST DATASET FROM THE WHOLE DATASET
test_input_path_list = image_path_list[:test_ind]
test_label_path_list = mask_path_list[:test_ind]



# SLICE VALID DATASET FROM THE WHOLE DATASET
valid_input_path_list = image_path_list[test_ind:valid_ind]
valid_label_path_list = mask_path_list[test_ind:valid_ind]



# SLICE TRAIN DATASET FROM THE WHOLE DATASET
train_input_path_list = image_path_list[valid_ind:]
train_label_path_list = mask_path_list[valid_ind:]

# ADD AUGMENTATION IMAGES
aug_size=int(len(aug_mask_path_list)/2)
train_input_path_list=aug_path_list[:aug_size]+train_input_path_list+aug_path_list[aug_size:]
train_label_path_list=aug_mask_path_list[:aug_size]+train_label_path_list+aug_mask_path_list[aug_size:]


# DEFINE STEPS PER EPOCH
steps_per_epoch = len(train_input_path_list)//batch_size


# CALL MODEL
model = FoInternNet(n_classes=2)

# DEFINE LOSS FUNCTION AND OPTIMIZER
criterion =  nn.MSELoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
# optimizer = optim.RMSprop(model.parameters(),lr=0.002, momentum=0.9)
# optimizer = optim.Adam(model.parameters(), lr=0.002)

# IF CUDA IS USED, IMPORT THE MODEL INTO CUDA
if cuda:
    model = model. cuda()


# TRAINING THE NEURAL NETWORK
traning_loss_list=[]
validation_loss_list=[]

for epoch in range(epochs):
    print('\nEpoch:' + str(epoch+1) + '/' + str(epochs))
    running_loss = 0
    
    #we have to shuffle images and masks so that they are not sync
    paired_images = list(zip(train_input_path_list,train_label_path_list))
    np.random.shuffle(paired_images)
    train_input_path_list, train_label_path_list = zip(*paired_images)

    for ind in tqdm.tqdm(range(steps_per_epoch)):
        batch_input_path_list = train_input_path_list[batch_size*ind:batch_size*(ind+1)]
        batch_label_path_list = train_label_path_list[batch_size*ind:batch_size*(ind+1)]

        batch_input = tensorize_image(batch_input_path_list, input_shape, cuda)
        batch_label = tensorize_mask(batch_label_path_list, input_shape, n_classes, cuda)

        optimizer.zero_grad()

        outputs = model(batch_input)
        loss = criterion(outputs, batch_label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
       
        if ind == steps_per_epoch-1:
            
            traning_loss_list.append(running_loss)
            print('\ntraining loss on epoch {}: {}'.format(epoch+1, running_loss))
            val_loss = 0
            for (valid_input_path, valid_label_path) in zip(valid_input_path_list, valid_label_path_list):
                batch_input = tensorize_image([valid_input_path], input_shape, cuda)
                batch_label = tensorize_mask([valid_label_path], input_shape, n_classes, cuda)
                outputs = model(batch_input)
                loss = criterion(outputs, batch_label)
                val_loss += loss.item()
                validation_loss_list.append(val_loss)
                break
            
            print('validation loss on epoch {}: {}'.format(epoch+1, val_loss) + '\n')

for img in test_input_path_list:
    shutil.copy(img,'../data/test_data/')

for msk in test_label_path_list:
    shutil.copy(msk,'../data/test_mask_data/')


torch.save(model,'model/model.pt')
print("Model Saved!")

normalized_training= [float(i)/max(traning_loss_list) for i in traning_loss_list]
normalized_validation=[float(j)/max(validation_loss_list) for j in validation_loss_list]
plt.plot(normalized_training,label='training loss list')
plt.plot(normalized_validation,label='validation loss list')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend()
plt.show()
