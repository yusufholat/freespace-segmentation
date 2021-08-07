import torch
import glob
import tqdm
input_shape = (224, 224)
n_classes=2
cuda=True
import numpy as np
import cv2
import os
from preprocess import tensorize_image
from mask_on_image import maskOnImage

model_path = 'model/first_model.pt'
model = torch.load(model_path)
model.eval()

test_data_path_list = glob.glob('../data/test_data/*')
test_data_list = os.listdir('../data/test_data/')

predict_file_path = '../data/predict_data/'

#for mask visualize from model output
# y=0
# for test_img in test:
#     tensorized_test_image = tensorize_image([test_img], input_shape, cuda)
#     output=model(tensorized_test_image)
#     print(output.shape)
#     torchvision.utils.save_image(output,('../data/model_image/'+str(y)+'.png'))
#     y+=1


def predict(test_data_path_list):

    for i in tqdm.tqdm(range(len(test_data_path_list))):
        batch_test = test_data_path_list[i:i+1]
        test_input = tensorize_image(batch_test, input_shape, cuda)
        output = model(test_input)
        out=torch.argmax(output,axis=1)
        out_cpu = out.cpu()
        outputs_list = out_cpu.detach().numpy()
        mask = np.squeeze(outputs_list,axis=0)
        
        predict = predict_file_path + test_data_list[i]
        
        img = cv2.imread(batch_test[0])
        mg = cv2.resize(img,(224,224))
        
        
        cpy_img  = mg.copy()
        mg[mask==0 ,:] = (155, 0, 125)
        opac_image = (mg/2+cpy_img/2).astype(np.uint8)
        
        cv2.imwrite(predict,opac_image)

predict(test_data_path_list)