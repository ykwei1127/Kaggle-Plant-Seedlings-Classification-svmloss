from model import VGG11
from torchvision import transforms
import csv
import torch, os
import numpy as np
import pandas as pd
from PIL import Image

weight_path = '/home/ykwei/Documents/svm_loss/weights/model.pth'
use_cuda = True
gpu_id = 0
classes = ['Black-grass','Charlock','Cleavers','Common Chickweed',
           'Common wheat','Fat Hen','Loose Silky-bent','Maize',
           'Scentless Mayweed','Shepherds Purse','Small-flowered Cranesbill','Sugar beet']

def test():  
    model = VGG11()
    weight = torch.load(weight_path)
    model.load_state_dict(weight)
    if use_cuda:
        torch.cuda.set_device(gpu_id)
        model = model.cuda()
    model.eval()

    testing_loss = 0.0
    predict_correct = 0

    data_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    with torch.no_grad():
        csvfile_test = open('/home/ykwei/Documents/svm_loss/sample_submission.csv')
        new_reader = csv.reader(csvfile_test)
        # 讀取csv標籤
        labels_test = []
        for line in new_reader:
            tmp = [line[0],line[1]]
            labels_test.append(tmp)
        csvfile_test.close()
        labels_test.remove(labels_test[0])

        result = pd.read_csv('/home/ykwei/Documents/snm_loss/sample_submission.csv')
        plist = []
        for i in range(len(labels_test)):
            image = Image.open('/home/ykwei/Documents/Kaggle-Plant-Seedlings-Classification/test/' + labels_test[i][0])
            image = data_transform(image).unsqueeze(0)
            if use_cuda:
                image =image.cuda()
            output = model(image)
            p = torch.argmax(output)
            p = p.item()
            plist.append(p)
            #data[i][1] = Label[p.item()]
            plist[i] = classes[plist[i]]
            if i % 100 == 0:
                print(i,'images finish predicting')

        # for i in range(len(plist)):
            

        result['species'] = plist
        #result.head()
        result.to_csv('submit.csv', index= False)
    

if __name__ == '__main__':
    test()
    print('test end')
