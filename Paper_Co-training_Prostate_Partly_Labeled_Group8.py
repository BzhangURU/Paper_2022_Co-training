#Introduction: This code will read prostate cancer dataset for training, validation and test. 
#Currently the training set is only available through material transfer agreement, please contact us for details.
#The validation and test sets are publicly availabel, you can set program_mode='only_test' for testing.
#To train start from scratch, set program_mode='normal_training'
#To train by resuming best model ever got in previous training, set program_mode='resume_best_training'
#To resume training from latest trained model, set program_mode='resume_latest_training'
#To only test already trained model on validation and test set, program_mode='only_test'
#To run program, simply implement "python Paper_Co-training_Prostate_Partly_Labeled_Group8.py" in terminal

import torchvision.transforms as transforms
import os
import torchvision.models as models
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import roc_auc_score
import copy
import warnings
warnings.filterwarnings('ignore')

gpu_id='0'#the index of GPU you use in training/testing
program_mode='only_test'#'normal_training', 'resume_best_training','resume_latest_training', 'only_test'
training_tile_size=224
batchsize=128
bs=50#how frequent to show on terminal, show once for each bs batchs
loss_contrastive_margin=40.0
loss_contrastive_weight=0.01#The value should be 0.2x(ratio of labeled data)
initial_lr=0.0001
lr_decay_factor=0.96
lr_step_size=1
total_epoch = 100
num_workers=8#This would greatly speed up training
my_modelname = 'Co_Training'
params_model={
        "image1_channels":1,
        "image2_channels":1,
        "classification_number":2,
    }

os.environ['CUDA_VISIBLE_DEVICES']=gpu_id
torch.cuda.empty_cache()

save_folder='Co-training_saved_models_and_results'
save_path = './Co-training_saved_models_and_results'

train_transformer2 = transforms.Compose([
    transforms.RandomRotation(360),#Random rotation to any angle, fill 0 for outside of image
    transforms.RandomCrop((training_tile_size,training_tile_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Normalize((0.1, 0.1), (0.1, 0.1))#You should calculate and input your (mean_H,mean_E),(std_H, std_E) in training set!!!
])

train_color_transformer_1channel=transforms.Compose([
    transforms.ColorJitter(brightness=0.1),
])

val_transformer2 = transforms.Compose([
    transforms.CenterCrop(training_tile_size),
    transforms.Normalize((0.1, 0.1), (0.1, 0.1))#You should calculate and input your (mean_H,mean_E),(std_H, std_E) in training set(not validation)
])

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class Co_Training_Model(nn.Module):
    def __init__(self,params):
        super(Co_Training_Model,self).__init__()
        image1_channels=params["image1_channels"]#
        image2_channels=params["image2_channels"]
        net_classification_number=params["classification_number"]

        self.model1=models.resnet18(pretrained=True)
        if image1_channels==1:
            self.my_model1_conv1=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            for filter_ind in range(64):
                for row in range(7):
                    for col in range(7):
                        self.my_model1_conv1.weight.data[filter_ind][0][row][col]=self.model1.conv1.weight.data[filter_ind][0][row][col]+\
                                                                        self.model1.conv1.weight.data[filter_ind][1][row][col]+\
                                                                        self.model1.conv1.weight.data[filter_ind][2][row][col]
            self.model1.conv1 = self.my_model1_conv1
        elif image1_channels!=3:
            self.my_model1_conv1=nn.Conv2d(image1_channels, 64, kernel_size=7, stride=2, padding=3)
            self.model1.conv1 = self.my_model1_conv1

        # change the output layer
        num_ftrs = self.model1.fc.in_features
        self.model1.fc = Identity()#do nothing
        print(self.model1)
        self.model2 = models.resnet18(pretrained=True)
        #change input channel
        if image2_channels==1:
            self.my_model2_conv1=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            for filter_ind in range(64):
                for row in range(7):
                    for col in range(7):
                        self.my_model2_conv1.weight.data[filter_ind][0][row][col]=self.model2.conv1.weight.data[filter_ind][0][row][col]+\
                                                                        self.model2.conv1.weight.data[filter_ind][1][row][col]+\
                                                                        self.model2.conv1.weight.data[filter_ind][2][row][col]
                        
            self.model2.conv1 = self.my_model2_conv1
        elif image2_channels==2:#HE images
            self.my_model2_conv1=nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            for filter_ind in range(64):
                for row in range(7):
                    for col in range(7):
                        self.my_model2_conv1.weight.data[filter_ind][0][row][col]=self.model2.conv1.weight.data[filter_ind][1][row][col]+\
                                                                        self.model2.conv1.weight.data[filter_ind][2][row][col]
                        self.my_model2_conv1.weight.data[filter_ind][1][row][col]=self.model2.conv1.weight.data[filter_ind][0][row][col]
            self.model2.conv1 = self.my_model2_conv1
        elif image2_channels!=3:
            self.my_model2_conv1=nn.Conv2d(image2_channels, 64, kernel_size=7, stride=2, padding=3)
            self.model2.conv1 = self.my_model2_conv1
        # change the output layer
        self.model2.fc = Identity()#do nothing
        self.fc=nn.Linear(num_ftrs,net_classification_number)
        self.myBN=torch.nn.BatchNorm1d(num_ftrs,affine=False, track_running_stats =False)

    def forward(self,image1,image2):#image1/image2 are batches
        image1_features=self.model1(image1)
        image2_features=self.model2(image2)
        image1_features_BN=self.myBN(image1_features)
        image2_features_BN=self.myBN(image2_features)
        avg_feature=torch.add(image1_features,image2_features)#element wise addtion
        avg_feature=torch.div(avg_feature, 2)
        raw_result=self.fc(avg_feature)
        return raw_result, image1_features_BN, image2_features_BN

def read_txt_category(txt_category):
    with open(txt_category) as file:
        lines = file.readlines()
    line_data = [line.strip() for line in lines]
    return line_data

class GPDataset(Dataset):
    def __init__(self, root_dir_H, root_dir_E, txt_benign, txt_cancer, txt_unlabeled, dataset_type, transform2=None):
        
        # root_dir_H: folder that saves all H-channel raw tiles
        # root_dir_E: folder that saves all E-channel raw tiles
        # txt_benign: txt file that saves name of labeled benign tiles
        # txt_cancer: txt file that saves name of labeled cancer tiles
        # txt_unlabeled: txt file that saves name of unlabeled tiles
        # File structure:
        # - root_dir_H
        #     - benign
        #         - 0.png
        #         - 1.png
        #         - ......
        #     - cancer
        #         - 0.png
        #         - 1.png
        #         - ......
        
        self.root_dir_H = root_dir_H
        self.root_dir_E = root_dir_E
        self.txt_category = [txt_benign, txt_cancer]
        self.classes = ['benign', 'cancer']
        self.num_cls = len(self.classes)
        self.img_list_H=[]
        self.img_list_E=[]
        for c in range(self.num_cls):#c is different class
            cls_list_H = [[os.path.join(self.root_dir_H,item), c] for item in read_txt_category(self.txt_category[c])]
            self.img_list_H += cls_list_H
            cls_list_E = [[os.path.join(self.root_dir_E,item), c] for item in read_txt_category(self.txt_category[c])]
            self.img_list_E += cls_list_E
        self.len_labeled=len(self.img_list_H)

        if dataset_type=="train_with_unlabeled":#if use "train", then no unlabeled data in training
            cls_list_H = [[os.path.join(self.root_dir_H,item), -1] for item in read_txt_category(txt_unlabeled)]
            self.img_list_H += cls_list_H
            cls_list_E = [[os.path.join(self.root_dir_E,item), -1] for item in read_txt_category(txt_unlabeled)]
            self.img_list_E += cls_list_E
        
        self.dataset_type=dataset_type
        self.transform2 = transform2

    def __len__(self):
        return len(self.img_list_H)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_H=np.asarray(Image.open(self.img_list_H[idx][0]))
        image_H=image_H.astype(float)
        image_H=image_H/255.0
        image_H = np.expand_dims(image_H, axis=0)

        image_E=np.asarray(Image.open(self.img_list_E[idx][0]))
        image_E=image_E.astype(float)
        image_E=image_E/255.0
        image_E = np.expand_dims(image_E, axis=0)
        tensor_of_image_H=torch.from_numpy(image_H.copy()).float()#change numpy array to tensor
        tensor_of_image_E=torch.from_numpy(image_E.copy()).float()
        
        if self.dataset_type=="train" or self.dataset_type=="train_with_unlabeled":
            tensor_of_image_H=train_color_transformer_1channel(tensor_of_image_H)
            tensor_of_image_E=train_color_transformer_1channel(tensor_of_image_E)
        tensor_of_image_HE=torch.cat((tensor_of_image_H, tensor_of_image_E), dim=0)

        tensor_of_image_HE=self.transform2(tensor_of_image_HE)#same crop, rotation... for H and E
        tensor_of_image1, tensor_of_image2=torch.split(tensor_of_image_HE, [1,1], dim=0)#dimension will be kept
        
        if int(self.img_list_H[idx][1])>=0:
            sample = {'img1': tensor_of_image1,
                    'img2': tensor_of_image2,
                    'label': int(self.img_list_H[idx][1]),
                    'labeled': 1.0}#has label or not
        else:
            sample = {'img1': tensor_of_image1,
                    'img2': tensor_of_image2,
                    'label': int(0),
                    'labeled': 0.0}#has label or not

        return sample

device = 'cuda'

def train(optimizer, epoch, train_loader):
    my_co_training_model.train()#will let program update each layer by backpropagation
    train_loss = 0
    train_correct = 0
    count_labeled=0
    avg_pos_dis=torch.tensor(0.0)
    avg_neg_dis=torch.tensor(0.0)
    for batch_index, batch_samples in enumerate(train_loader):
        # move data to device
        data1, data2, target, labeled = batch_samples['img1'].to(device), batch_samples['img2'].to(device), \
            batch_samples['label'].to(device), batch_samples['labeled'].to(device)
        
        output, batch_image1_feature, batch_image2_feature = my_co_training_model(data1, data2)
        #batch_image feaure are normalized
        loss_internal_12 = (batch_image1_feature - batch_image2_feature).pow(2).sum(1).sqrt()#sum in dim 1
        roll_batch_image2_feature=torch.roll(batch_image2_feature, 1, 0)#roll by 1 step, index x goes to x+1 to create random pair, dim=0
        loss_external_12=(batch_image1_feature - roll_batch_image2_feature).pow(2).sum(1).sqrt()
        avg_pos_dis=torch.add(avg_pos_dis,torch.sum(loss_internal_12))
        avg_neg_dis=torch.add(avg_neg_dis,torch.sum(loss_external_12))
        loss_contrastive=torch.sub(loss_internal_12,loss_external_12)
        loss_contrastive=torch.add(loss_contrastive,loss_contrastive_margin)
        loss_contrastive[loss_contrastive < 0.0] = 0.0
        loss_contrastive=torch.mul(loss_contrastive,loss_contrastive_weight)
        loss_contrastive=torch.sum(loss_contrastive)

        loss_func = nn.CrossEntropyLoss(reduction="none")
        loss_cross_entropy_batch=loss_func(output, target.long())

        labeled_boolean_list=[True if x > 0.5 else False for x in labeled]
        labeled_boolean_list=torch.tensor(labeled_boolean_list)

        loss_cross_entropy_batch=loss_cross_entropy_batch[labeled_boolean_list]
        loss_cross_entropy=torch.sum(loss_cross_entropy_batch)

        loss=loss_cross_entropy+loss_contrastive#This is total loss in whole batch, but will print avg loss in each case 
        if batch_index==0:
            print('loss_cross_entropy in first batch=')
            print(loss_cross_entropy)
            print('loss_contrastive in first batch=')
            print(loss_contrastive)
            print('avg pos dis in first batch=')
            print(torch.div(avg_pos_dis,float(batchsize)))
            print('avg neg dis in first batch=')
            print(torch.div(avg_neg_dis,float(batchsize)))
        train_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()#upgrade weight
        output_labeled=output[labeled_boolean_list]
        target_labeled=target[labeled_boolean_list]
        count_labeled+=torch.sum(labeled)
        if torch.numel(output_labeled)>0:#check number of elements
            pred = output_labeled.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(target_labeled.long().view_as(pred)).sum().item()
    
        if batch_index % bs == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}'.format(
                epoch, batch_index, len(train_loader),
                100.0 * batch_index / len(train_loader), loss.item()/ bs))
    scheduler.step()#update learning rate
    avg_pos_dis=torch.div(avg_pos_dis, float(len(train_loader)))
    avg_neg_dis=torch.div(avg_neg_dis, float(len(train_loader)))
    print('Train set: Average loss: {:.4f}, Accuracy in labeled data: {}/{} ({:.2f}%)\n'.format(
        train_loss/count_labeled, train_correct, count_labeled,
        100.0 * train_correct / count_labeled))
    f = open(save_folder+'/20220728_Co-training_Prostate_Partly_Labeled_group8_train01_{}.txt'.format(my_modelname), 'a+')
    f.write('Train set: Epoch: {} Average loss: {:.4f}, Accuracy in labeled data: {}/{} ({:.2f}%), avg pos pair dis: {}, avg neg pair dis: {}\n'.format(epoch, 
        train_loss/count_labeled, train_correct, count_labeled,
        100.0 * float(train_correct) / float(count_labeled), torch.div(avg_pos_dis,float(batchsize)), torch.div(avg_neg_dis,float(batchsize))))
    f.close()

def val(my_val_model):
    my_val_model.eval()
    val_loss = 0.0
    correct = 0
    loss_func = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        pred_list=[]
        score_list=[]
        target_list=[]
        for batch_index, batch_samples in enumerate(val_loader):
            data1, data2, target, labeled = batch_samples['img1'].to(device), batch_samples['img2'].to(device), \
                batch_samples['label'].to(device), batch_samples['labeled'].to(device)
            output, batch_image1_feature, batch_image2_feature = my_val_model(data1,data2)
            val_loss += loss_func(output, target.long())
            score = F.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.long().view_as(pred)).sum().item()
            target_np=target.long().cpu().numpy()
            pred_list= np.concatenate((pred_list, pred.cpu().numpy()), axis=None)
            score_list= np.concatenate((score_list, score.cpu().numpy()[:,1]), axis=None)
            target_list= np.concatenate((target_list, target_np), axis=None)
            val_acc=100.0 * correct / len(val_loader.dataset)
        
#######################This block is for visualizing results of each tile#########################################
            if program_mode =='only_test': # 'normal_training', 'resume_best_training','resume_latest_training', 'only_test'
                path_val_scores=save_folder+'/20220728_Co-training_Prostate_Partly_Labeled_group8_val_scores.txt'
                f = open(path_val_scores, 'a+')
                f.write('scores  prediction  target')
                for row in range(score.size(dim=0)):
                    for i in range(params_model["classification_number"]):
                        f.write('{:.4f} '.format(score[row][i]))
                    f.write('{} '.format(pred[row][0]))
                    f.write('{} '.format(target_np[row]))
                    f.write('\n')
                    f.write('\n')
                    f.write('\n')
                f.close()
                if batch_index==0:
                    print("Scores, prediction, target of each sample are saved in "+path_val_scores)
        #print('Count correct predictions in val is {}'.format(correct))
        val_loss/=float(len(val_loader.dataset))
           
    return target_list, score_list, pred_list, val_acc, val_loss

def test(my_test_model):
    my_test_model.eval()
    test_loss = 0.0
    correct = 0
    loss_func = nn.CrossEntropyLoss(reduction="sum")
    # Don't update model
    with torch.no_grad():
        pred_list=[]
        score_list=[]
        target_list=[]
        # Predict
        for batch_index, batch_samples in enumerate(test_loader):
            data1, data2, target, labeled = batch_samples['img1'].to(device), batch_samples['img2'].to(device), \
                batch_samples['label'].to(device), batch_samples['labeled'].to(device)
            output, batch_image1_feature, batch_image2_feature = my_test_model(data1,data2)
            test_loss += loss_func(output, target.long())
            score = F.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.long().view_as(pred)).sum().item()
            target_np=target.long().cpu().numpy()
            pred_list= np.concatenate((pred_list, pred.cpu().numpy()), axis=None)
            score_list= np.concatenate((score_list, score.cpu().numpy()[:,1]), axis=None)
            target_list= np.concatenate((target_list, target_np), axis=None)
            test_acc=100.0 * correct / len(test_loader.dataset)

#######################This block is for visualizing results of each tile#########################################
            if program_mode =='only_test': # 'normal_training', 'resume_best_training','resume_latest_training', 'only_test'
                path_test_scores=save_folder+'/20220728_Co-training_Prostate_Partly_Labeled_group8_test_scores.txt'
                f = open(path_test_scores, 'a+')#changable for HE
                f.write('scores  prediction  target')
                for row in range(score.size(dim=0)):
                    for i in range(params_model["classification_number"]):
                        f.write('{:.4f} '.format(score[row][i]))
                    f.write('{} '.format(pred[row][0]))
                    f.write('{} '.format(target_np[row]))
                    f.write('\n')
                    f.write('\n')
                    f.write('\n')
                f.close()
                if batch_index==0:
                    print("Scores, prediction, target of each sample are saved in "+path_test_scores)

        #print('Count correct predictions in test is {}'.format(correct))
        test_loss/=float(len(test_loader.dataset))

    return target_list, score_list, pred_list, test_acc, test_loss

if __name__ == '__main__':
    txt_root_dir='./dataset/dataset_list_with_unlabeled/label_unlabel_ratio_1_19_group_8/'
    trainset = GPDataset(
                            root_dir_H='./dataset/Training_set_256x256_H_my_code/',#dataset from our institution, please contact us for training set
                            root_dir_E='./dataset/Training_set_256x256_E_my_code/',
                            txt_benign=txt_root_dir+'benign/'+'20220301_benign_train_file_list.txt',
                            txt_cancer=txt_root_dir+'cancer/'+'20220301_cancer_train_file_list.txt',
                            txt_unlabeled=txt_root_dir+'unlabeled/'+'20220301_unlabeled_train_file_list.txt',
                            
                            dataset_type='train_with_unlabeled',#"train"(only use labeled data)  'train_with_unlabeled'(use both labeled, unlabeled)
                            transform2 = train_transformer2)
    valset = GPDataset(
                            root_dir_H='./dataset/Test_set_256x256_H_my_code/',#dataset from TCGA
                            root_dir_E='./dataset/Test_set_256x256_E_my_code/',
                            txt_benign=txt_root_dir+'benign/'+'20220301_benign_val_file_list.txt',
                            txt_cancer=txt_root_dir+'cancer/'+'20220301_cancer_val_file_list.txt',
                            txt_unlabeled=txt_root_dir+'VOID',

                            dataset_type='validation',
                            transform2 = val_transformer2)
    testset = GPDataset(
                            root_dir_H='./dataset/Test_set_256x256_H_my_code/',#dataset from TCGA
                            root_dir_E='./dataset/Test_set_256x256_E_my_code/',
                            txt_benign=txt_root_dir+'benign/'+'20220301_benign_test_file_list.txt',
                            txt_cancer=txt_root_dir+'cancer/'+'20220301_cancer_test_file_list.txt',
                            txt_unlabeled=txt_root_dir+'VOID',
                            
                            dataset_type='test',
                            transform2 = val_transformer2)
    print("Training set count:")
    print(trainset.__len__())
    print("Validation set count:")
    print(valset.__len__())
    print("Test set count:")
    print(testset.__len__())

    # Check whether the specified path exists or not
    isExist = os.path.exists(save_path)
    if not isExist:
        os.makedirs(save_path)
        print("The new directory: "+save_folder+ " for saving models is created!")

    train_loader = DataLoader(trainset, batch_size=batchsize, num_workers=num_workers, drop_last=False, shuffle=True)
    val_loader = DataLoader(valset, batch_size=batchsize, num_workers=num_workers, drop_last=False, shuffle=False)
    test_loader = DataLoader(testset, batch_size=batchsize, num_workers=num_workers, drop_last=False, shuffle=False)

    my_co_training_model = Co_Training_Model(params_model)
    get_best_model=0
    best_epoch_test_acc=0.0
    ################################################### train ######################################################
    if program_mode !='only_test': # 'normal_training', 'resume_best_training','resume_latest_training', 'only_test'
        start_epoch=1
        best_acc=0.0
        best_epoch=1
        if program_mode=='resume_best_training':# 'normal_training', 'resume_best_training', 'resume_latest_training', 'only_test'
            #choose current_best for path2weights and current_result_path
            path2weights=save_folder+'/20220728_Co-training_Prostate_Partly_Labeled_group8_current_best.pt'
            my_co_training_model.load_state_dict(torch.load(path2weights))
            current_result_path=save_folder+'/20220728_Co-training_Prostate_Partly_Labeled_group8_current_best_for_resuming.txt'
        elif program_mode=='resume_latest_training':# 'normal_training', 'resume_best_training','resume_latest_training', 'only_test'
            #choose current_latest for path2weights and current_result_path
            path2weights=save_folder+'/20220728_Co-training_Prostate_Partly_Labeled_group8_current_latest.pt'
            my_co_training_model.load_state_dict(torch.load(path2weights))
            current_result_path=save_folder+'/20220728_Co-training_Prostate_Partly_Labeled_group8_current_latest_for_resuming.txt'
        if program_mode=='resume_best_training' or program_mode=='resume_latest_training':
            current_result_file = open(current_result_path, 'r')
            current_epoch=current_result_file.readline()
            start_epoch=int(current_epoch)+1
            acc=float(current_result_file.readline())
            test_acc=float(current_result_file.readline())
            initial_lr=float(current_result_file.readline())
            best_epoch=int(current_result_file.readline())
            best_acc=float(current_result_file.readline())
            best_epoch_test_acc=float(current_result_file.readline())
            current_result_file.close()

        acc_list = []
        vote_pred = np.zeros(valset.__len__())
        vote_score = np.zeros(valset.__len__())

        optimizer = optim.Adam(my_co_training_model.parameters(), initial_lr)
        scheduler = StepLR(optimizer, step_size=lr_step_size, gamma=lr_decay_factor)
        device = torch.device("cuda:0")
        my_co_training_model.to(device)
        for epoch in range(start_epoch, total_epoch+1):
            print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
            current_lr=optimizer.param_groups[0]['lr']
            train(optimizer, epoch, train_loader)
            
            val_target_list, val_scorelist, val_pred_list, val_acc, val_loss = val(my_co_training_model)
            TP = ((val_pred_list == 1) & (val_target_list == 1)).sum()
            TN = ((val_pred_list == 0) & (val_target_list == 0)).sum()
            FN = ((val_pred_list == 0) & (val_target_list == 1)).sum()
            FP = ((val_pred_list == 1) & (val_target_list == 0)).sum()
            
            if TP + FP==0:
                precision=1.0
            else:
                precision = float(TP) / float(TP + FP)
            if TP+FN==0:
                recall=1.0
            else:
                recall = float(TP) / float(TP + FN)
            if recall+precision==0.0:
                F1=0.0
            else:
                F1 = 2 * recall * precision / (recall + precision)
            TPR = recall
            TNR = float(TN)/float(TN+FP)
            AUC = roc_auc_score(val_target_list, val_scorelist)
            
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print('The epoch is {}, average val accuracy: {:.4f}, average loss: {}, previous best val {} with test {} at epoch {}\n'.format(epoch, val_acc, val_loss, best_acc, best_epoch_test_acc, best_epoch))
            f = open(f'{save_folder}/20220728_Co-training_Prostate_Partly_Labeled_group8_val01_{my_modelname}.txt', 'a+')
            f.write('\n Val: The epoch is {}, average accuracy: {:.4f}, average loss: {},TP= {}, TN= {}, FN= {}, FP= {}, F1= {:.4f}, TPR= {:.4f}, TNR= {:.4f}, AUC= {:.4f}, Current Time ={}\n'.format(epoch, val_acc,
    val_loss,TP,TN,FN,FP,F1,TPR,TNR,AUC, current_time))
            f.close()

            if best_acc<val_acc:
                best_acc=val_acc
                best_epoch=epoch
                my_best_model=copy.deepcopy(my_co_training_model)
                get_best_model=1
                torch.save(my_co_training_model.state_dict(), save_folder+"/20220728_Co-training_Prostate_Partly_Labeled_group8_current_best.pt".format(my_modelname))
                test_target_list, test_score_list, test_pred_list, test_acc, test_loss = test(my_co_training_model)
                best_epoch_test_acc=test_acc
                current_result_path=save_folder+'/20220728_Co-training_Prostate_Partly_Labeled_group8_current_best_for_resuming.txt'
                current_result_file = open(current_result_path, 'w')#overwrite
                current_result_file.write(f'{epoch}\n')
                current_result_file.write(f'{val_acc}\n')
                current_result_file.write(f'{test_acc}\n')
                current_result_file.write(f'{current_lr}\n')
                current_result_file.write(f'{best_epoch}\n')
                current_result_file.write(f'{best_acc}\n')
                current_result_file.write(f'{best_epoch_test_acc}\n')
                current_result_file.close()
                
                TP = ((test_pred_list == 1) & (test_target_list == 1)).sum()
                TN = ((test_pred_list == 0) & (test_target_list == 0)).sum()
                FN = ((test_pred_list == 0) & (test_target_list == 1)).sum()
                FP = ((test_pred_list == 1) & (test_target_list == 0)).sum()
                
                if TP + FP==0:
                    precision=1.0
                else:
                    precision = float(TP) / float(TP + FP)
                if TP+FN==0:
                    recall=1.0
                else:
                    recall = float(TP) / float(TP + FN)
                if recall+precision==0.0:
                    F1=0.0
                else:
                    F1 = 2 * recall * precision / (recall + precision)
                
                TPR=recall
                TNR=float(TN)/float(TN+FP)
                AUC = roc_auc_score(test_target_list, test_score_list)

                print('test acc',test_acc)
                f = open(f'{save_folder}/20220728_Co-training_Prostate_Partly_Labeled_group8_test01_{my_modelname}.txt', 'a+')#changable for HE
                f.write('\n Test: The epoch is {}, average accuracy: {:.4f}, average loss: {},TP= {}, TN= {}, FN= {}, FP= {}, F1= {:.4f}, TPR= {:.4f}, TNR= {:.4f}, AUC= {:.4f}, Current Time ={}\n'.format(epoch, test_acc,
    test_loss,TP,TN,FN,FP,F1,TPR,TNR,AUC,current_time))
                f.close()

            if epoch%10==0:
                torch.save(my_co_training_model.state_dict(), save_folder+"/20220728_Co-training_Prostate_Partly_Labeled_group8_current_latest.pt".format(my_modelname))
                test_target_list, test_score_list, test_pred_list, test_acc, test_loss = test(my_co_training_model)
                current_result_path = save_folder+'/20220728_Co-training_Prostate_Partly_Labeled_group8_current_latest_for_resuming.txt'
                current_result_file = open(current_result_path, 'w')#overwrite
                current_result_file.write(f'{epoch}\n')
                current_result_file.write(f'{val_acc}\n')
                current_result_file.write(f'{test_acc}\n')
                current_result_file.write(f'{current_lr}\n')
                current_result_file.write(f'{best_epoch}\n')
                current_result_file.write(f'{best_acc}\n')
                current_result_file.write(f'{best_epoch_test_acc}\n')
                current_result_file.close()
                
        f = open(f'{save_folder}/20220728_Co-training_Prostate_Partly_Labeled_group8_val01_{my_modelname}.txt', 'a+')#changable for HE
        f.write('best epoch: {} best validation accuracy:{} test acc at best val: {}.pt'.format(best_epoch,best_acc,best_epoch_test_acc))
        f.close()        
    
    #############this block is only for reloading model and testing#########################################
    if program_mode =='only_test': # 'normal_training', 'resume_best_training','resume_latest_training', 'only_test'
        path2test_weights=save_folder+"/20220728_Co-training_Prostate_Partly_Labeled_group8_current_best.pt"#changable for HE
        my_co_training_model.load_state_dict(torch.load(path2test_weights))
        device = torch.device("cuda:0")
        my_co_training_model.to(device)
        val_target_list, val_scorelist, val_pred_list, val_acc, val_loss = val(my_co_training_model)
        TP = ((val_pred_list == 1) & (val_target_list == 1)).sum()
        TN = ((val_pred_list == 0) & (val_target_list == 0)).sum()
        FN = ((val_pred_list == 0) & (val_target_list == 1)).sum()
        FP = ((val_pred_list == 1) & (val_target_list == 0)).sum()
        if TP + FP==0:
            precision=1.0
        else:
            precision = float(TP) / float(TP + FP)
        if TP+FN==0:
            recall=1.0
        else:
            recall = float(TP) / float(TP + FN)
        if recall+precision==0.0:
            F1=0.0
        else:
            F1 = 2 * recall * precision / (recall + precision)
        print("Validation set: ")
        print('TP=',TP,'TN=',TN,'FN=',FN,'FP=',FP)
        print('TP+FP',TP+FP)
        print('precision',precision)
        print('recall',recall)
        print('F1',F1)
        acc = float(TP + TN) / float(TP + TN + FP + FN)
        TPR=recall
        TNR=float(TN)/float(TN+FP)
        print('acc',acc)
        AUC = roc_auc_score(val_target_list, val_scorelist)
        print('AUC', AUC)
        print('\n')
        
        f = open(f'{save_folder}/20220728_Co-training_Prostate_Partly_Labeled_group8_val01_load_model_{my_modelname}.txt', 'a+')
        f.write('\n Val: average accuracy: {:.4f}, TP= {}, TN= {}, FN= {}, FP= {}, F1= {:.4f}, TPR= {:.4f}, TNR= {:.4f}, AUC= {:.4f}\n'.format(val_acc,
        TP,TN,FN,FP,F1,TPR,TNR,AUC))
        f.close()

        test_target_list, test_score_list, test_pred_list, test_acc, test_loss = test(my_co_training_model)
        TP = ((test_pred_list == 1) & (test_target_list == 1)).sum()
        TN = ((test_pred_list == 0) & (test_target_list == 0)).sum()
        FN = ((test_pred_list == 0) & (test_target_list == 1)).sum()
        FP = ((test_pred_list == 1) & (test_target_list == 0)).sum()
        if TP + FP==0:
            precision=1.0
        else:
            precision = float(TP) / float(TP + FP)
        if TP+FN==0:
            recall=1.0
        else:
            recall = float(TP) / float(TP + FN)
        if recall+precision==0.0:
            F1=0.0
        else:
            F1 = 2 * recall * precision / (recall + precision)
        print("Test set: ")
        print('TP=',TP,'TN=',TN,'FN=',FN,'FP=',FP)
        print('TP+FP',TP+FP)
        print('precision',precision)
        print('recall',recall)
        acc = float(TP + TN) / float(TP + TN + FP + FN)
        
        TPR=recall
        TNR=float(TN)/float(TN+FP)
        print('F1',F1)
        print('acc',acc)
        AUC = roc_auc_score(test_target_list, test_score_list)
        print('AUC', AUC)
        print('\n')
        
        f = open(f'{save_folder}/20220728_Co-training_Prostate_Partly_Labeled_group8_test01_load_model_{my_modelname}.txt', 'a+')
        f.write('\n Test: average accuracy: {:.4f}, TP= {}, TN= {}, FN= {}, FP= {}, F1= {:.4f}, TPR= {:.4f}, TNR= {:.4f}, AUC= {:.4f}\n'.format(test_acc,
        TP,TN,FN,FP,F1,TPR,TNR,AUC))
        f.close()

    if get_best_model==1:
        torch.save(my_best_model.state_dict(), save_folder+"/20220728_Co-training_Prostate_Partly_Labeled_group8_best_{}_{}_val_acc{}_test_acc{}.pt".format(my_modelname,best_epoch,best_acc, best_epoch_test_acc))

