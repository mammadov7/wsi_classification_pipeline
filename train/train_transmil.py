import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, copy, itertools, glob, datetime
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support
from sklearn.datasets import load_svmlight_file
from collections import OrderedDict
from pathlib import Path
from transmil import TransMIL

from Opt.lookahead import Lookahead
from Opt.radam import RAdam


def get_bag_feats(csv_file_df, args):
    if args.dataset == 'TCGA-lung-default':
        feats_csv_path = 'datasets/tcga-dataset/tcga_lung_data_feats/' + csv_file_df.iloc[0].split('/')[1] + '.csv'
    else:
        feats_csv_path = csv_file_df.iloc[0]
    df = pd.read_csv(feats_csv_path)
    # df = pd.read_csv(feats_csv_path,index_col=0)
    # if df.shape[1]%384!=0:
    #     df = pd.read_csv(feats_csv_path)
    #     print(feats_csv_path)
    feats = shuffle(df).reset_index(drop=True)
    feats = feats.to_numpy()
    label = np.zeros(args.num_classes)
    if args.num_classes==1:
        label[0] = csv_file_df.iloc[1]
    else:
        if int(csv_file_df.iloc[1])<=(len(label)-1):
            label[int(csv_file_df.iloc[1])] = 1
        
    return label, feats

def train(train_df, milnet, criterion, optimizer, args):
    milnet.train()
    csvs = shuffle(train_df).reset_index(drop=True)
    total_loss = 0
    bc = 0
    Tensor = torch.cuda.FloatTensor
    for i in range(len(train_df)):
        optimizer.zero_grad()
        label, feats = get_bag_feats(train_df.iloc[i], args)
        feats = dropout_patches(feats, args.dropout_patch)
        bag_label = Variable(Tensor(np.array([label])))
        bag_feats = Variable(Tensor(np.array([feats])))
        # bag_feats = bag_feats.view(-1, args.feats_size)
        bag_prediction = milnet(bag_feats)
        loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item()
        # sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f' % (i, len(train_df), loss.item()))
    return total_loss / len(train_df)

def dropout_patches(feats, p):
    p = 0 if len(feats) < 1000 else p
    idx = np.random.choice(np.arange(feats.shape[0]), int(feats.shape[0]*(1-p)), replace=False)
    sampled_feats = np.take(feats, idx, axis=0)
    pad_idx = np.random.choice(np.arange(sampled_feats.shape[0]), int(feats.shape[0]*p), replace=False)
    pad_feats = np.take(sampled_feats, pad_idx, axis=0)
    sampled_feats = np.concatenate((sampled_feats, pad_feats), axis=0)
    return sampled_feats

def test(test_df, milnet, criterion, optimizer, args):
    milnet.eval()
    csvs = shuffle(test_df).reset_index(drop=True)
    total_loss = 0
    test_labels = []
    test_predictions = []
    Tensor = torch.cuda.FloatTensor
    with torch.no_grad():
        for i in range(len(test_df)):
            label, feats = get_bag_feats(test_df.iloc[i], args)
            bag_label = Variable(Tensor(np.array([label])))
            bag_feats = Variable(Tensor(np.array([feats])))
            # bag_feats = bag_feats.view(-1, args.feats_size)
            bag_prediction = milnet(bag_feats)
            bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
            total_loss = total_loss + bag_loss.item()
            # sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f' % (i, len(test_df), loss.item()))
            test_labels.extend([label])
            if args.num_classes > 1:
                test_predictions.extend([torch.softmax(bag_prediction.squeeze(),dim=0).squeeze().cpu().numpy()])
            else :
                test_predictions.extend([torch.sigmoid(bag_prediction).squeeze().cpu().numpy()])
    test_labels = np.array(test_labels)
    test_predictions = np.array(test_predictions)
    auc_value, _, thresholds_optimal = multi_label_roc(test_labels, test_predictions, args.num_classes, pos_label=1)
    if args.num_classes==1:
        class_prediction_bag = copy.deepcopy(test_predictions)
        class_prediction_bag[test_predictions>=thresholds_optimal[0]] = 1
        class_prediction_bag[test_predictions<thresholds_optimal[0]] = 0
        test_predictions = class_prediction_bag
        test_labels = np.squeeze(test_labels)
    else:        
        for i in range(args.num_classes):
            class_prediction_bag = copy.deepcopy(test_predictions[:, i])
            class_prediction_bag[test_predictions[:, i]>=thresholds_optimal[i]] = 1
            class_prediction_bag[test_predictions[:, i]<thresholds_optimal[i]] = 0
            test_predictions[:, i] = class_prediction_bag
    bag_score = 0
    for i in range(0, len(test_df)):
        bag_score = np.array_equal(test_labels[i], test_predictions[i]) + bag_score         
    avg_score = bag_score / len(test_df)
    
    return total_loss / len(test_df), avg_score, auc_value, thresholds_optimal


def real_test(test_df, milnet, criterion, optimizer, args, thresholds_optimal):
    milnet.eval()
    csvs = shuffle(test_df).reset_index(drop=True)
    total_loss = 0
    test_labels = []
    test_predictions = []
    Tensor = torch.cuda.FloatTensor
    with torch.no_grad():
        for i in range(len(test_df)):
            label, feats = get_bag_feats(test_df.iloc[i], args)
            bag_label = Variable(Tensor(np.array([label])))
            bag_feats = Variable(Tensor(np.array([feats])))
            # bag_feats = bag_feats.view(-1, args.feats_size)

            bag_prediction = milnet(bag_feats)
            bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
            total_loss = total_loss + bag_loss.item()
            # sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f' % (i, len(test_df), loss.item()))
            test_labels.extend([label])
            if args.num_classes > 1:
                test_predictions.extend([torch.softmax((bag_prediction).squeeze(),dim=0).squeeze().cpu().numpy()])
            else :
                test_predictions.extend([torch.sigmoid(bag_prediction).squeeze().cpu().numpy()])
    test_labels = np.array(test_labels)
    test_predictions = np.array(test_predictions)
    auc_value, _, _ = multi_label_roc(test_labels, test_predictions, args.num_classes, pos_label=1)
    if args.num_classes==1:
        class_prediction_bag = copy.deepcopy(test_predictions)
        class_prediction_bag[test_predictions>=thresholds_optimal[0]] = 1
        class_prediction_bag[test_predictions<thresholds_optimal[0]] = 0
        test_predictions = class_prediction_bag
        test_labels = np.squeeze(test_labels)
    else:        
        for i in range(args.num_classes):
            class_prediction_bag = copy.deepcopy(test_predictions[:, i])
            class_prediction_bag[test_predictions[:, i]>=thresholds_optimal[i]] = 1
            class_prediction_bag[test_predictions[:, i]<thresholds_optimal[i]] = 0
            test_predictions[:, i] = class_prediction_bag
    bag_score = 0
    for i in range(0, len(test_df)):
        bag_score = np.array_equal(test_labels[i], test_predictions[i]) + bag_score         
    avg_score = bag_score / len(test_df)
    
    return total_loss / len(test_df), avg_score, auc_value

def multi_label_roc(labels, predictions, num_classes, pos_label=1):
    fprs = []
    tprs = []
    thresholds = []
    thresholds_optimal = []
    aucs = []
    if len(predictions.shape)==1:
        predictions = predictions[:, None]
    for c in range(0, num_classes):
        label = labels[:, c]
        prediction = predictions[:, c]
        fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
        c_auc = roc_auc_score(label, prediction)
        aucs.append(c_auc)
        thresholds.append(threshold)
        thresholds_optimal.append(threshold_optimal)
    return aucs, thresholds, thresholds_optimal

def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def main():
    parser = argparse.ArgumentParser(description='Train DSMIL on 20x patch features learned by SimCLR')
    parser.add_argument('--num_classes', default=1, type=int, help='Number of output classes [2]')
    parser.add_argument('--feats_size', default=384, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--lr', default=0.0002, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--num_epochs', default=200, type=int, help='Number of total training epochs [40|200]')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--weight_decay', default=5e-3, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--dataset', default='TCGA-lung-default', type=str, help='Dataset folder name')
    parser.add_argument('--split', default=0.2, type=float, help='Training/Validation split [0.2]')
    parser.add_argument('--model', default='transmil', type=str, help='MIL model [dsmil]')
    parser.add_argument('--dropout_patch', default=0, type=float, help='Patch dropout rate [0]')
    parser.add_argument('--dropout_node', default=0, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--non_linearity', default=1, type=float, help='Additional nonlinear operation [0]')
    parser.add_argument('--q_n', default=128, type=int, help='Number of hiddent parameters')
    parser.add_argument('--average', type=bool, default=True, help='Average the score of max-pooling and bag aggregating')
    args = parser.parse_args()
    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpu_ids)
    
    
    milnet = TransMIL(input_size=args.feats_size, n_classes=args.num_classes).cuda()

    if args.num_classes > 1:
        if args.dataset[:5] == 'bracs':
            criterion = nn.CrossEntropyLoss(weight=torch.tensor(np.array([0.64860427, 2.53205128, 0.94047619]),dtype=torch.float).cuda())
        else:
            criterion = nn.CrossEntropyLoss()
    else :
        criterion = nn.BCEWithLogitsLoss()

    print(args.dataset,args.q_n,args.non_linearity)
    print(args.lr)
    print(args.weight_decay)

    
    optimizer = Lookahead(RAdam([ {'params':milnet.parameters(), ' weight_decay':args.weight_decay}],lr=0.0002, weight_decay=args.lr))

    # optimizer = torch.optim.Adam(milnet.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, 0.000005)
    
    train_csv, val_csv = [], []
    if args.dataset == 'TCGA-lung-default':
        bags_csv = 'datasets/tcga-dataset/TCGA.csv'
    else:
        bags_csv = os.path.join('datasets', args.dataset, args.dataset+'.csv')
        train_csv = os.path.join('datasets', args.dataset, 'train.csv')
        val_csv = os.path.join('datasets', args.dataset, 'val.csv')
    
    bags_path = pd.read_csv(bags_csv)
    train_csv = pd.read_csv(train_csv)
    val_csv = pd.read_csv(val_csv)

    train_csv = [ Path(x).name for x in train_csv['path'].tolist() ]
    val_csv = [ Path(x).name for x in val_csv['path'].tolist() ]
    
    train_path = bags_path[bags_path.apply(lambda x: Path(x['path']).name in train_csv, axis=1)]
    val_path = bags_path[bags_path.apply(lambda x: Path(x['path']).name in val_csv, axis=1)]
    try:
        test_path = pd.read_csv(os.path.join('datasets', args.dataset, 'test.csv'))
    except:
        test_path=[]

    best_score = 0

    save_path = os.path.join('weights/'+args.model, datetime.datetime.now().strftime("%Y%m%d%H%M%S%f"))
    print(save_path)
    os.makedirs(save_path, exist_ok=True)
    run = len(glob.glob(os.path.join(save_path, '*.pth')))
    best_mil=[]
    patience = 0
    thresholds_optimal=0.0
    for epoch in range(1, args.num_epochs):
        patience+=1
        train_path = shuffle(train_path).reset_index(drop=True)
        val_path = shuffle(val_path).reset_index(drop=True)
        train_loss_bag = train(train_path, milnet, criterion, optimizer, args) # iterate all bags
        test_loss_bag, avg_score, aucs, thresholds_optimal = test(val_path, milnet, criterion, optimizer, args)
        if args.dataset=='TCGA-lung':
            print('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, auc_LUAD: %.4f, auc_LUSC: %.4f' % 
                  (epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score, aucs[0], aucs[1]))
        else:
            print('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, AUC: ' % 
                  (epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score) + '|'.join('class-{}>>{}'.format(*k) for k in enumerate(aucs))) 
        scheduler.step()
        # current_score = test_loss_bag
        current_score = sum(aucs)
        if current_score > best_score:
            
            best_score = current_score
            save_name = os.path.join(save_path, str(run+1)+'.pth')
            torch.save(milnet.state_dict(), save_name)
            if args.dataset=='TCGA-lung':
                print('Best model saved at: ' + save_name + ' Best thresholds: LUAD %.4f, LUSC %.4f' % (thresholds_optimal[0], thresholds_optimal[1]))
            else:
                print('Best model saved at: ' + save_name)
                print('Best thresholds ===>>> '+ '|'.join('class-{}>>{}'.format(*k) for k in enumerate(thresholds_optimal)))
    
            if len(test_path) == 0:
                continue
            print('\nBest Test Set with val threshold')
            test_loss_bag, avg_score, aucs = real_test(test_path, milnet, criterion, optimizer, args, thresholds_optimal)
            print('Test loss: %.4f, Test average score: %.4f, Test AUC: ' % (test_loss_bag, avg_score) + '|'.join('class-{}>>{}'.format(*k) for k in enumerate(aucs))) 
            patience=0
            
            # print('\nBest Test Set with test threshold')
            # test_loss_bag, avg_score, aucs, thresholds_optimal = test(test_path, milnet, criterion, optimizer, args)
            # print('Test loss: %.4f, Test average score: %.4f, Test AUC: ' % (test_loss_bag, avg_score) + '|'.join('class-{}>>{}'.format(*k) for k in enumerate(aucs))) 
        # if patience == 20:
        #     break
    test_loss_bag, avg_score, aucs = real_test(test_path, milnet, criterion, optimizer, args, thresholds_optimal)
    print('Last model Test')
    print('Test loss: %.4f, Test average score: %.4f, Test AUC: ' % (test_loss_bag, avg_score) + '|'.join('class-{}>>{}'.format(*k) for k in enumerate(aucs))) 
    save_name = os.path.join(save_path, 'last.pth')
    torch.save(milnet.state_dict(), save_name)
if __name__ == '__main__':
    main()