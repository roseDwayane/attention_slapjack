import os
import torch

import attention

torch.cuda.empty_cache()
import pickle
#import EEGNet
#import Models.tiny_model as tiny_model
#import cumbersome_model
#import UNet_family
#from Criteria import CrossEntropyLoss2d, FocalLoss
import torch.nn as nn
import torch.backends.cudnn as cudnn
from myDataset import myDataset
import time
import torch.optim.lr_scheduler
import numpy as np
from scipy import signal
from sklearn.metrics import confusion_matrix
from utils import draw_cm, imgSave, numpy_SNR, draw_psd, SNR_cal
#from dataGenerator import dataInit, dataDelete, dataRestore
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def SNR(args, val_loader, model, epoch):
    model.eval()
    for i ,(input, target, max_num) in enumerate(val_loader):
        if args.onGPU == True:
            input = input.cuda()
            target = target.cuda()
        with torch.no_grad():
            # run the mdoel
            decode = model(input)

        i_t, i_d = SNR_cal(input, target, decode, max_num)
        tmean = np.nanmean(i_t)
        tstd = np.nanstd(i_t)
        dmean = np.nanmean(i_d)
        dstd = np.nanstd(i_d)

        print("SNR(shap): ", i_d.shape)
        #print(np.nanmean(i_t))
        print(tmean, tstd, dmean, dstd)
        break
    return tmean, tstd, dmean, dstd


def draw(args, mode, result, epoch):
    '''
    :param args: general arguments
    :param val_loader: loaded for validation dataset
    :param model: model
    :return: non

    Channel_location = [    "FP1", "FP2",
                    "F7", "F3", "FZ", "F4", "F8",
                 "FT7", "FC3", "FCZ", "FC4", "FT8",
                    "T4", "C3", "FZ", "C4", "T4",
                 "TP7", "CP3", "CPZ", "CP4", "TP8",
                    "T5", "P3", "PZ", "P4", "T6",
                          "O1", "OZ", "O2"]
    '''
    Channel_location = ["A_Fp1", "Fz", "F3", "F7", "FT9", "FC5", "FC1", "C3",
                        "T7", "TP9", "CP5", "CP1", "PZ", "P3", "P7", "O1",
                        "OZ", "O2", "P4", "P8", "TP10", "CP6", "CP2", "CZ",
                        "C4", "T8", "FT10", "FC6", "FC2", "F4", "F8", "Fp2",
                        "B_Fp1", "Fz", "F3", "F7", "FT9", "FC5", "FC1", "C3",
                        "T7", "TP9", "CP5", "CP1", "PZ", "P3", "P7", "O1",
                        "OZ", "O2", "P4", "P8", "TP10", "CP6", "CP2", "CZ",
                        "C4", "T8", "FT10", "FC6", "FC2", "F4", "F8", "Fp2"]

    print("draw: ", result.shape)
    draw_cm(result, Channel_location)
    imgSave(args.savefig+str(mode)+'/', 'e'+str(epoch))
    #print("draw: ", cm)

def val(args, val_loader, model, criterion):
    '''
    :param args: general arguments
    :param val_loader: loaded for validation dataset
    :param model: model
    :param criterion: loss function
    :return: average epoch loss
    '''
    #switch to evaluation mode
    model.eval()

    epoch_loss = []
    result = []
    total_batches = len(val_loader)
    for i, (data, label) in enumerate(val_loader):
        start_time = time.time()
        if args.onGPU:
            data = data.cuda()
            label = label.cuda()

        output, matrix_A = model(data)


        # set the grad to zero
        # loss = criterion(output, label)
        # loss = criterion(output, torch.max(label, 1)[1])
        loss = criterion(output, data)
        loss.backward()

        epoch_loss.append(loss.item())
        time_taken = time.time() - start_time

        new_result = torch.mean(matrix_A, 0).cpu().detach().numpy()
        if i == 0:
            epoch_matrix_A = new_result
        else:
            epoch_matrix_A = np.concatenate((epoch_matrix_A, new_result), axis=0)
        # epoch_matrix_A = torch.cat((epoch_matrix_A, torch.mean(matrix_A, 0)), 0)
        # epoch_matrix_A.append(torch.mean(matrix_A, 0))
        time_taken = time.time() - start_time

        print('[%3d/%3d] total_loss: %.8f time:%.8f' % (i, total_batches, loss.item(), time_taken))

    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
    average_epoch_matrix_A = np.mean(epoch_matrix_A, axis=0)

    return average_epoch_loss_train, average_epoch_matrix_A

def train(args, train_loader, model, criterion, optimizer, epoch):
    '''
    :param args: general arguments
    :param train_loader: loaded for training dataset
    :param model: model
    :param criterion: loss function
    :param optimizer: optimization algo, such as ADAM or SGD
    :param epoch: epoch number
    :return: average epoch loss, overall pixel-wise accuracy, per class accuracy, per class iu, and mIOU
    '''
    # switch to train mode
    model.train()

    epoch_loss = []
    epoch_matrix_A = []

    total_batches = len(train_loader)
    print("train:", train_loader)
    for i, (data, label) in enumerate(train_loader):
        start_time = time.time()
        if args.onGPU:
            data = data.cuda()
            label = label.cuda()

        #output = model(data)
        output, matrix_A = model(data)

        #set the grad to zero
        optimizer.zero_grad()
        #loss = criterion(output, torch.max(label, 1)[1])
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())

        new_result = torch.mean(matrix_A, 0).cpu().detach().numpy()
        if i == 0:
            epoch_matrix_A = new_result
        else:
            epoch_matrix_A = np.concatenate((epoch_matrix_A, new_result), axis=0)
        #epoch_matrix_A = torch.cat((epoch_matrix_A, torch.mean(matrix_A, 0)), 0)
        #epoch_matrix_A.append(torch.mean(matrix_A, 0))
        time_taken = time.time() - start_time

        print('[%3d/%3d] total_loss: %.8f time:%.8f' % (i, total_batches, loss.item(), time_taken))

    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
    average_epoch_matrix_A = np.mean(epoch_matrix_A, axis=0)

    return average_epoch_loss_train, average_epoch_matrix_A

def save_checkpoint(state, is_best, save_path):
    """
    Save model checkpoint.
    :param state: model state
    :param is_best: is this checkpoint the best so far?
    :param save_path: the path for saving
    """
    filename = 'checkpoint.pth.tar'
    torch.save(state, os.path.join(save_path, filename))
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, os.path.join(save_path, 'BEST_' + filename))

def netParams(model):
    '''
    helper function to see total network parameters
    :param model: model
    :return: total network parameters
    '''
    total_paramters = 0
    for parameter in model.parameters():
        i = len(parameter.size())
        p = 1
        for j in range(i):
            p *= parameter.size(j)
        total_paramters += p

    return total_paramters

def trainValidateSegmentation(args):
    '''
    Main function for trainign and validation
    :param args: global arguments
    :return: None
    '''
    # check if processed data file exists or not
    if not os.path.isfile(args.loadpickle):
        print('Error while pickling data. Please checking data processing firstly.')
    #        exit(-1)
    else:
        data = pickle.load(open(args.loadpickle, "rb"))

    # load the model

    if args.model == 'tiny_model':
        # model = tiny_model.TinyModel(args.classes, p=6, q=10, Pretrain=args.pretrained)
        pass
    elif args.model == 'cumbersome_model':
        model = cumbersome_model.UNet(n_channels=64, n_classes=len(args.data_class), bilinear=True)
        #model = UNet_family.NestedUNet(num_classes=30, input_channels=30, bilinear=True)
    elif args.model == 'EEGNet':
        #model = EEGNet.Net(n_classes=len(args.data_class))
        model = EEGNet.EEGNet(n_classes=len(args.data_class))
    elif args.model == 'attention':
        model = attention.MyModel(input_channel=64)

    args.savedir = args.savedir + '/'
    total_paramters = netParams(model)
    print('Total network parameters: ' + str(total_paramters))

    if args.onGPU:
        # model = model.to(device)
        model = model.cuda()

    # create the directory if not exist
    if not os.path.exists(args.save):
        os.mkdir(args.save)

    if not os.path.exists(args.savedir):
        os.mkdir(args.savedir)

    # define optimization criteria
    # weight = torch.from_numpy(data['classWeights']) # convert the numpy array to torch
    # weight = torch.FloatTensor([0.500001, 0.5000001]) # convert the numpy array to torch
    # if args.onGPU:
    #     weight = weight.cuda()

    criteria = nn.MSELoss()
    #criteria = nn.CrossEntropyLoss()
    # criteria = CrossEntropyLoss2d(weight) #weight
    # criteria = FocalLoss(3, weight)

    if args.onGPU:
        criteria = criteria.cuda()

    # since we training from scratch, we create data loaders at different scales
    # so that we can generate more augmented data and prevent the network from overfitting

    #trainLoader = torch.utils.data.DataLoader(
    #    myDataLoader.MyDataset(data['trainIm'], data['trainAnnot'], (args.inWidth, args.inHeight), flag_aug=0),
    #    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    # ****#
    # if args.onGPU:
    #     cudnn.benchmark = True
    torch.backends.cudnn.enabled = False
    # ****#

    start_epoch = 0

    if args.resume:  # 當機回復
        if os.path.isfile(args.resumeLoc):
            print("=> loading checkpoint '{}'".format(args.resume))
            #checkpoint = torch.load(args.resumeLoc, map_location='cpu')
            checkpoint = torch.load(args.resumeLoc, map_location='cpu')
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    logFileLoc = args.savedir + args.logFile
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write("Parameters: %s" % (str(total_paramters)))
        logger.write("\n%s\t%s\t%s\t%s\t%s\t" % ('Epoch', 'Loss(Tr)', 'Loss(Ts)', 'Loss(val)', 'Learning_rate'))
    logger.flush()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # optimizer = torch.optim.Adam(model.parameters(), args.lr, (0.9, 0.999), eps=1e-08, weight_decay=5e-4)
    # we step the loss by 2 after step size is reached
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.milestones, gamma=0.1, last_epoch=-1)

    best_loss = 100

    for my_iter in range(1):

        trainset = myDataset(mode=0, data_class=args.data_class)  # file='./EEG_EC_Data_csv/train.txt'
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

        testset = myDataset(mode=1, data_class=args.data_class)  # file='./EEG_EC_Data_csv/train.txt'
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        # valLoader = torch.utils.data.DataLoader(
        #    myDataLoader.MyDataset(data['valIm'], data['valAnnot'], (args.inWidth, args.inHeight), flag_aug=0),
        #    batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        valset = myDataset(mode=2, data_class=args.data_class)
        val_loader = torch.utils.data.DataLoader(
            valset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)


        for epoch in range(start_epoch, args.max_epochs):
            start_time = time.time()
            scheduler.step(epoch)
            lr = 0
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
            print("Learning rate: " + str(lr))

            # train for one epoch
            # We consider 1 epoch with all the training data (at different scales)

            #time.sleep(30)

            lossTr, Tr_result = train(args, train_loader, model, criteria, optimizer, epoch)
            print("Tr_result: ", Tr_result.shape)
            draw(args, "train", Tr_result, epoch)
            lossTs, Ts_result = val(args, test_loader, model, criteria)
            draw(args, "test", Ts_result, epoch)
            lossVal, val_result = val(args, val_loader, model, criteria)
            #time.sleep(1)

            #draw(args, test_loader, model, epoch)
            #draw_sub(args, train_loader, model, start_epoch)
            #draw(args, val_loader, model, epoch)
            #tmean, tstd, dmean, dstd = SNR(args, train_loader, model, epoch)
            #print(tmean, tstd, dmean, dstd)

            # Did validation loss improve?
            is_best = lossTs < best_loss
            best_loss = min(lossTs, best_loss)

            if not is_best:
                epochs_since_improvement += 1
                print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
            else:
                epochs_since_improvement = 0

            # Save checkpoint

            state = {'epoch': epoch + 1,
                     'arch': str(model),
                     'epochs_since_improvement': epochs_since_improvement,
                     'best_loss': best_loss,
                     'state_dict': model.state_dict(),
                     'lossTr': lossTr,
                     'lossTs': lossTs,
                     'lossVal': lossVal,
                     ' lr': lr}
            save_checkpoint(state, is_best, args.savedir)

            logger.write("\n%d\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f" % (epoch, lossTr, lossTs, lossVal, lr))
            logger.flush()
            print("Epoch : " + str(epoch) + ' Details')
            print("\nEpoch No.: %d/%d\tTrain Loss = %.8f\tVal Loss = %.8f" % (
            epoch, args.max_epochs, lossTr, lossVal))
            time_taken = time.time() - start_time
            print("Time: ", time_taken)

        '''
                if os.path.isfile(args.resumeLoc):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resumeLoc, map_location='cpu')
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            #draw_sub(args, train_loader, model, start_epoch)
        '''

    logger.close()

class model_train_parameter():
    def __init__(self, data_class, save, data=0):
        self.model = "attention"  # cumbersome_model, EEGNet, attention
        self.max_epochs = 60
        self.num_workers = 0
        self.batch_size = 128
        self.sample_rate = 256
        self.step_loss = 100  # Decrease learning rate after how many epochs.
        self.milestones = [50, 100, 125, 140]
        self.loss = data_class
        self.lr = 0.001  # 'Initial learning rate'
        self.save = save
        self.savedata = data
        self.savedir = self.save + '/modelsave'  # directory to save the results
        self.savefig = self.save + '/pic_'
        self.resume = True  # Use this flag to load last checkpoint for training
        self.resumeLoc = self.save + '/modelsave/checkpoint.pth.tar'
        self.data_class = data_class
        self.logFile = 'model_trainValLog.txt'  # File that stores the training and validation logs
        self.onGPU = True  # Run on CPU or GPU. If TRUE, then GPU.
        self.pretrained = ''  # Pretrained model
        self.loadpickle = './'



def main_train():
    for i in range(1):
        i = 1
        date = "1121_1"
        label = ["singleplayer", "singlebystander", "coop", "comp"]
        #dataRestore(name)
        #trainValidateSegmentation(args=model_train_parameter(["singleplayer", "singlebystander", "coop", "comp"], './' + date + '_sp_sb_co_cm'))
        trainValidateSegmentation(args=model_train_parameter([label[0], label[1]], './' + date + '_sp_sb'))
        trainValidateSegmentation(args=model_train_parameter([label[0], label[2]], './' + date + '_sp_co'))
        trainValidateSegmentation(args=model_train_parameter([label[0], label[3]], './' + date + '_sp_cm'))
        trainValidateSegmentation(args=model_train_parameter([label[1], label[2]], './' + date + '_sb_co'))
        trainValidateSegmentation(args=model_train_parameter([label[1], label[3]], './' + date + '_sb_cm'))
        trainValidateSegmentation(args=model_train_parameter([label[2], label[3]], './' + date + '_co_cm'))
        #dataDelete("./" + name + "_simulate_data/")
if __name__ == '__main__':
    main_train()