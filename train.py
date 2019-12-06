from utils.datasets import LoadImagesAndLabels
from models import MobileFaceNet
import tqdm 
import argparse
import os
from torchvision import models
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm


hyp ={'lr0': 0.0015,  
      'momentum': 0.95,
      'weight_decay': 0.00045, 
      'loss_hyp':{
          'cut_point':(0,200,202,714),
          'w':(1,0.1,0.1)
        }
      }


criterion = nn.MSELoss()
def compute_loss(preds, labels):
    total_loss_list = [w * criterion(preds[i:j],labels[i:j]) 
        for w,i,j in zip(hyp['loss_hyp']['w'], hyp['loss_hyp']['cut_point'][:-1], hyp['loss_hyp']['cut_point'][1:])]
    total_loss = total_loss_list[0] #+ total_loss_list[1] + total_loss_list[2]  #TODO
    return total_loss

# criterion = nn.MSELoss(reduce=False, size_average=False)
# criterion = nn.CrossEntropyLoss()

def train():
    batch_size = args.batch_size
    epochs = args.epochs
    train_path = args.train_path
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(train_path)
    print(device)
    print('epochs',epochs)
    print('batch size', batch_size)
    start_epoch = 0

    ##############################################################################################################
    # student model
    model = MobileFaceNet(714).to(device)
    resume = True
    if resume:
        chkpt = torch.load(args.model_path)
        model.load_state_dict(chkpt)
        # model = model.to(device)
    transfer = False
    if transfer:
        for i, param in enumerate(model.parameters()):
            if i < 24:
                param.requires_grad = False

    ##############################################################################################################
    # pg0, pg1 = [], []  # optimizer parameter groups
    # for k, v in dict(model.named_parameters()).items():
    #     if 'Conv2d.weight' in k:
    #         pg1 += [v]  # parameter group 1 (apply weight_decay)
    #     else:
    #         pg0 += [v]  # parameter group 0
    # optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    # optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    # del pg0, pg1
    # optimizer = optim.SGD(model.parameters(), lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    optimizer = optim.Adam(model.parameters(), lr=hyp['lr0'] ,weight_decay=hyp['weight_decay'])
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[round(args.epochs * x) for x in [0.8, 0.9]], gamma=0.1)
    scheduler.last_epoch = start_epoch - 1

    # Optimizer
    for k, v in dict(model.named_parameters()).items():
        # print(k,v.shape)
        pass
    dataset = LoadImagesAndLabels(train_path,
                                  img_size=112,
                                  batch_size=16,
                                  augment=True,
                                  hyp=hyp,  # augmentation hyperparameters
                                  rect=False,  # rectangular training
                                  image_weights=False,
                                  cache_labels= False,
                                  cache_images= False)
    # num_workers = 0
    # https://discuss.pytorch.org/t/eoferror-ran-out-of-input-when-enumerating-the-train-loader/22692/7
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=min([os.cpu_count(), batch_size, 4]),
                                             shuffle=True,  # Shuffle=True unless rectangular training is used
                                             pin_memory=True, 
                                             collate_fn=dataset.collate_fn)  
    # model.nc = nc 
    model.hyp = hyp
    nb = len(dataloader)
    
    running_loss = [0.0]
    print('Starting %s for %g epochs...' % ('training', epochs))
    for epoch in range(start_epoch, epochs):
        model.train()
        # pbar = tqdm(dataloader, total=nb)
        for i,(imgs, labels) in enumerate(dataloader):
            if imgs is None or labels is None:
                continue

            imgs = imgs.to(device)
            labels = labels.to(device)
            preds = model(imgs)
            optimizer.zero_grad()
            # _, preds = torch.max(outputs, 1)
            loss = compute_loss(preds, labels)
            loss.backward()
            optimizer.step()
            
            running_loss.append(running_loss[-1] + loss.item() * imgs.size(0))
            if i % 600 == 0:
                # with open(r'weights/gg.log', 'w+') as f:
                #     f.writelines([str(loss.item()),'\n'])
                print(loss.item())

        scheduler.step()

        if tb_writer:
            titles = ['running_loss']
            for xi, title in zip(running_loss, titles):
                tb_writer.add_scalar(title, xi, epoch)
        # save model
        if True:  # epoch % 1 == 0:
            # with open(r'weights/gg.log', 'w+') as f:
            #     loss_toines(loss_to_log)
            #     f.write('_log = [str(cum_loss_i) for cum_loss_i in running_loss]
            #     f.writel\n')
            print(os.path.splitext(args.model_path)[0] + str(epoch) + '.pth')
            torch.save(model.state_dict(), os.path.splitext(args.model_path)[0] + str(epoch) + '.pth')
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128) 
    parser.add_argument('--train-path', type=str, default=r'./data/glint_train.txt')
    # parser.add_argument('--train-path', type=str, default=r'./data/ma_train.txt')

    parser.add_argument('--model-path', type=str, default=r'./weights/mobile_facestudent.pth')
    args = parser.parse_args()
    tb_writer = None
    try:
        # Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter()
        print('summury writer has launched')
    except:
        pass
    train()
    