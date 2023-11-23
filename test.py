import os
import sys
import random
import warnings
from argparse import ArgumentParser

import numpy as np
import torch
from torch.optim import Adam
import torch.nn.functional as F
from losses import ncc_loss,gradient_loss
import logging
from network import Regnet,Unet,Augnet,SpatialTransformer
from datagenerators import MRIDataset,get_lab



def dice(vol1, vol2, labels=None, nargout=1):
    '''
    Dice [1] volume overlap metric

    The default is to *not* return a measure for the background layer (label = 0)

    [1] Dice, Lee R. "Measures of the amount of ecologic association between species."
    Ecology 26.3 (1945): 297-302.

    Parameters
    ----------
    vol1 : nd array. The first volume (e.g. predicted volume)
    vol2 : nd array. The second volume (e.g. "true" volume)
    labels : optional vector of labels on which to compute Dice.
        If this is not provided, Dice is computed on all non-background (non-0) labels
    nargout : optional control of output arguments. if 1, output Dice measure(s).
        if 2, output tuple of (Dice, labels)

    Output
    ------
    if nargout == 1 : dice : vector of dice measures for each labels
    if nargout == 2 : (dice, labels) : where labels is a vector of the labels on which
        dice was computed
    '''

    if labels is None:
        labels = np.unique(np.concatenate((vol1, vol2)))
        labels = np.delete(labels, np.where(labels == 0))  # remove background

    dicem = np.zeros(len(labels))
    for idx, lab in enumerate(labels):
        top = 2 * np.sum(np.logical_and(vol1 == lab, vol2 == lab))
        bottom = np.sum(vol1 == lab) + np.sum(vol2 == lab)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon.
        dicem[idx] = top / bottom

    if nargout == 1:
        return dicem
    else:
        return (dicem, labels)

def test(args):

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    dataset = MRIDataset(args.datapath,args.shape,label_mapping=args.label_mapping,batch_size=1,dataname='CANDI')
    (vol_atlas, seg_atlas, ids_atlas), \
    (vol_train, _, ids_train), \
    (vol_test, seg_test, ids_test) = dataset.load_dataset()

   
    
    model = Augnet(args.shape)
    model.load_state_dict(torch.load('./CANDI_Model/10000.pth'))
    model.cuda()
    dice_scores = np.zeros((vol_test.shape[0],28))

    seg_atlas = get_lab(seg_atlas,args.label_mapping)
    vol_atlas, seg_atlas = vol_atlas.transpose(0,4,1,2,3), seg_atlas.transpose(0,4,1,2,3)
    vol_atlas, seg_atlas = torch.from_numpy(vol_atlas).cuda(),torch.from_numpy(seg_atlas).cuda()
    trf = SpatialTransformer(args.shape,mode='nearest')

    for i in range(vol_test.shape[0]):


        vol, seg, id1 = vol_test[i:i+1].transpose(0,4,1,2,3), seg_test[i:i+1].transpose(0,4,1,2,3), ids_test[i]
        vol = torch.from_numpy(vol).cuda()
        with torch.no_grad():
            out = model.decoder(model.encoder(vol))
            pre = out.argmax(1,keepdim=True)
            
        pre = pre.cpu().numpy()
        dic = dice(pre,get_lab(seg,args.label_mapping),labels=list(range(1,29)))
        print(id1,'\n',dic.mean())
        dice_scores[i] = dic
    print('final_result:')
    print(dice_scores.mean(1).mean(),dice_scores.mean(1).std())
    print(dice_scores.mean(1).min(),dice_scores.mean(1).max())
    print(dice_scores.mean(1))
    np.save('res.npy',dice_scores.mean(1))



if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--gpu",
                        type=str,
                        default='0',
                        help="gpu id")
    parser.add_argument("--datapath",
                        type=str,
                        default='./CANDI',
                        help="data folder with training vols")

    parser.add_argument("--n_iter",
                        type=int,
                        dest="n_iter",
                        default=40000,
                        help="number of iterations")
    
    parser.add_argument("--n_save_iter", 
                        type=int,
                        dest="n_save_iter", 
                        default=5000,
                        help="frequency of model saves")

    parser.add_argument("--model_dir", 
                        type=str,
                        dest="model_dir", 
                        default='CANDI_seg',
                        help="models folder")
    
    args = parser.parse_args()
    args.label_mapping = [0, 16, 10, 49, 8, 47, 4, 43, 7, 46, 12, 51, 2, 41, 28, 60, 11, 50, 13, 52, 17, 53, 14, 15, 18, 54, 24, 3, 42]
    args.shape = (160, 160, 128)


    test(args)

