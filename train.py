import os
import sys
import random
import warnings
from argparse import ArgumentParser

import numpy as np
import torch
from torch.optim import Adam
import torch.nn.functional as F
from losses import ncc_loss,gradient_loss,entropy_loss
import logging
from network import Regnet, Augnet, SpatialTransformer
from datagenerators import MRIDataset


def train(args):

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)
    logging.basicConfig(filename=os.path.join(args.model_dir,'log.log'), level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    dataset = MRIDataset(args.datapath,args.shape,label_mapping=args.label_mapping,batch_size=1,dataname='CANDI')
    (vol_atlas, seg_atlas, ids_atlas), \
    (vol_train, _, ids_train), \
    (vol_test, seg_test, ids_test) = dataset.load_dataset()

    train_gen = dataset.gen_register_batch()

    
    model = Augnet(args.shape)
    trf = SpatialTransformer(args.shape,'nearest')
    trf2 = SpatialTransformer(args.shape)
    model.train()
    model.cuda()
    opt = Adam(model.parameters(),lr=0.0001)

    for i in range(0,args.n_iter):

        # Save model checkpoint
        if i % args.n_save_iter == 0:
            save_file_name = os.path.join(args.model_dir, '%d.pth' % i)
            torch.save(model.state_dict(), save_file_name)

        # Generate the moving images and convert them to tensors.
        atlas_batch, atlas_seg_batch, train_batch = next(train_gen)
        atlas_batch = torch.from_numpy(atlas_batch).cuda()
        train_batch = torch.from_numpy(train_batch).cuda()
        atlas_seg_batch = torch.from_numpy(atlas_seg_batch).cuda()

        atlas_feats, train_feats = model.encoder(atlas_batch), model.encoder(train_batch)
        flow = model.reg(atlas_feats,train_feats)
        train_out = model.decoder(train_feats)
        warp_batch = model.reg.spa1(atlas_batch,flow)


        aug_flow = flow.detach()
        warp_batch2 = trf2(atlas_batch,aug_flow).detach()
        warp_seg = trf(atlas_seg_batch.float(),aug_flow)
        warp_f = trf2(atlas_feats[0].detach(),aug_flow)
        sim = F.cosine_similarity(warp_f,train_feats[0].detach(),dim=1)
        mask = (sim>0.9).float()

    
        out = model.decoder(model.encoder(warp_batch2))
        seg_loss = F.cross_entropy(out,warp_seg.long().squeeze(1))
        seg_loss2 = F.cross_entropy(train_out,warp_seg.long().squeeze(1),reduction='none')
        seg_loss2 = (seg_loss2*mask).sum()/(mask.sum()+1e-6)

        sim_loss = ncc_loss(warp_batch,train_batch)
        grad_loss = gradient_loss(flow)
        loss = sim_loss + grad_loss + seg_loss + 0.1*seg_loss2

        logging.info("%d,%f,%f,%f,%f,%f" % (i, loss.item(),sim_loss.item(),grad_loss.item(),seg_loss.item(),seg_loss2.item()))

        opt.zero_grad()
        loss.backward()
        opt.step()
    save_file_name = os.path.join(args.model_dir, 'final_model.pth' )
    torch.save(model.state_dict(), save_file_name)


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
                        default=10000,
                        help="number of iterations")
    
    parser.add_argument("--n_save_iter", 
                        type=int,
                        dest="n_save_iter", 
                        default=1000,
                        help="frequency of model saves")

    parser.add_argument("--model_dir", 
                        type=str,
                        dest="model_dir", 
                        default='CANDI_Model',
                        help="models folder")
    
    args = parser.parse_args()
    args.label_mapping = [0, 16, 10, 49, 8, 47, 4, 43, 7, 46, 12, 51, 2, 41, 28, 60, 11, 50, 13, 52, 17, 53, 14, 15, 18, 54, 24, 3, 42]
    args.shape = (160, 160, 128)


    train(args)

