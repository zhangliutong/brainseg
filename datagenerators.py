import glob
import os
import sys
from webbrowser import get
import numpy as np



label_mapping = [0, 16, 10, 49, 8, 47, 4, 43, 7, 46, 12, 51, 2, 41, 28, 60, 11, 50, 13, 52, 17, 53, 14, 15, 18, 54, 24, 3, 42]

def get_lab(labels,label_mapping):
    final_lab = np.zeros(labels.shape,dtype=int)
    for i in range(len(label_mapping)):
        final_lab[labels==label_mapping[i]] = i
    return final_lab

class MRIDataset(object):
    def __init__(self, data_path, img_shape, label_mapping, batch_size, mode=None, dataname=None):
        self.dataname = dataname
        self.data_path = data_path
        self.img_shape = img_shape
        self.label_mapping = label_mapping
        self.batch_size = batch_size

    def load_dataset(self, load_seg=True):
        self.atlas_pathlist = glob.glob(os.path.join(self.data_path, 'theone/vol', '*_procimg.npy'))
        self.train_pathlist = glob.glob(os.path.join(self.data_path, 'train/vol', '*_procimg.npy'))
        self.test_pathlist = glob.glob(os.path.join(self.data_path, 'test/vol', '*_procimg.npy'))

        self.atlas = self.load_vol_and_seg(pathlist = self.atlas_pathlist, load_seg=True)
        self.train = self.load_vol_and_seg(pathlist = self.train_pathlist, load_seg=True)
        self.test = self.load_vol_and_seg(pathlist = self.test_pathlist, load_seg=True)
        print ('atlas name:{}'.format(self.atlas[-1]))
        print ('atlas num:{}, train num:{}, test num:{}'.format(self.atlas[0].shape[0], self.train[0].shape[0], self.test[0].shape[0]))

        if self.dataname == 'ABIDE_benchmark':
            self.val_pathlist = glob.glob(os.path.join(self.data_path, 'val/vol', '*_procimg.npy'))
            self.unseen_pathlist = glob.glob(os.path.join(self.data_path, 'unseen/vol', '*_procimg.npy'))

            self.val = self.load_vol_and_seg(pathlist = self.val_pathlist, load_seg=True)
            self.unseen = self.load_vol_and_seg(pathlist = self.unseen_pathlist, load_seg=True)
            print ('unseen num:{}, val num:{}'.format(self.unseen[0].shape[0], self.val[0].shape[0]))

            return self.atlas, self.train, self.val, self.test, self.unseen

        return self.atlas, self.train, self.test

    def load_vol_and_seg(self, pathlist=None, load_seg=True):
        ####init vol, seg, namelist
        load_num = len(pathlist)
        vols = np.zeros((load_num,) + self.img_shape, dtype=np.float32)
        if load_seg:
            segs = np.zeros((load_num,) + self.img_shape, dtype=int)
        else:
            segs = None
        ids = []

        ####load
        for i in range(load_num):
            vol = np.load(pathlist[i])
            if load_seg:
                ####volname: vol/BPDwoPsy_031_procimg.npy, segname: seg/BPDwoPsy_031_seg.npy
                seg = np.load(pathlist[i].replace('vol', 'seg').replace('procimg', 'seg'))
                ####skull strip following previous works
                vol *= (seg > 0)
                unique_labels = np.unique(seg)
                for j in unique_labels:
                    if j not in self.label_mapping:
                        seg[seg == j] = 0
                segs[i] = seg
            max_val = np.percentile(vol,99.99)
            
            #vol = vol/vol.max()
            #vol = np.clip(vol/max_val,0,1)
            vols[i] = vol
            ids.append(os.path.basename(pathlist[i]))
        
        vols = vols[..., np.newaxis]
        if load_seg:
            segs = segs[..., np.newaxis]

        return vols, segs, ids
    
    def gen_register_batch(self):
        vol_atlas, seg_atlas, _ = self.atlas
        vol_train, _, _ = self.train
        #vol_train_all = np.concatenate([vol_atlas, vol_train], axis=0)
        vol_train_all = vol_train

        train_num = vol_train_all.shape[0]
        print (train_num)
        atlas_batch = np.tile(vol_atlas, (self.batch_size, 1, 1, 1, 1))
        atlas_seg_batch = np.tile(seg_atlas, (self.batch_size, 1, 1, 1, 1))
        atlas_seg_batch = get_lab(atlas_seg_batch,self.label_mapping)
        atlas_batch, atlas_seg_batch = atlas_batch.transpose(0,4,1,2,3), atlas_seg_batch.transpose(0,4,1,2,3)
        while True:
            idx_3d = np.random.choice(train_num, self.batch_size, replace=True)
            train_batch = vol_train_all[idx_3d]
            train_batch = train_batch.transpose(0,4,1,2,3)
            yield atlas_batch, atlas_seg_batch, train_batch 

    
    def gen_seg_batch(self):
        vol_train, seg_train, _ = self.train
        #vol_train_all = np.concatenate([vol_atlas, vol_train], axis=0)
        vol_train_all = vol_train
        seg_train_all = seg_train

        train_num = vol_train_all.shape[0]
        print (train_num)
        
        while True:
            idx_3d = np.random.choice(train_num, self.batch_size, replace=True)
            train_batch = vol_train_all[idx_3d]
            train_seg_batch = seg_train_all[idx_3d]
            train_seg_batch = get_lab(train_seg_batch,self.label_mapping)

            train_batch, train_seg_batch = train_batch.transpose(0,4,1,2,3), train_seg_batch.transpose(0,4,1,2,3)

            yield train_batch, train_seg_batch

if __name__ == "__main__":
    dataset = MRIDataset('./CANDI', (160,160,128),label_mapping=label_mapping,batch_size=1,dataname='CANDI')
    (vol_atlas, seg_atlas, ids_atlas), \
    (vol_train, _, ids_train), \
    (vol_test, seg_test, ids_test) = dataset.load_dataset()

    train_gen = dataset.gen_register_batch()




