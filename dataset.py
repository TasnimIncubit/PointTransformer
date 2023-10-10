import numpy as np
import os
from torch.utils.data import Dataset
import torch
from pointnet_util import farthest_point_sample, pc_normalize
import json

# new addition
# old working values only surface and center points
max_val = np.array([[157.44165455, 174.82071387, 179.34331285]])
min_val = np.array([[-160.78322951, -189.21882333, -167.02990417]])
max_min_diff = np.array([[318.22488406, 364.03953721, 346.37321702]])
mean = np.array([[0.50736506, 0.52182236, 0.46978139]]) 
std = np.array([[0.06373268, 0.06003646, 0.08100745]])
#################################
# modified for pipe center prediction
#################################
def load_data_pipe(partition):

    if partition == 'train':
        # data_dir = '/home/tasnim/Point-Transformers/pipe_data/dataset_uniform_train/numpy_all_points'
        data_dir = '/home/tasnim/from_004/Point-Transformers/pipe_data/dataset_uniform_train/numpy_all_points'
        # data_dir = '/home/tasnim/from_004/Point-Transformers/data_one_train_test'
    else:
        # data_dir = '/home/tasnim/Point-Transformers/pipe_data/dataset_uniform_test/numpy_all_points'
        data_dir = '/home/tasnim/from_004/Point-Transformers/pipe_data/dataset_uniform_test/numpy_all_points'
        # data_dir = '/home/tasnim/from_004/Point-Transformers/data_one_train_test'

    data_batchlist, label_batchlist = [], []
    np_files = sorted(os.listdir(data_dir))
    
    for f in np_files:
        np_file_points = np.load(os.path.join(data_dir, f))
        data = np_file_points[:,0] # shape (4096,3)
        label = np_file_points[:,1] # shape (4096,3)

        d_data = data #  shape (4096,3)
        d_label = label #  shape (4096,3)
        
        data = np.expand_dims(data, axis = 0)#  shape (1,4096,3)
        label = np.expand_dims(label, axis = 0) # shape (1,4096,3)


        # make each value 0 to 1 and then 0 mean 1 std
        data = (data - min_val)/max_min_diff # shape (1,4096,3)
        data = (data - mean)/std # shape (1,4096,3)
        
        label = (label - min_val)/max_min_diff # shape (1,4096,3)
        label = (label - mean)/std # shape (1,4096,3)


        # dist = []
        # for i in range(0, len(d_data), 1):
        #    dist.append(np.linalg.norm(d_data[i]-d_label[i]))
        # delta = np.asarray(dist)

        

        # label = delta # (4096,)
        # label = label.reshape(-1,1) # (4096,1)
        # label = np.expand_dims(label, axis=0) # (1,4096,1)


        

        
        data_batchlist.append(data)
        label_batchlist.append(label)


    data_batches = np.concatenate(data_batchlist, 0)
    seg_batches = np.concatenate(label_batchlist, 0)
    #print('data_batches: ', np.shape(data_batches))
    #print('seg_batches: ', np.shape(seg_batches))

    return data_batches, seg_batches


class ModelNetDataLoader(Dataset):
    def __init__(self, root, npoint=1024, split='train', uniform=False, normal_channel=True, cache_size=15000):
        self.root = root
        self.npoints = npoint
        self.uniform = uniform
        self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.normal_channel = normal_channel

        shape_ids = {}
        shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
        shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d'%(split,len(self.datapath)))

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if index in self.cache:
            point_set, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            cls = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints,:]

            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

            if not self.normal_channel:
                point_set = point_set[:, 0:3]

            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls)

        return point_set, cls

    def __getitem__(self, index):
        return self._get_item(index)


class PartNormalDataset(Dataset):
    def __init__(self, root='./data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500, split='train', class_choice=None, normal_channel=False):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.normal_channel = normal_channel


        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        if not class_choice is  None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}
        # print(self.cat)

        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            # print(fns[0][0:-4])
            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            # print(os.path.basename(fns))
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]

        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        # for cat in sorted(self.seg_classes.keys()):
        #     print(cat, self.seg_classes[cat])

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000


    def __getitem__(self, index):
        if index in self.cache:
            point_set, cls, seg = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
            data = np.loadtxt(fn[1]).astype(np.float32)
            if not self.normal_channel:
                point_set = data[:, 0:3]
            else:
                point_set = data[:, 0:6]
            seg = data[:, -1].astype(np.int32)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        # resample
        point_set = point_set[choice, :]
        seg = seg[choice]

        return point_set, cls, seg

    def __len__(self):
        return len(self.datapath)



class pipe_dataset(Dataset):

    def __init__(self, num_points=4096, partition='train'):
        self.data, self.seg = load_data_pipe(partition)
        #print('self.data:',np.shape(self.data))
        self.partition = partition  
        self.num_points = num_points

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]

        # seg = self.seg[item]
        #print('seg shape = ', np.shape(seg))

        # working line
        seg = self.seg[item][:self.num_points]


        # pointcloud = self.data[item][:self.num_points]
        # seg = self.seg[item][:self.num_points]
        #print('shapes: ', np.shape(pointcloud) )
        if self.partition == 'train':
            indices = list(range(pointcloud.shape[0]))
            np.random.shuffle(indices)
            pointcloud = pointcloud[indices]

            # working line
            seg = seg[indices]

        seg = torch.LongTensor(seg)
        # print('seg tensor shape: ', seg.size())
        # print(pointcloud)
        return pointcloud, seg

    def __len__(self):
        return self.data.shape[0]
    


    # def __init__(self, num_points=4096, partition='train', test_area='1'):
    #     self.data, self.seg = load_data_semseg(partition, test_area)
    #     self.num_points = num_points
    #     self.partition = partition    
    #     self.semseg_colors = load_color_semseg()

    # def __getitem__(self, item):
    #     pointcloud = self.data[item][:self.num_points]
    #     seg = self.seg[item][:self.num_points]
    #     if self.partition == 'train':
    #         indices = list(range(pointcloud.shape[0]))
    #         np.random.shuffle(indices)
    #         pointcloud = pointcloud[indices]
    #         seg = seg[indices]
    #     seg = torch.LongTensor(seg)
    #     return pointcloud, seg

    # def __len__(self):
    #     return self.data.shape[0]


if __name__ == '__main__':
    #data = ModelNetDataLoader('modelnet40_normal_resampled/', split='train', uniform=False, normal_channel=True)
    #DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    #for point,label in DataLoader:
        #print(point.shape)
        #print(label.shape)
    
    pipe_dataset = pipe_dataset(partition='train', num_points=4096)
    pipe_dataloader = torch.utils.data.DataLoader(pipe_dataset, num_workers=8, batch_size=32, shuffle=True, drop_last=True)
    for point,label in pipe_dataloader:
        print(point.shape)
        print(label.shape)
