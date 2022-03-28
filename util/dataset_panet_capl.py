import os
import os.path
import cv2
import numpy as np

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import random
import time
import tqdm

# import SharedArray as SA


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

manual_seed = 123
torch.manual_seed(manual_seed)
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)
torch.cuda.manual_seed_all(manual_seed)
random.seed(manual_seed)

def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(mode, split=0, data_root=None, data_list=None, sub_list=None, sub_val_list=None):    
    assert split in [0, 1, 2, 3, 10, 11, 999]
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))

    # Shaban uses these lines to remove small objects and we strictly follow that
    # if util.change_coordinates(mask, 32.0, 0.0).sum() > 2:
    #    filtered_item.append(item)      
    # which means the mask will be downsampled to 1/32 of the original, 
    # therefore the smallest area should be larger than 2 * 32 * 32    
    image_label_list = []  
    list_read = open(data_list).readlines()
    print("Totally {} samples in {} set.".format(len(list_read), split))
    print("Starting Checking image&label pair {} list...".format(split))
    print("Subclasses are: {}...".format(sub_list))
    print("Processing data...".format(sub_list))
    sub_class_file_list = {}
    for sub_c in sub_list:
        sub_class_file_list[sub_c] = []

    for line in tqdm.tqdm(list_read):
        line = line.strip()
        line_split = line.split(' ')
        image_name = os.path.join(data_root, line_split[0])
        label_name = os.path.join(data_root, line_split[1])
                
        item = (image_name, label_name)
        label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)
        label_class = np.unique(label).tolist()

        if 0 in label_class:
            # remove background class
            label_class.remove(0)
        if 255 in label_class:
            # remove ignore class
            label_class.remove(255)

        valid_flag = 0
        new_label_class = []       
        for c in label_class:
            if c in sub_list:
                valid_flag = 1
                tmp_label = np.zeros_like(label)
                target_pix = np.where(label == c)
                tmp_label[target_pix[0],target_pix[1]] = 1 
                if tmp_label.sum() >= 2 * 32 * 32:      
                    # following Shaban's code: pixel number should be larger than 2 with 32X downsampling
                    new_label_class.append(c)

        label_class = new_label_class    


        if len(label_class) > 0:
            image_label_list.append(item) 
            for c in label_class:
                if c in sub_list:
                    sub_class_file_list[c].append(item)

    print("Checking image&label pair {} list done! All {} pairs.".format(split, len(image_label_list)))
    return image_label_list, sub_class_file_list



class SemData(Dataset):
    def __init__(self, split=3, shot=1, data_root=None, data_list=None, transform=None, mode='train', all_zero_ratio=0.,\
                use_coco=False, save2shm=False):
        assert mode in ['train', 'val', 'test']
        
        self.mode = mode
        self.split = split  
        self.shot = shot
        self.data_root = data_root  
        self.all_zero_ratio = all_zero_ratio    

        if not use_coco:
            self.class_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
            if self.split == 3:
                self.sub_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
                self.sub_val_list = [16,17,18,19,20]
            elif self.split == 2:
                self.sub_list = [1,2,3,4,5,6,7,8,9,10,16,17,18,19,20]
                self.sub_val_list = [11,12,13,14,15]
            elif self.split == 1:
                self.sub_list = [1,2,3,4,5,11,12,13,14,15,16,17,18,19,20]
                self.sub_val_list = [6,7,8,9,10]
            elif self.split == 0:
                self.sub_list = [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
                self.sub_val_list = [1,2,3,4,5]
        else:
            print('INFO: using COCO')
            self.class_list = list(range(1, 81))
            if self.split == 3:
                self.sub_val_list = list(range(4, 81, 4))
                self.sub_list = list(set(self.class_list) - set(self.sub_val_list))                    
            elif self.split == 2:
                self.sub_val_list = list(range(3, 80, 4))
                self.sub_list = list(set(self.class_list) - set(self.sub_val_list))    
            elif self.split == 1:
                self.sub_val_list = list(range(2, 79, 4))
                self.sub_list = list(set(self.class_list) - set(self.sub_val_list))    
            elif self.split == 0:
                self.sub_val_list = list(range(1, 78, 4))
                self.sub_list = list(set(self.class_list) - set(self.sub_val_list))    
  

        save_dir = './saved_npy/'
        if not os.path.exists(save_dir):
            print('===== Create: ', save_dir)
            os.mkdir(save_dir)
        if use_coco:
            path_np_data_list = str(self.mode) + '_split_' + str(split) + '_shot_' + str(1) + '_np_data_list_cocosplit.npy'   # the term shot here does not mean anything
            path_np_sub_list = str(self.mode) + '_split_' + str(split) + '_shot_' + str(1) + '_np_sub_list_cocosplit.npy'                
        else:
            path_np_data_list = str(self.mode) + '_split_' + str(split) + '_shot_' + str(1) + '_np_data_list_pascal.npy'
            path_np_sub_list = str(self.mode) + '_split_' + str(split) + '_shot_' + str(1) + '_np_sub_list_pascal.npy'
        path_np_data_list = os.path.join(save_dir, path_np_data_list)
        path_np_sub_list = os.path.join(save_dir, path_np_sub_list)


        if not os.path.exists(path_np_data_list):
            print('==== Split [{}] | Creating new lists and will save to **{}** and **{}**'.format(split, path_np_data_list, path_np_sub_list))
            if self.mode == 'train':
                self.data_list, self.sub_class_file_list = make_dataset(mode, split, data_root, data_list, self.sub_list, self.sub_val_list)
                assert len(self.sub_class_file_list.keys()) == len(self.sub_list)
            elif self.mode == 'val':
                self.data_list, self.sub_class_file_list = make_dataset(mode, split, data_root, data_list, self.sub_val_list, [])
                assert len(self.sub_class_file_list.keys()) == len(self.sub_val_list) 

            np_data_list = np.array(self.data_list)
            np_sub_list = np.array(self.sub_class_file_list)
            np.save(path_np_data_list, np_data_list)
            np.save(path_np_sub_list, np_sub_list)
        else:
            print('[{}] Loading saved lists from **{}** and **{}**'.format(split, path_np_data_list, path_np_sub_list))
            self.data_list = list(np.load(path_np_data_list))
            try:
                self.sub_class_file_list = np.load(path_np_sub_list).item()       # dict type
            except:
                self.sub_class_file_list = np.load(path_np_sub_list, allow_pickle=True).item()        

        self.use_coco = use_coco
        self.transform = transform
        if self.use_coco:
            self.base_class_num = 61
        else:
            self.base_class_num = 16

        self.save2shm = save2shm
        if self.save2shm:
            ## [DEPRECATED]
            pass


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        label_class = []
        image_path, label_path = self.data_list[index]

        if self.use_coco:                                  
            if not os.path.exists(image_path) or not os.path.exists(label_path):
                print('COCO | {} and {} do not exist!'.format(image_path, label_path))
                exit(0)                       
        else:
            if not os.path.exists(image_path) or not os.path.exists(label_path):
                print('VOC | {} and {} do not exist!'.format(image_path, label_path))
                exit(0)       

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Query Image & label shape mismatch: " + image_path + " " + label_path + "\n")) 
        raw_label = label.copy()
    
        label_class = np.unique(label).tolist()
        if 0 in label_class:
            label_class.remove(0)
        if 255 in label_class:
            label_class.remove(255) 
        new_label_class = []       
        for c in label_class:
            if c in self.sub_val_list:
                if self.mode == 'val' or self.mode == 'test':
                    new_label_class.append(c)
            if c in self.sub_list:
                if self.mode == 'train':
                    new_label_class.append(c)
        label_class = new_label_class    
        assert len(label_class) > 0, print(valid_flag)


        class_chosen = label_class[random.randint(1,len(label_class))-1]
        if self.mode == 'train':   
            if np.random.rand() >= self.all_zero_ratio:
                # if all_zero_ratio is 0, the training is as normal
                # normal
                class_chosen = class_chosen
                target_pix = np.where(label == class_chosen)
                ignore_pix = np.where(label == 255)
                label[:,:] = 0
                if target_pix[0].shape[0] > 0:
                    label[target_pix[0],target_pix[1]] = 1 
                label[ignore_pix[0],ignore_pix[1]] = 255  
            else:
                # hard [DEPRECATED]
                remain_class = list(set(self.sub_list) - set(label_class))
                assert len(remain_class) > 0
                class_chosen = remain_class[random.randint(1,len(remain_class))-1]
                ignore_pix = np.where(label == 255)
                label[:,:] = 0
                label[ignore_pix[0],ignore_pix[1]] = 255 
        else:
            class_chosen = class_chosen
            target_pix = np.where(label == class_chosen)
            ignore_pix = np.where(label == 255)
            label[:,:] = 0
            if target_pix[0].shape[0] > 0:
                label[target_pix[0],target_pix[1]] = 1 
            label[ignore_pix[0],ignore_pix[1]] = 255            

        raw_label_classes = np.unique(raw_label).tolist()
        if 0 in raw_label_classes:
            raw_label_classes.remove(0)
        if 255 in raw_label_classes:
            raw_label_classes.remove(255)
        for c in raw_label_classes:
            x,y = np.where(raw_label == c)
  
            if c in self.sub_list:
                raw_label[x[:], y[:]] = (self.sub_list.index(c) + 1)   
            elif c in self.sub_val_list:
                if self.mode == 'train':          
                    ## following PPNet (ECCV 2020) to set these values to 255.
                    ## https://github.com/Xiangyi1996/PPNet-PyTorch
                    raw_label[x[:], y[:]] = 255
                else:
                    raw_label[x[:], y[:]] = (self.sub_val_list.index(c) + self.base_class_num)  

        if self.mode == 'train':
            ignore_pix = np.where(raw_label == 255)
            label[ignore_pix[0],ignore_pix[1]] = 255 


        file_class_chosen = self.sub_class_file_list[class_chosen]
        num_file = len(file_class_chosen)

        support_image_path_list = []
        support_label_path_list = []
        support_idx_list = []
        for k in range(self.shot):
            support_idx = random.randint(1,num_file)-1
            support_image_path = image_path
            support_label_path = label_path
            while((support_image_path == image_path and support_label_path == label_path) or support_idx in support_idx_list):
                support_idx = random.randint(1,num_file)-1
                support_image_path, support_label_path = file_class_chosen[support_idx]                
            support_idx_list.append(support_idx)
            support_image_path_list.append(support_image_path)
            support_label_path_list.append(support_label_path)

        support_image_list = []
        support_label_list = []
        support_raw_label_list = []
        subcls_list = []
        for k in range(self.shot):  
            if self.mode == 'train':
                subcls_list.append(self.sub_list.index(class_chosen))
            else:
                subcls_list.append(self.sub_val_list.index(class_chosen))
            support_image_path = support_image_path_list[k]
            support_label_path = support_label_path_list[k] 
            if self.use_coco: 
                if not os.path.exists(support_image_path) or not os.path.exists(support_label_path):
                    print('COCO | {} and {} do not exist!'.format(support_image_path, support_label_path))
                    exit(0)                       
            else:
                if not os.path.exists(image_path) or not os.path.exists(label_path):
                    print('VOC | {} and {} do not exist!'.format(support_image_path, support_label_path))
                    exit(0)                   
            
 
            support_image = cv2.imread(support_image_path, cv2.IMREAD_COLOR)      
            support_image = cv2.cvtColor(support_image, cv2.COLOR_BGR2RGB)
            support_image = np.float32(support_image)
            support_label = cv2.imread(support_label_path, cv2.IMREAD_GRAYSCALE)

            support_raw_label = support_label.copy()
            bg_pix = np.where((support_label != 255) * (support_label != class_chosen))
            target_pix = np.where(support_label == class_chosen)
            ignore_pix = np.where(support_label == 255)
            support_label[:,:] = 0
            support_label[target_pix[0],target_pix[1]] = 1
            support_label[bg_pix[0],bg_pix[1]] = 2 # background
            support_label[ignore_pix[0],ignore_pix[1]] = 255
            if support_image.shape[0] != support_label.shape[0] or support_image.shape[1] != support_label.shape[1]:
                raise (RuntimeError("Support Image & label shape mismatch: " + support_image_path + " " + support_label_path + "\n")) 

            raw_label_classes = np.unique(support_raw_label).tolist()
            if 0 in raw_label_classes:
                raw_label_classes.remove(0)
            if 255 in raw_label_classes:
                raw_label_classes.remove(255)
            for c in raw_label_classes:
                x,y = np.where(support_raw_label == c)
    
                if c in self.sub_list:
                    support_raw_label[x[:], y[:]] = (self.sub_list.index(c) + 1)  
                elif c in self.sub_val_list:
                    if self.mode == 'train':    
                        support_raw_label[x[:], y[:]] = 255                                     
                    else:
                        support_raw_label[x[:], y[:]] = (self.sub_val_list.index(c) + self.base_class_num)  

            if self.mode == 'train':
                ignore_pix = np.where(support_raw_label == 255)
                support_label[ignore_pix[0],ignore_pix[1]] = 255 

            support_image_list.append(support_image)
            support_label_list.append(support_label)
            support_raw_label_list.append(support_raw_label)
        assert len(support_label_list) == self.shot and len(support_image_list) == self.shot



        ori_label = label.copy()
        if self.transform is not None:
            image, label, raw_label = self.transform(image, label, raw_label)
            for k in range(self.shot):
                support_image_list[k], support_label_list[k], support_raw_label_list[k] = self.transform(support_image_list[k], support_label_list[k], support_raw_label_list[k])
        
        support_bg_mask_list = []
        for k in range(self.shot):
            bg_pix = np.where(support_label_list[k] == 2)
            ignore_pix = np.where(support_label_list[k] == 255)
            bg_mask = np.zeros_like(support_label_list[k])
            bg_mask[bg_pix[0],bg_pix[1]] = 1
            bg_mask[ignore_pix[0], ignore_pix[1]] = 255
            support_label_list[k][bg_pix[0],bg_pix[1]] = 0
            support_bg_mask_list.append(torch.from_numpy(bg_mask))
            
        s_raw_ys = support_raw_label_list
        s_xs = support_image_list
        s_ys = support_label_list
        s_ys_bg = support_bg_mask_list
        s_x = s_xs[0].unsqueeze(0)
        for i in range(1, self.shot):
            s_x = torch.cat([s_xs[i].unsqueeze(0), s_x], 0)
        s_y = s_ys[0].unsqueeze(0)
        for i in range(1, self.shot):
            s_y = torch.cat([s_ys[i].unsqueeze(0), s_y], 0)
        s_y_bg = s_ys_bg[0].unsqueeze(0)
        for i in range(1, self.shot):
            s_y_bg = torch.cat([s_ys_bg[i].unsqueeze(0), s_y_bg], 0)
        s_raw_y = s_raw_ys[0].unsqueeze(0)
        for i in range(1, self.shot):
            s_raw_y = torch.cat([s_raw_ys[i].unsqueeze(0), s_raw_y], 0)

  
        if self.mode == 'train': 
            return image, label, raw_label, s_x, s_y, s_y_bg, s_raw_y, subcls_list
        else:
            return image, label, raw_label, s_x, s_y, s_y_bg, s_raw_y, subcls_list, ori_label

