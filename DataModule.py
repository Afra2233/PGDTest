
from torchvision import transforms
from PIL import Image

T= transforms.Compose([transforms.Resize((224,224),interpolation=Image.NEAREST),transforms.ToTensor()])
from transformers import AutoTokenizer
import time
import shutil
import os
import pickle
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import *
from typing import Any, Callable, Optional, Tuple
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from utils import load_imagenet_folder2name
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from torchvision.datasets.folder import default_loader
from PIL import Image
from typing import List
from utils import to_rgb,refine_classname,load_imagenet_label2folder


import torchvision.transforms as transforms
import pytorch_lightning as pl



ImageNet_MEAN = (0.485, 0.456, 0.406)
ImageNet_STD = (0.229, 0.224, 0.225)

mu_img = torch.tensor(ImageNet_MEAN).view(3, 1, 1).cuda()
std_img = torch.tensor(ImageNet_STD).view(3, 1, 1).cuda()


preprocess = transforms.Compose([
    transforms.ToTensor()
])
preprocess224_a = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])
preprocess224 = transforms.Compose([
    transforms.Lambda(lambda image: to_rgb(image)),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])
preprocess224_interpolate = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
preprocess112_interpolate = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor()
])



def get_text_prompts_train(args, train_dataset, template='This is a photo of a {}'):
    class_names = train_dataset.classes
    if args.dataset == 'ImageNet' or args.dataset == 'tinyImageNet':
        folder2name = load_imagenet_folder2name('imagenet_classes_names.txt')
        new_class_names = []
        for each in class_names:
            new_class_names.append(folder2name[each])

        class_names = new_class_names

    class_names = refine_classname(class_names)
    texts_train = [template.format(label) for label in class_names]
    #now tokenize it!


    return texts_train


def get_text_prompts_val(val_dataset_list, val_dataset_name, template='This is a photo of a {}'):
    texts_list = []
    for cnt, each in enumerate(val_dataset_list):
        if hasattr(each, 'clip_prompts'):
            texts_tmp = each.clip_prompts
        else:
            class_names = each.classes
            if val_dataset_name[cnt] == 'ImageNet' or val_dataset_name[cnt] == 'tinyImageNet':
                
                folder2name = load_imagenet_folder2name('imagenet_classes_names.txt')
                new_class_names = []
                for class_name in class_names:
                    new_class_names.append(folder2name[class_name])
                class_names = new_class_names

            class_names = refine_classname(class_names)
            texts_tmp = [template.format(label) for label in class_names]
        texts_list.append(texts_tmp)
    assert len(texts_list) == len(val_dataset_list)
    return texts_list


class CustomImageNetDataset(Dataset):
    def __init__(self, img_dir, label_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        with open(label_file, 'r') as f:
            lines = f.readlines()
        self.img_names = [line.split()[0] for line in lines]
        self.labels = [int(line.split()[1]) for line in lines]
        label2name = load_imagenet_label2folder('imagenet_classes_names.txt')
        self.classes = []
        for label in self.labels:
            self.classes.append(label2name[str(label + 1)])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_name)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

class MyDataModule(pl.LightningDataModule):
    def __init__(self,Cache_dir, dataset: str,batch_size: int,imagenet_root: str="none", tinyimagenet_root: str="none",  val_dataset_names: List[str]=None,**kwargs):
        super().__init__()
        self.cache_dir = Cache_dir
        self.imagenet_root = imagenet_root
        self.tinyimagenet_root = tinyimagenet_root
        self.datasetname = dataset
        self.val_dataset_names = val_dataset_names if val_dataset_names is not None else ['cifar10', 'cifar100', 'STL10', 'SUN397', 'Food101',
                                'oxfordpet', 'flowers102', 'dtd', 'fgvc_aircraft',
                                'tinyImageNet', 'ImageNet', 'Caltech101', 'Caltech256', 'PCAM'] #StanfordCars --url; no longer valid. 'EuroSAT' --ssl error
        self.batch_size = batch_size
        self.template = 'This is a photo of a {}'
    
        
    def prepare_data(self):
        # No preparation needed
        
        if self.datasetname == 'cifar100':
            self.dataset= CIFAR100(root=self.imagenet_root, transform=preprocess, download=True, train=True)
        elif self.datasetname == 'cifar10':
            self.dataset= CIFAR10(root=self.imagenet_root, transform=preprocess, download=True, train=True)
        elif self.datasetname == 'ImageNet':
            assert self.imagenet_root is not None
            print(f"Loading ImageNet from {self.imagenet_root}")
            self.dataset= ImageFolder(os.path.join(self.imagenet_root, 'train'), transform=preprocess224)
        elif self.datasetname == 'tinyImageNet':
            assert self.tinyimagenet_root is not None
            print(f"Loading tinyImageNet from {self.tinyimagenet_root}")
            self.dataset= ImageFolder(os.path.join(self.tinyimagenet_root, 'train'), transform=preprocess224)
        else:
            print(f"Train dataset {self.dataset} not implemented")
            raise NotImplementedError
        
        
        class_names = self.dataset.classes
        class_names = refine_classname(class_names)
        if self.datasetname == 'ImageNet' or self.datasetname == 'tinyImageNet':
            from utils import load_imagenet_folder2name
            folder2name = load_imagenet_folder2name('imagenet_classes_names.txt')
            new_class_names = []
            for each in class_names:
                new_class_names.append(folder2name[each])

            class_names = new_class_names
        texts_train = [self.template.format(label) for label in class_names]

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = self.dataset
            # self.val_datasets = self.load_val_datasets()

            val_dataset_list = []
        
            if 'cifar10' in self.val_dataset_names:
                val_dataset_list.append(CIFAR10(root=self.imagenet_root, transform=preprocess, download=True, train=False))
            if 'cifar100' in self.val_dataset_names:
                val_dataset_list.append(CIFAR100(root=self.imagenet_root, transform=preprocess,download=True, train=False))
            if 'Caltech101'in self.val_dataset_names:
                    val_dataset_list.append(Caltech101(root=self.imagenet_root, target_type='category', transform=preprocess224,
                                                        download=True))
            if 'PCAM' in self.val_dataset_names:
                    val_dataset_list.append(PCAM(root=self.imagenet_root, split='test', transform=preprocess224,
                                                    download=True))
            if 'STL10' in self.val_dataset_names:
                    val_dataset_list.append(STL10(root=self.imagenet_root, split='test',
                                                    transform=preprocess, download=True))
            if 'SUN397' in self.val_dataset_names:
                    val_dataset_list.append(SUN397(root=self.imagenet_root,
                                                    transform=preprocess224, download=True))
            # if 'StanfordCars' in self.val_dataset_names:                                                   #no longer available for download
            #         val_dataset_list.append(StanfordCars(root=self.imagenet_root, split='test',
            #                                                 transform=preprocess224, download=True))
            if 'Food101' in self.val_dataset_names: 
                    val_dataset_list.append(Food101(root=self.imagenet_root, split='test',
                                                    transform=preprocess224, download=True))
            if 'oxfordpet' in self.val_dataset_names:
                    val_dataset_list.append(OxfordIIITPet(root=self.imagenet_root, split='test',
                                                            transform=preprocess224, download=True))
            if 'EuroSAT' in self.val_dataset_names:
                    val_dataset_list.append(EuroSAT(root=self.imagenet_root,
                                                    transform=preprocess224, download=True))
            if 'Caltech256' in self.val_dataset_names:
                    val_dataset_list.append(Caltech256(root=self.imagenet_root, transform=preprocess224,
                                                        download=True))
            if 'flowers102' in self.val_dataset_names:
                    val_dataset_list.append(Flowers102(root=self.imagenet_root, split='test',
                                                        transform=preprocess224, download=True))
            if 'Country211' in self.val_dataset_names:
                    val_dataset_list.append(Country211(root=self.imagenet_root, split='test',
                                                        transform=preprocess224, download=True))
            if 'dtd' in self.val_dataset_names:
                    val_dataset_list.append(DTD(root=self.imagenet_root, split='test',
                                                transform=preprocess224, download=True))
            if 'fgvc_aircraft' in self.val_dataset_names:
                    val_dataset_list.append(FGVCAircraft(root=self.imagenet_root, split='test',
                                                            transform=preprocess224, download=True))
            if 'hateful_memes' in self.val_dataset_names:
                    val_dataset_list.append(HatefulMemes(root=self.imagenet_root, splits=['test_seen', 'test_unseen'],
                                                            transform=preprocess224_interpolate))
            if 'ImageNet' in self.val_dataset_names:
                    val_dataset_list.append(ImageFolder(os.path.join(self.imagenet_root, 'val'), transform=preprocess224))
            if 'tinyImageNet' in self.val_dataset_names:
                    val_dataset_list.append(ImageFolder(
                        os.path.join(self.tinyimagenet_root, 'val'),
                        transform=preprocess224))
                
            #concat datasets...
                        

            texts_list = []
            for cnt, each in enumerate(val_dataset_list):
                if hasattr(each, 'clip_prompts'):
                    texts_tmp = each.clip_prompts
                else:
                    class_names = each.classes
                    if self.val_dataset_names[cnt] == 'ImageNet' or self.val_dataset_names[cnt] == 'tinyImageNet':
                        from utils import load_imagenet_folder2name
                        folder2name = load_imagenet_folder2name('imagenet_classes_names.txt')
                        new_class_names = []
                        for class_name in class_names:
                            new_class_names.append(folder2name[class_name])
                        class_names = new_class_names

                    class_names = refine_classname(class_names)
                    texts_tmp = [self.template.format(label) for label in class_names]
                texts_list.append(texts_tmp)
            assert len(texts_list) == len(val_dataset_list)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        return [DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True) for dataset in self.val_datasets]

    def test_dataloader(self):
        return [DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True) for dataset in self.val_datasets]










































## An Example dataset, needs to implement a torch.data.utils.Dataset. This one automatically loads COCO for us from MSCOCO annotations, which we extend to include our own tokenizer

# class myDataset(CocoCaptions):
#     def __init__(self, root, annFile, *args, **kwargs):
#         print('Loading COCO dataset')
#         #check if root and annfile exist
#         if not os.path.exists(root):
#             print('Error: root directory does not exist: {}'.format(root))
#             return None
#         if not os.path.exists(annFile):
#             print('Error: annFile does not exist: {}'.format(annFile))
#             return None
#         #if using the HEC, you may want a Path(annFile).touch() call here to update the altered time if your files are stored in $global_scratch
#         super().__init__(root, annFile, *args, **kwargs)
#         print('Done')
#     def __getitem__(self, index: int):
#         try:
#             img, target= super().__getitem__(index)
#         except Exception as e:
#             print(e)
#             print('Error loading image:', index)
#             return None
#         target=torch.cat([tokenizer(
#                     sent,                      # Sentence to encode.
#                     add_special_tokens = True, # Add '[CLS]' and '[SEP]'
#                     max_length = 77,           # Pad & truncate all sentences.
#                     padding = "max_length",
#                     truncation=True,
#                     return_attention_mask = False,   # Construct attn. masks.
#                     return_tensors = 'pt',     # Return pytorch tensors.
#                 )['input_ids'] for sent in target[:5]],dim=0)
#         return img,target


# # Dataset

# class myDataModule(pl.LightningDataModule):
#     ## This dataModule takes care of downloading the data per node and then PL may replace the sampler if doing distributed multi-node training. 
#     ## Some settings here may be worth editing if on a machine where Pin memory, or workers are limited. 
#     def __init__(self, Cache_dir='./', T=None, batch_size=256):
#         super().__init__()
#         self.data_dir = Cache_dir
#         self.ann_dir=os.path.join(self.data_dir,"annotations")
#         self.batch_size = batch_size
#         self.T=T
#         self.splits={"train":[],"val":[],"test":[]}
        
#     def train_dataloader(self):
#         if not hasattr(self, 'train'):
#             if os.path.exists("train.pt"):
#                 self.train=torch.load("train.pt")
#             else:
#                 self.download_data()
#         # IF you know that you're only ever using 1 gpu (HEC /local runs only...) then consider using https://lightning-bolts.readthedocs.io/en/latest/dataloaders/async.html
#         return torch.utils.data.DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=4, prefetch_factor=4, pin_memory=True,drop_last=True)
#     def val_dataloader(self):
#         if not hasattr(self, 'val'):
#             #check if esprevalidation.pt exists in the directory
#             if os.path.exists("val.pt"):
#                 self.val_dataset=torch.load("val.pt")
#             else:
#                 self.download_data()
#         return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, prefetch_factor=4, pin_memory=True,drop_last=True)
#     def test_dataloader(self):
#         if not hasattr(self, 'test'):
#             #check for espretest.pt in the directory
#             if os.path.exists("test.pt"):
#                 self.test=torch.load("test.pt")
#             else:

#                 self.download_data()

#         return torch.utils.data.DataLoader(self.test, batch_size=self.batch_size, shuffle=True, num_workers=4, prefetch_factor=4, pin_memory=True,drop_last=True)
#     def prepare_data(self):
#         '''called only once and on 1 GPU'''
#         # # download data
#         #If using the HEC, consider altering this to call directly the cmdline WGET/CURL  and then unzip -DD to modify the dates. 
        
#         if not os.path.exists(self.data_dir):
#             os.makedirs(self.data_dir,exist_ok=True)
#         if not os.path.exists(self.ann_dir):
#             os.makedirs(self.ann_dir,exist_ok=True)
#         urls=['http://images.cocodataset.org/zips/train2014.zip',
#                 'http://images.cocodataset.org/zips/val2014.zip',
#                 'http://images.cocodataset.org/zips/test2015.zip',
#                 'http://images.cocodataset.org/zips/train2017.zip',
#                 'http://images.cocodataset.org/zips/val2017.zip',
#                 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
#                 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
#                 ]

#         objs=[]
#         for url in urls:
#             print("url:",url)
#             name=str(url).split('/')[-1]
        
            
#             location=self.data_dir # if name.startswith("annotations") else self.ann_dir
#             #print("Location", location) #/Data/train2014.zip
#             #time.sleep(5)
#             #print('Downloading',url)
#             if name.endswith(".zip"):
#                 name=name[:-4]
#             if name.startswith("train"):
#                 self.splits['train'].append(name)
#             elif name.startswith("val"):
#                 self.splits['val'].append(name)
#             elif name.startswith("test"):
#                 self.splits['test'].append(name)
#             obj=SmartDL(url,os.path.join(location,str(url).split('/')[-1]),progress_bar=False)
#             obj.FileName=name
#             if not os.path.exists(obj.get_dest()):

#                 objs.append(obj)#SmartDL(url, self.data_dir,)
#                 obj.start(blocking=False)
#                 print("obj Path ",obj.get_dest())
#         for obj in objs:
#             while not obj.isFinished():
#                 #print("Speed: %s" % obj.get_speed(human=True))
#                 print("Eta: %s" % obj.get_eta(human=True))
#                 time.sleep(5)
#             if obj.isSuccessful():
#                 print("Downloaded: %s" % obj.get_dest())

#             path = obj.get_dest()
#             if obj.FileName.startswith("annotations"):
#                 print("Extracting annotations")
#                 print("path:",path)

#                 with zipfile.ZipFile(path, 'r') as zip_ref:
#                     try:
#                         zip_ref.extractall(self.data_dir)
#                     except:
#                         print("Error extracting annotations")
#                         print("path:",path)
#                         print("ann_dir:",self.ann_dir)
#             #wget.download("http://images.cocodataset.org/zips/train2014.zip",out=self.cocodir)
#             else:
#                 print("Extracting images")
#                 print("path:",path)
#                 if obj.FileName.endswith(".zip"):
#                     print("Extracting zip")
#                     with zipfile.ZipFile(path, 'r') as zip_ref:
#                         try:
#                             zip_ref.extractall(self.data_dir)
#                         except:
#                             print("Error extracting images")
#                             print("path:",path)
#                             print("data_dir:",self.data_dir)
#                 print("Extracted: %s" % path)


#     def setup(self, stage=None):
#         '''called on each GPU separately - stage defines if we are at fit or test step'''
#         print("Entered COCO datasetup")
#         if stage == 'fit' or stage is None:
#             TrainSets=[]
#             for version in self.splits['train']:
                
#                 annfile=os.path.join(self.ann_dir,'{}_{}.json'.format('captions',version))
#                 dir=os.path.join(self.data_dir,version)

#                 #time.sleep(2)
#                 dset=myDataset(root=dir, annFile=annfile, transform=self.T)
#                 assert(len(dset)>0)
#                 TrainSets.append(dset)
#             self.train = ConcatDataset(TrainSets)

#             ValSets=[]
#             for version in self.splits['val']:
#                 print("BUILDING SPLIT : ",version)
                
#                 annfile=os.path.join(self.ann_dir,'{}_{}.json'.format('captions',version))
#                 dir=os.path.join(self.data_dir,version)
#                 print("annfile:",annfile)
#                 print("dir:",dir)
#                 ValSets.append(myDataset(root=dir, annFile=annfile, transform=self.T))
#             self.val = ConcatDataset(ValSets)
#             # torch.save(self.train,"train.pt")
#             # torch.save(self.val,"val.pt")    
#         if stage == 'test' or stage is None:
#             TestSets=[]
#             for version in self.splits['test']:
#                 print("BUILDING SPLIT : ",version)
#                 annfile=os.path.join(self.ann_dir,'{}_{}.json'.format('captions',version))
#                 dir=os.path.join(self.data_dir,version)
                
#                 print("annfile:",annfile)
#                 print("dir:",dir)
#                 TestSets.append(myDataset(root=dir, annFile=annfile, transform=self.T))
#             self.test = ConcatDataset(TestSets)
