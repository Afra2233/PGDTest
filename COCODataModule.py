
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
from torch.utils.data import Dataset, DataLoader,default_collate
#import the collate function from pytorch 
# from torch.utils.data.dataloader import default_collate

from utils import load_imagenet_folder2name
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from torchvision.datasets.folder import default_loader
from PIL import Image
from typing import List
from utils import to_rgb,refine_classname,load_imagenet_label2folder

from pySmartDL import SmartDL
import torchvision.transforms as transforms
import pytorch_lightning as pl

# def ourCollate_fn(batch):
#     print(batch)
#     try:
#         output=default_collate(batch)
#     except Exception as e:
        
#         print(batch)
#         print("Error in collate")
#         output=ourCollate_fn(batch[:-1])
#     return output

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
import random

def CustomCOCODatasetWithClasses(CocoCaptions):
    def __init__(self, root, annFile, transform=None,**kwargs):
        super(CocoCaptions, self).__init__(root, annFile, transform=transform)
        self.transform = transform

    def lookup_classes(self, idx):
        ann_id = self.ids[idx]
        target = self.coco.anns[ann_id]['category_id']
        return target
    def __getitem__(self, idx):
        img, target = super(CocoCaptions, self).__getitem__(idx)
        classes=self.lookup_classes(idx)      
        captions=random.choice(target)
        captions=clip.tokenize(captions)
        


        if self.transform:
            img = self.transform(img)
        return img, captions, classes
import clip

class CustomtorchVisionDataset2(Dataset):
    def __init__(self, dataset, texts):
        self.dataset = dataset
        self.texts = texts
        self.tokenizer=clip.tokenize
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        text = self.texts[label] #A picture of {label}
        text = self.tokenizer(text) #should be 77 long
        #i keep getting an error saying it's resizing non-resizable storage. This is caused because the image is not in RGB format. ? 



        return image, label, text 

class MyDataModule(pl.LightningDataModule):
    def __init__(self,Cache_dir, dataset: str,batch_size: int,imagenet_root: str="none", tinyimagenet_root: str="none",  val_dataset_names: List[str]=None,**kwargs):
        super().__init__()
        self.cache_dir = Cache_dir
        self.imagenet_root = imagenet_root
        self.tinyimagenet_root = tinyimagenet_root
        self.datasetname = dataset    #not used any more! 
        self.val_dataset_names = val_dataset_names if val_dataset_names is not None else ['cifar10', 'cifar100', 'STL10', 'SUN397', 'Food101',
                                'oxfordpet', 'flowers102', 'dtd', 'fgvc_aircraft',
                                'Caltech256', 'PCAM'] #StanfordCars --url; no longer valid. 'EuroSAT' --ssl error 'Caltech101'- md5? 'tinyImageNet', 'ImageNet', 
        self.train_dataset_names = ["coco"]
        self.batch_size = batch_size
        if kwargs.get("debug",False):
            print("Debugging")
            print(" ---------------------------------------DEBUGGING---------------------------------------")

            self.val_dataset_names = ['cifar10']
            self.train_dataset_names = ['cifar10']

        self.template = 'This is a photo of a {}'
        self.preprocess = preprocess224_interpolate
        
    def prepare_data(self):
        # No preparation needed
        self.setup(download=True)


    def setup(self, stage=None,download=False):

        if stage == 'fit' or stage is None:
            self.train_dataset_dict={}
            self.train_text_names_dict={}
            
            if 'coco' in self.train_dataset_names:
                #download the coco dataset
                from torchvision.datasets import CocoCaptions
                from torchvision.transforms import transforms
                from PIL import Image
                import zipfile
                from torch.utils.data import Dataset, DataLoader
                if not os.path.exists(self.cache_dir):
                    os.makedirs(self.cache_dir,exist_ok=True)
                if not os.path.exists(os.path.join(self.cache_dir,"annotations")):
                    URLS=["http://images.cocodataset.org/zips/train2017.zip","http://images.cocodataset.org/annotations/annotations_trainval2017.zip"]
                    
                    for url in URLS:
                        print("Downloading",url)
                        obj=SmartDL(url,os.path.join(self.cache_dir,str(url).split('/')[-1]),progress_bar=False)
                        obj.FileName=str(url).split('/')[-1]
                        if not os.path.exists(obj.get_dest()):
                            obj.start(blocking=False)
                            print("obj Path ",obj.get_dest())
                        while not obj.isFinished():
                            #print("Speed: %s" % obj.get_speed(human=True))
                            print("Eta: %s" % obj.get_eta(human=True))
                            time.sleep(5)
                        if obj.isSuccessful():
                            print("Downloaded: %s" % obj.get_dest())
                        path = obj.get_dest()
                        if obj.FileName.startswith("annotations"):
                            print("Extracting annotations")
                            print("path:",path)

                            with zipfile.ZipFile(path, 'r') as zip_ref:
                                try:
                                    zip_ref.extractall(self.cache_dir)
                                except:
                                    print("Error extracting annotations")
                                    print("path:",path)
                                    print("ann_dir:",self.ann_dir)
                        else:
                            print("Extracting images")
                            print("path:",path)
                            if obj.FileName.endswith(".zip"):
                                print("Extracting zip")
                                with zipfile.ZipFile(path, 'r') as zip_ref:
                                    try:
                                        zip_ref.extractall(self.cache_dir)
                                    except:
                                        print("Error extracting images")
                                        print("path:",path)
                                        print("data_dir:",self.data_dir)
                            print("Extracted: %s" % path)
                #now load the dataset
                annFile=os.path.join(self.cache_dir,"annotations","captions_train2017.json")
                root=os.path.join(self.cache_dir,"train2017")
                self.train_dataset_dict.update({"coco":CustomCOCODatasetWithClasses(root,annFile,transform=self.preprocess)})
                self.train_text_names_dict.update({"coco":get_text_prompts_train(self, self.train_dataset_dict["coco"])})
            self.train_datasets = [CustomtorchVisionDataset2(dataset, class_names) for dataset, class_names in [(self.train_dataset_dict[k], self.train_text_names_dict[k]) for k in self.train_dataset_dict.keys()]]
            self.train_dataset = torch.utils.data.ConcatDataset(self.train_datasets)
            # self.val_datasets = self.load_val_datasets()
            ##################validation datasets##################
            val_dataset_dict = {}
        
            if 'cifar10' in self.val_dataset_names:
                val_dataset_dict.update({'cifar10': CIFAR10(root=self.imagenet_root, transform=self.preprocess, download=download, train=True)})
            if 'cifar100' in self.val_dataset_names:
                val_dataset_dict.update({'cifar100': CIFAR100(root=self.imagenet_root, transform=self.preprocess, download=download, train=False)})
            if 'Caltech101'in self.val_dataset_names:
                val_dataset_dict.update({'Caltech101': Caltech101(root=self.imagenet_root, target_type='category', transform=self.preprocess, download=download)})
            if 'PCAM' in self.val_dataset_names:
                val_dataset_dict.update({'PCAM': PCAM(root=self.imagenet_root, split='test', transform=self.preprocess, download=download)})
                    # val_dataset_list.append(PCAM(root=self.imagenet_root, split='test', transform=preprocess224,
                    #                                 download=True))
            if 'STL10' in self.val_dataset_names:
                val_dataset_dict.update({'STL10': STL10(root=self.imagenet_root, split='test', transform=self.preprocess, download=download)})
                   
            if 'SUN397' in self.val_dataset_names:
                val_dataset_dict.update({'SUN397': SUN397(root=self.imagenet_root, transform=self.preprocess, download=download)})
                    # val_dataset_list.append(SUN397(root=self.imagenet_root,
                    #                                 transform=preprocess224, download=True))
            # if 'StanfordCars' in self.val_dataset_names:                                                   #no longer available for download
            #         val_dataset_list.append(StanfordCars(root=self.imagenet_root, split='test',
                                                            # transform=preprocess224, download=True))
            if 'Food101' in self.val_dataset_names: 
                val_dataset_dict.update({'Food101': Food101(root=self.imagenet_root, split='test', transform=self.preprocess, download=download)})  ##is it this one that makes it crash> 
                    # val_dataset_list.append(Food101(root=self.imagenet_root, split='test',
                    #                                 transform=preprocess224, download=True))
            if 'oxfordpet' in self.val_dataset_names:
                val_dataset_dict.update({'oxfordpet': OxfordIIITPet(root=self.imagenet_root, split='test', transform=self.preprocess, download=download)})
                    # val_dataset_list.append(OxfordIIITPet(root=self.imagenet_root, split='test',
                    #                                         transform=preprocess224, download=True))
            if 'EuroSAT' in self.val_dataset_names:
                val_dataset_dict.update({'EuroSAT': EuroSAT(root=self.imagenet_root, transform=self.preprocess, download=download)})
                    # val_dataset_list.append(EuroSAT(root=self.imagenet_root,
                                                    # transform=preprocess224, download=True))
            # if 'Caltech256' in self.val_dataset_names:  <==========================This is the wry bastard causing errors! 
            #     val_dataset_dict.update({'Caltech256': Caltech256(root=self.imagenet_root, transform=self.preprocess, download=True)})
            #         # val_dataset_list.append(Caltech256(root=self.imagenet_root, transform=preprocess224,
                                                        # download=True))
            if 'flowers102' in self.val_dataset_names:
                val_dataset_dict.update({'flowers102': Flowers102(root=self.imagenet_root, split='test', transform=self.preprocess, download=download)})
                    # val_dataset_list.append(Flowers102(root=self.imagenet_root, split='test',
                                                        # transform=preprocess224, download=True))
            if 'Country211' in self.val_dataset_names:
                val_dataset_dict.update({'Country211': Country211(root=self.imagenet_root, split='test', transform=self.preprocess, download=download)})
                    # val_dataset_list.append(Country211(root=self.imagenet_root, split='test',
                                                        # transform=preprocess224, download=True))
            if 'dtd' in self.val_dataset_names:
                val_dataset_dict.update({'dtd': DTD(root=self.imagenet_root, split='test', transform=self.preprocess, download=download)})
                    # val_dataset_list.append(DTD(root=self.imagenet_root, split='test',
                                                # transform=preprocess224, download=True))
            if 'fgvc_aircraft' in self.val_dataset_names:
                val_dataset_dict.update({'fgvc_aircraft': FGVCAircraft(root=self.imagenet_root, split='test', transform=self.preprocess, download=download)})
                    # val_dataset_list.append(FGVCAircraft(root=self.imagenet_root, split='test',
                                                            # transform=preprocess224, download=True))
            if 'hateful_memes' in self.val_dataset_names:
                val_dataset_dict.update({'hateful_memes': HatefulMemes(root=self.imagenet_root, splits=['test_seen', 'test_unseen'],
                                                            transform=self.preprocess,download=download)})
            if 'ImageNet' in self.val_dataset_names:
                    val_dataset_list.append(ImageFolder(os.path.join(self.imagenet_root, 'val'), transform=preprocess224))
            if 'tinyImageNet' in self.val_dataset_names:
                    val_dataset_list.append(ImageFolder(
                        os.path.join(self.tinyimagenet_root, 'val'),
                        transform=preprocess224))
                
            #concat datasets...
                        

            texts_list = []
            for name, each in val_dataset_dict.items():
                if hasattr(each, 'clip_prompts'):
                    texts_tmp = each.clip_prompts
                elif hasattr(each, 'classes'):

                    class_names = each.classes
                    if name == 'ImageNet' or name == 'tinyImageNet':
                        from utils import load_imagenet_folder2name
                        folder2name = load_imagenet_folder2name('imagenet_classes_names.txt')
                        new_class_names = []
                        for class_name in class_names:
                            if folder2name.get("class_name", None) is None:
                                print(f"Class name {class_name} not found in imagenet_classes_names.txt")
                            new_class_names.append(folder2name.get("class_name", class_name))
                        class_names = new_class_names

                    class_names = refine_classname(class_names)
                    texts_tmp = [self.template.format(label) for label in class_names]
                else:
                     #print the names of the datasets that don't have classes
                    print(f"Dataset {name} does not have classes")
                    #and print it's attributes
                    print(dir(each))
                texts_list.append(texts_tmp)
            self.val_datasets = [each for each in val_dataset_dict.values()]
            #print names for each dataset
            print("Names for each dataset")
            print(["{}, {}".format(idx,each) for idx,each in enumerate(val_dataset_dict.keys())])
            self.val_texts = texts_list
            self.val_datasets= [CustomtorchVisionDataset2(dataset, texts) for dataset, texts in zip(self.val_datasets, self.val_texts)]


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=16 ,pin_memory=True,prefetch_factor=4,drop_last=True)

    def val_dataloader(self):
        return [DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=16, pin_memory=True,prefetch_factor=4,drop_last=True) for dataset in self.val_datasets]

    def test_dataloader(self):
        return [DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=16, pin_memory=True,prefetch_factor=4,drop_last=True) for dataset in self.val_datasets]






#write some tests to see if the dataloader is working correctly.
if __name__ =='__main__':
    #test the dataloader
    import os
    import torch
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder
    from PIL import Image

    #from COCODataModule import MyDataModule
    import pytorch_lightning as pl
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dataset', type=str, default='coco')
    parser.add_argument('--imagenet_root', type=str, default='./data')
    parser.add_argument('--tinyimagenet_root', type=str, default='./data')
    parser.add_argument('--cache_dir', type=str, default='./data')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    dm = MyDataModule(Cache_dir=args.cache_dir, dataset=args.dataset, batch_size=args.batch_size, imagenet_root=args.imagenet_root, tinyimagenet_root=args.tinyimagenet_root, debug=args.debug)
    dm.setup()
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    for batch in train_loader:
        for each in batch:
            print(each.shape)
        break