
from torchvision import transforms
from PIL import Image

T= transforms.Compose([transforms.Resize((224,224),interpolation=Image.NEAREST),transforms.ToTensor()])
from transformers import AutoTokenizer
import time
import shutil
import os
import pickle
import json
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import *
from typing import Any, Callable, Optional, Tuple
from PIL import Image
from torch.utils.data import Dataset, DataLoader,default_collate
#import the collate function from pytorch wfw
# from torch.utils.data.dataloader import default_collate
from utils import load_imagenet_folder2name
from datasets import caltech,country211,eurosat,flowers102,oxford_iiit_pet,pcam,stanford_cars,sun397

#import the collate function from pytorch wfw


import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from torchvision.datasets.folder import default_loader
from PIL import Image
from typing import List
from utils import to_rgb,load_imagenet_label2folder,refine_classname
import torchvision.transforms as transforms
import pytorch_lightning as pl
import ssl
ssl._create_default_https_context = ssl._create_unverified_context



ImageNet_MEAN = (0.485, 0.456, 0.406)
ImageNet_STD = (0.229, 0.224, 0.225)



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


import clip

class CustomtorchVisionDataset2(Dataset):
    def __init__(self, dataset, tokenized_text, other_texts):
        self.dataset = dataset
        self.tokenized_texts = tokenized_text
        self.default_text=other_texts
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        text=self.default_text
        try:

            text = self.tokenized_texts[label] #A picture of {label}
            #print("text:",text.shape)
        except:
            print("Error in getting text")
            print("label:",label)
            print("len of dataset:",len(self.dataset))
            print("len of texts:",len(self.tokenized_texts))
            # text="A picture of something"
        # text = self.tokenizer(text) #should be 77 long
        #i keep getting an error saying it's resizing non-resizable storage. This is caused because the image is not in RGB format. ? 



        return image, label, text 



'''
Add COCO captions here ...




'''
class MyDataModule(pl.LightningDataModule):
    def __init__(self,Cache_dir, dataset: str,batch_size: int,test_batch_size:int=-1, imagenet_root: str="./data", tinyimagenet_root: str=None,  val_dataset_names: List[str]=None,**kwargs):
        super().__init__()
        self.cache_dir = Cache_dir
        self.imagenet_root = imagenet_root
        self.tinyimagenet_root = tinyimagenet_root if tinyimagenet_root is not None else self.imagenet_root
        self.datasetname = dataset    #not used any more! 
        #####################################################################################################################
        # self.val_dataset_names = val_dataset_names if val_dataset_names is not None else ['cifar10', 'cifar100', 'STL10', 'Food101',
        #                          'flowers102', 'dtd', 'fgvc_aircraft','tinyImageNet',# 'ImageNet','SUN397'
        #                         'Caltech256', 'PCAM'] 
        # self.train_dataset_names = val_dataset_names if val_dataset_names is not None else ['cifar10', 'cifar100', 'STL10', 'Food101',
        #                         'flowers102', 'dtd', 'fgvc_aircraft','tinyImageNet', 
        #                          'PCAM']   
       #####################################################################################################################

        self.val_dataset_names = val_dataset_names if val_dataset_names is not None else ['tinyImageNet'] 
        self.train_dataset_names = val_dataset_names if val_dataset_names is not None else ['tinyImageNet']
                                 
        self.test_dataset_names = ['tinyImageNet']                     
       
        # self.test_dataset_names = ['cifar10', 'cifar100', 'STL10', 'Food101',
        #                        'flowers102', 'dtd', 'fgvc_aircraft','tinyImageNet',# 'ImageNet','SUN397'
        #                         'Caltech256', 'PCAM''ImageNet','SUN397','oxfordpet', 'EuroSAT','Caltech211']
        # ,'Caltech101'，ImageNet

        self.batch_size = batch_size
        self.test_batch_size = test_batch_size if test_batch_size>0 else batch_size
        if kwargs.get("debug",False):
            print("Debugging")
            print(" ---------------------------------------DEBUGGING---------------------------------------")

            self.val_dataset_names = ['cifar10','cifar100']
            self.train_dataset_names = ['cifar10']

        self.template = 'This is a photo of a {}'
        self.preprocess = preprocess224_interpolate
        self.ISHEC=os.getenv("ISHEC",False)
        self.tokenizer=clip.tokenize
        self.default=self.tokenizer("A picture of something")
        #hopefully this reduces memory needs 
    def prepare_data(self):
        # No preparation needed
        self.setup(download=True)

    def refine_classname(self, class_names):
        class_tokens=[]
        for i, class_name in enumerate(class_names):
            class_names[i] = class_name.lower().replace('_', ' ').replace('-', ' ').replace('/', ' ')
            class_names[i] = self.template.format(class_names[i])
            tokens = self.tokenizer(class_names[i])
            class_tokens.append(tokens)
        return class_tokens
    def setup(self, stage=None,download=False):

        if True: #stage == 'fit' or stage is None or stage == 'test':
            self.train_dataset_dict={}
            self.train_text_names_dict={}
            self.plainnames_dict={}

              #test part 

            if 'cifar100' in self.train_dataset_names:
                self.train_dataset_dict.update({'cifar100': CIFAR100(root=self.imagenet_root, transform=self.preprocess, download=download, train=True)})
                self.plainnames_dict.update({'cifar100':self.train_dataset_dict['cifar100'].classes})
                class_names =self.refine_classname(self.train_dataset_dict['cifar100'].classes)
                self.train_text_names_dict.update({'cifar100':class_names})
            if 'cifar10' in self.train_dataset_names:
                self.train_dataset_dict.update({'cifar10': CIFAR10(root=self.imagenet_root, transform=self.preprocess, download=download, train=True)})
                self.plainnames_dict.update({'cifar10':self.train_dataset_dict['cifar10'].classes})
                class_names =self.refine_classname(self.train_dataset_dict['cifar10'].classes)
                self.train_text_names_dict.update({'cifar10':class_names})
            if 'Caltech101' in self.train_dataset_names:
                self.train_dataset_dict.update({'Caltech101': Caltech101(root=self.imagenet_root, target_type='category', transform=self.preprocess, download=download)})
                self.plainnames_dict.update({'Caltech101':self.train_dataset_dict['Caltech101'].classes})
                class_names =self.refine_classname(self.train_dataset_dict['Caltech101'].classes)
                self.train_text_names_dict.update({'Caltech101':class_names})

            
            if 'STL10' in self.train_dataset_names:
                self.train_dataset_dict.update({'STL10': STL10(root=self.imagenet_root, split='train', transform=self.preprocess, download=download)})
                self.plainnames_dict.update({'STL10':self.train_dataset_dict['STL10'].classes})
                class_names =self.refine_classname(self.train_dataset_dict['STL10'].classes)
                self.train_text_names_dict.update({'STL10':class_names})
            if 'SUN397' in self.train_dataset_names:
                self.train_dataset_dict.update({'SUN397': SUN397(root=self.imagenet_root, transform=self.preprocess, download=download)})
                self.plainnames_dict.update({'SUN397':self.train_dataset_dict['SUN397'].classes})
                class_names =self.refine_classname(self.train_dataset_dict['SUN397'].classes)
                self.train_text_names_dict.update({'SUN397':class_names})
            if 'Food101' in self.train_dataset_names:
                self.train_dataset_dict.update({'Food101': Food101(root=self.imagenet_root, split='train', transform=self.preprocess, download=download)})
                class_names =self.refine_classname(self.train_dataset_dict['Food101'].classes)
                self.plainnames_dict.update({'Food101':self.train_dataset_dict['Food101'].classes})
                self.train_text_names_dict.update({'Food101':class_names})
            if 'oxfordpet' in self.train_dataset_names:
                self.train_dataset_dict.update({'oxfordpet': OxfordIIITPet(root=self.imagenet_root, split='trainval', transform=self.preprocess, download=download)})
                class_names =self.refine_classname(self.train_dataset_dict['oxfordpet'].classes)
                self.plainnames_dict.update({'oxfordpet':self.train_dataset_dict['oxfordpet'].classes})
                self.train_text_names_dict.update({'oxfordpet':class_names})
            if 'EuroSAT' in self.train_dataset_names:
                self.train_dataset_dict.update({'EuroSAT': EuroSAT(root=self.imagenet_root, transform=self.preprocess, download=download)})
                class_names =self.refine_classname(self.train_dataset_dict['EuroSAT'].classes)
                self.plainnames_dict.update({'EuroSAT':self.train_dataset_dict['EuroSAT'].classes})
                self.train_text_names_dict.update({'EuroSAT':class_names})
            if 'Caltech256' in self.train_dataset_names:
                self.train_dataset_dict.update({'Caltech256': Caltech256(root=self.imagenet_root, split=["train"],transform=self.preprocess, download=download)})
                class_names =self.refine_classname(self.train_dataset_dict['Caltech256'].categories)
                self.plainnames_dict.update({'Caltech256':self.train_dataset_dict['Caltech256'].categories})
                self.train_text_names_dict.update({'Caltech256':class_names})
            # if 'flowers102' in self.train_dataset_names:
            #     self.train_dataset_dict.update({'flowers102': Flowers102(root=self.imagenet_root, split='train', transform=self.preprocess, download=download)})
            #     print("flowers102")
            #     print(self.train_dataset_dict['flowers102'].__dir__())
            #     class_names =self.refine_classname(self.train_dataset_dict['flowers102'].)
            #     self.train_text_names_dict.update({'flowers102':[self.template.format(label) for label in class_names]})
            if 'Country211' in self.train_dataset_names:
                try:
                    self.train_dataset_dict.update({'Country211': Country211(root=self.imagenet_root, split='train', transform=self.preprocess, download=download)})
                    class_names =self.refine_classname(self.train_dataset_dict['Country211'].classes)
                    self.plainnames_dict.update({'Country211':self.train_dataset_dict['Country211'].classes})
                    self.train_text_names_dict.update({'Country211':class_names})
                except Exception as e:
                    print(f"[WARNING] Could not load Caltech211: {e}")
            if 'dtd' in self.train_dataset_names:
                self.train_dataset_dict.update({'dtd': DTD(root=self.imagenet_root, split='train', transform=self.preprocess, download=download)})
                class_names =self.refine_classname(self.train_dataset_dict['dtd'].classes)
                self.plainnames_dict.update({'dtd':self.train_dataset_dict['dtd'].classes})
                self.train_text_names_dict.update({'dtd':class_names})
            if 'fgvc_aircraft' in self.train_dataset_names:
                self.train_dataset_dict.update({'fgvc_aircraft': FGVCAircraft(root=self.imagenet_root, split='train', transform=self.preprocess, download=download)})
                class_names =self.refine_classname(self.train_dataset_dict['fgvc_aircraft'].classes)
                self.plainnames_dict.update({'fgvc_aircraft':self.train_dataset_dict['fgvc_aircraft'].classes})
                self.train_text_names_dict.update({'fgvc_aircraft':class_names})
            if 'hateful_memes' in self.train_dataset_names:
                self.train_dataset_dict.update({'hateful_memes': HatefulMemes(root=self.imagenet_root, splits=['train'], transform=self.preprocess,download=download)})
                class_names =self.refine_classname(self.train_dataset_dict['hateful_memes'].classes)
                self.plainnames_dict.update({'hateful_memes':self.train_dataset_dict['hateful_memes'].classes})
                self.train_text_names_dict.update({'hateful_memes':class_names})
            if 'ImageNet' in self.train_dataset_names:
                #download first...
                #get imagenet files and download them

                #
                if not os.path.exists(os.path.join(self.imagenet_root,"ImageNet")):
                    os.makedirs(os.path.join(self.imagenet_root,"ImageNet"),exist_ok=True)
                    # URLS=['http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar',
                    # 'http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar',
                    # 'http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_test.tar',
                    # 'http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_devkit_t12.tar.gz']
                    URLS = ['https://www.image-net.org/data/imagenet10k_eccv2010.tar']
                    for url in URLS:
                        print("Downloading",url)
                        #use pysmartdl to download the files
                        from pySmartDL import SmartDL
                        obj=SmartDL(url,os.path.join(self.imagenet_root,url.split('/')[-1]),progress_bar=True)
                        obj.start()
                        if obj.isSuccessful():
                            print("Downloaded: %s" % obj.get_dest())
                        else:
                            print("There were errors")
                            print(obj.get_errors())
                        #extract the files
                        if url.endswith(".tar"):
                            import tarfile
                            with tarfile.open(obj.get_dest(), 'r') as tar_ref:
                                tar_ref.extractall(self.imagenet_root)
                        elif url.endswith(".tar.gz"):
                            import tarfile
                            with tarfile.open(obj.get_dest(), 'r:gz') as tar_ref:
                                tar_ref.extractall(self.imagenet_root)
                        else:
                            print("Unknown file type")
                        
            


                self.train_dataset_dict.update({'ImageNet': ImageFolder(os.path.join(self.imagenet_root,"ImageNet",'train'), transform=preprocess224)})
                class_names = self.train_dataset_dict['ImageNet'].classes
                class_names = [class_name.lower().replace('_', ' ').replace('-', ' ').replace('/', ' ') for class_name in class_names]
                folder2name = load_imagenet_folder2name('imagenet_classes_names.txt')
                new_class_names = []
                for each in class_names:
                    new_class_names.append(folder2name[each])

                class_names = new_class_names
                self.plainnames_dict.update({'ImageNet':class_names})
                class_names=self.refine_classname(class_names)
                self.train_text_names_dict.update({'ImageNet':class_names})




            if 'tinyImageNet' in self.train_dataset_names:
                if not os.path.exists(os.path.join(self.tinyimagenet_root,"tiny-imagenet-200")):
                    #download tinyimagenet
                    #get tinyimagenet files and download them
                    URLS=['http://cs231n.stanford.edu/tiny-imagenet-200.zip']
                    for url in URLS:
                        print("Downloading",url)
                        #use pysmartdl to download the files
                        from pySmartDL import SmartDL
                        obj=SmartDL(url,os.path.join(self.tinyimagenet_root,url.split('/')[-1]),progress_bar=False)
                        obj.start()
                        if obj.isSuccessful():
                            print("Downloaded: %s" % obj.get_dest())
                        else:
                            print("There were errors")
                            print(obj.get_errors())
                        #extract the files
                        if url.endswith(".zip"):
                            import zipfile
                            with zipfile.ZipFile(obj.get_dest(), 'r') as zip_ref:
                                zip_ref.extractall(self.tinyimagenet_root)
                        else:
                            print("Unknown file type")
                        #load the dataset
                self.train_dataset_dict.update({'tinyImageNet': ImageFolder(os.path.join(self.tinyimagenet_root,'tiny-imagenet-200','train'), transform=preprocess224)})
                class_names = self.train_dataset_dict['tinyImageNet'].classes
                class_names = [class_name.lower().replace('_', ' ').replace('-', ' ').replace('/', ' ') for class_name in class_names]
                folder2name = load_imagenet_folder2name('imagenet_classes_names.txt')
                new_class_names = []
                for each in class_names:
                    new_class_names.append(folder2name[each])

                class_names = new_class_names
                self.plainnames_dict.update({'tinyImageNet':class_names})
                class_names=self.refine_classname(class_names)
                self.train_text_names_dict.update({'tinyImageNet':class_names})

            self.train_datasets = [CustomtorchVisionDataset2(dataset, class_names,self.default) for dataset, class_names in [(self.train_dataset_dict[k], self.train_text_names_dict[k]) for k in self.train_dataset_dict.keys()]]
            self.train_dataset = torch.utils.data.ConcatDataset(self.train_datasets)
            # self.val_datasets = self.load_val_datasets()
            ##################validation datasets##################
            with open(os.path.join(".","train_class_names.json"),'w') as f:
                json.dump(self.plainnames_dict,f)
            val_dataset_dict = {}
        
            if 'cifar10' in self.val_dataset_names:
                val_dataset_dict.update({'cifar10': CIFAR10(root=self.imagenet_root, transform=self.preprocess, download=download, train=True)})
            if 'cifar100' in self.val_dataset_names:
                val_dataset_dict.update({'cifar100': CIFAR100(root=self.imagenet_root, transform=self.preprocess, download=download, train=False)})
            if 'Caltech101'in self.val_dataset_names:
                val_dataset_dict.update({'Caltech101': Caltech101(root=self.imagenet_root, target_type='category', transform=self.preprocess, download=download)})
         
            if 'STL10' in self.val_dataset_names:
                val_dataset_dict.update({'STL10': STL10(root=self.imagenet_root, split='test', transform=self.preprocess, download=download)})
                   
            if 'SUN397' in self.val_dataset_names:
                val_dataset_dict.update({'SUN397': SUN397(root=self.imagenet_root, transform=self.preprocess, download=download)})
                    # val_dataset_list.append(SUN397(root=self.imagenet_root,
                    #                                 transform=preprocess224, download=True))
           
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
                    #download imagenet
                    #get imagenet files and download them
                    if not os.path.exists(os.path.join(self.imagenet_root,"ImageNet")):
                        URLS=['http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar',
                        'http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar',
                        'http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_test.tar',
                        'http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_devkit_t12.tar.gz']
                        for url in URLS:
                            print("Downloading",url)
                            #use pysmartdl to download the files
                            from pySmartDL import SmartDL
                            obj=SmartDL(url,os.path.join(self.imagenet_root,url.split('/')[-1]),progress_bar=False)
                            obj.start()
                            if obj.isSuccessful():
                                print("Downloaded: %s" % obj.get_dest())
                            else:
                                print("There were errors")
                                print(obj.get_errors())
                            #extract the files
                            if url.endswith(".tar"):
                                import tarfile
                                with tarfile.open(obj.get_dest(), 'r') as tar_ref:
                                    tar_ref.extractall(self.imagenet_root)
                            elif url.endswith(".tar.gz"):
                                import tarfile
                                with tarfile.open(obj.get_dest(), 'r:gz') as tar_ref:
                                    tar_ref.extractall(self.imagenet_root)
                            else:
                                print("Unknown file type")
                            #load the dataset
                        val_dataset_dict.update({'ImageNet': ImageFolder(os.path.join(self.imagenet_root, 'val'), transform=preprocess224)})
                        # val_dataset_list.append(ImageFolder(os.path.join(self.imagenet_root, 'val'), transform=preprocess224))
            if 'tinyImageNet' in self.val_dataset_names:
                    #download tinyimagenet
                    #get tinyimagenet files and download them
                    if not os.path.exists(os.path.join(self.tinyimagenet_root,"tiny-imagenet-200")):
                        URLS=['http://cs231n.stanford.edu/tiny-imagenet-200.zip']
                        for url in URLS:
                            print("Downloading",url)
                            #use pysmartdl to download the files
                            from pySmartDL import SmartDL
                            obj=SmartDL(url,os.path.join(self.tinyimagenet_root,url.split('/')[-1]),progress_bar=False)
                            obj.start()
                            if obj.isSuccessful():
                                print("Downloaded: %s" % obj.get_dest())
                            else:
                                print("There were errors")
                                print(obj.get_errors())
                            #extract the files
                            if url.endswith(".zip"):
                                import zipfile
                                with zipfile.ZipFile(obj.get_dest(), 'r') as zip_ref:
                                    zip_ref.extractall(self.tinyimagenet_root)
                            else:
                                print("Unknown file type")
                            #load the dataset
                        #step one: open the val folder at tiny-imagenet-200/val, which is a list of file names and their classes in a text file
                        #step two: make a list of files, and their classes
                        #step three, make a set of folders with the class names, and move the files to the folders
                        #step four: load the dataset
                    if os.path.exists(os.path.join(self.tinyimagenet_root,"tiny-imagenet-200","val",'images')):
                 
                        #step one
                        with open(os.path.join(self.tinyimagenet_root,"tiny-imagenet-200","val","val_annotations.txt"),'r') as f:
                     
                            lines=f.readlines()
                            #step two
                            val_files=[line.split()[0] for line in lines]
                            val_classes=[line.split()[1] for line in lines]
                        #step three
                        for val_file, val_class in zip(val_files,val_classes):
               
                            if not os.path.exists(os.path.join(self.tinyimagenet_root,"tiny-imagenet-200","val",val_class)):
                                os.makedirs(os.path.join(self.tinyimagenet_root,"tiny-imagenet-200","val",val_class),exist_ok=True)
                            if not os.path.exists(os.path.join(self.tinyimagenet_root,"tiny-imagenet-200","val",val_class,val_file)):
                                shutil.move(os.path.join(self.tinyimagenet_root,"tiny-imagenet-200","val",'images',val_file),os.path.join(self.tinyimagenet_root,"tiny-imagenet-200","val",val_class,val_file))
                      
                        #step four - remove the images folder
                        shutil.rmtree(os.path.join(self.tinyimagenet_root,"tiny-imagenet-200","val",'images'))
                        
                    val_dataset_dict.update({'tinyImageNet': ImageFolder(os.path.join(self.tinyimagenet_root,'tiny-imagenet-200', 'val'), transform=preprocess224)})


                    # val_dataset_list.append(ImageFolder(
                    #     os.path.join(self.tinyimagenet_root, 'val'),
                    #     transform=preprocess224))
            
            #concat datasets..                    
            texts_list = []
            for name, each in val_dataset_dict.items():
                if hasattr(each, 'clip_prompts'):
                    texts_tmp = each.clip_prompts
                elif hasattr(each, 'classes'):

                    class_names = each.classes
                    if name == 'ImageNet' or name == 'tinyImageNet':
                        folder2name = load_imagenet_folder2name('imagenet_classes_names.txt')
                        new_class_names = []
                        for class_name in class_names:
                            if folder2name.get("class_name", None) is None:
                                print(f"Class name {class_name} not found in imagenet_classes_names.txt")
                            new_class_names.append(folder2name.get("class_name", class_name))
                        class_names = new_class_names

                    texts_tmp = self.refine_classname(class_names)
                   
                else:
                     #print the names of the datasets that don't have classes
                    print(f"Dataset {name} does not have classes")
                    #and print it's attributes
                    print(dir(each))
                texts_list.append(texts_tmp)
            self.val_datasets = [each for each in val_dataset_dict.values()]
            #print names for each dataset
            print("Names for each val dataset")
            print(["{}, {}".format(idx,each) for idx,each in enumerate(val_dataset_dict.keys())])
            self.val_texts = texts_list
            self.val_datasets= [CustomtorchVisionDataset2(dataset, texts,self.default) for dataset, texts in zip(self.val_datasets, self.val_texts)]
 
 ##################test datasets##################
     #self.test_dataset_names = ['SUN397','oxfordpet', 'EuroSAT','Caltech211', 'hateful_memes','ImageNet','Caltech101']
            test_dataset_dict = {}  
            # if 'Caltech256' in self.train_dataset_names:
            #     self.train_dataset_dict.update({'Caltech256': Caltech256(root=self.imagenet_root, split=["test"],transform=self.preprocess, download=download)})
            #     class_names =self.refine_classname(self.test_dataset_dict['Caltech256'].categories)
            #     self.plainnames_dict.update({'Caltech256':self.test_dataset_dict['Caltech256'].categories})
            #     self.train_text_names_dict.update({'Caltech256':class_names})
            if 'cifar10' in self.test_dataset_names:
                test_dataset_dict.update({'cifar10': CIFAR10(root=self.imagenet_root, transform=self.preprocess, download=download, train=True)})
            if 'cifar100' in self.test_dataset_names:
                test_dataset_dict.update({'cifar100': CIFAR100(root=self.imagenet_root, transform=self.preprocess, download=download, train=False)})
          
            if 'STL10' in self.test_dataset_names:
                test_dataset_dict.update({'STL10': STL10(root=self.imagenet_root, split='test', transform=self.preprocess, download=download)})
                   
            #                                 transform=preprocess224, download=True))
           
            if 'Food101' in self.test_dataset_names: 
                test_dataset_dict.update({'Food101': Food101(root=self.imagenet_root, split='test', transform=self.preprocess, download=download)})  ##is it this one that makes it crash> 
                    # val_dataset_list.append(Food101(root=self.imagenet_root, split='test',
                    #                                 transform=preprocess224, download=True))
            
            if 'dtd' in self.test_dataset_names:
                test_dataset_dict.update({'dtd': DTD(root=self.imagenet_root, split='test', transform=self.preprocess, download=download)})
                    # val_dataset_list.append(DTD(root=self.imagenet_root, split='test',
                                                # transform=preprocess224, download=True))
            if 'fgvc_aircraft' in self.test_dataset_names:
                test_dataset_dict.update({'fgvc_aircraft': FGVCAircraft(root=self.imagenet_root, split='test', transform=self.preprocess, download=download)})
                    # val_dataset_list.append(FGVCAircraft(root=self.imagenet_root, split='test',
                                                            # transform=preprocess224, download=True))
            if 'hateful_memes' in self.test_dataset_names:
                test_dataset_dict.update({'hateful_memes': HatefulMemes(root=self.imagenet_root, splits=['test_seen', 'test_unseen'],
                                                            transform=self.preprocess,download=download)})
            if 'SUN397' in self.test_dataset_names:
                # test_dataset_dict.update({'SUN397': SUN397(root=self.imagenet_root, transform=self.preprocess, download=download)})
                    # val_dataset_list.append(SUN397(root=self.imagenet_root,
                    #                                 transform=preprocess224, download=True))
                test_dataset_dict.update({'SUN397':sun397.SUN397(root=self.imagenet_root,transform=self.preprocess, download=download)})
           
            
            if 'oxfordpet' in self.test_dataset_names:
                test_dataset_dict.update({'oxfordpet': oxford_iiit_pet.OxfordIIITPet(root=self.imagenet_root, split='test', transform=self.preprocess, download=download)})
                    # val_dataset_list.append(OxfordIIITPet(root=self.imagenet_root, split='test',
                    #                                         transform=preprocess224, download=True))
            if 'EuroSAT' in self.test_dataset_names:
                test_dataset_dict.update({'EuroSAT': eurosat.EuroSAT(root=self.imagenet_root, transform=self.preprocess, download=download)})
                    # val_dataset_list.append(EuroSAT(root=self.imagenet_root,
                                                    # transform=preprocess224, download=True))
           
            if 'Country211' in self.test_dataset_names:
                test_dataset_dict.update({'Country211': country211.Country211(root=self.imagenet_root, split='test', transform=self.preprocess, download=download)})
                    # val_dataset_list.append(Country211(root=self.imagenet_root, split='test',
                                                        # transform=preprocess224, download=True))
            if 'tinyImageNet' in self.test_dataset_names:
                test_dataset_dict.update({'tinyImageNet': ImageFolder(
                    os.path.join(self.tinyimagenet_root,'tiny-imagenet-200', 'test'), 
                    transform=preprocess224)})

            # if 'Caltech256' in self.test_dataset_names:
            #     test_dataset_dict.update({'Caltech256': Caltech256(
            #         root=self.imagenet_root,  
            #         transform=self.preprocess, 
                   
            #         download=True)})

            if 'PCAM' in self.test_dataset_names:
                test_dataset_dict.update({'PCAM': pcam.PCAM(
                    root=self.imagenet_root, 
                    split='test', 
                    transform=self.preprocess, 
                    download=True)})

            if 'Caltech211' in self.test_dataset_names:
                test_dataset_dict.update({'Caltech211': country211.Country211(
                    root=self.imagenet_root, 
                    split='test', 
                    transform=self.preprocess, 
                    download=True)})

            if 'flowers102' in self.test_dataset_names:
                test_dataset_dict.update({'flowers102': flowers102.Flowers102(
                    root=self.imagenet_root, 
                    split='test', 
                    transform=self.preprocess, 
                    download=True)})                # transform=preprocess224, download=True))
                        #ft(root=self.imagenet_root, split='test',
                                                                        # transform=preprocess224, download=True))
                        # if 'hateful_memes' in self.test_dataset_names:
                        #     test_dataset_dict.update({'hateful_memes': HatefulMemes(root=self.imagenet_root, splits=['test_seen', 'test_unseen'],
            #                                                 transform=self.preprocess,download=download)})
            if 'Caltech101'in self.test_dataset_names:
                test_dataset_dict.update({'Caltech101': caltech.Caltech101(root=self.imagenet_root, target_type='category', transform=self.preprocess, download=download)})
                     
            if 'ImageNet' in self.test_dataset_names:
                root_imagenet = os.path.join(self.imagenet_root, 'imagenet1k', 'val', 'sorted_images')

                
                test_dataset_dict.update({'ImageNet': ImageFolder(root= root_imagenet, transform=preprocess224)})

                    #download imagenet
                    #get imagenet files and download them
                # if not os.path.exists(os.path.join(self.imagenet_root,"ImageNet1K")):
                #     URLS=['https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar',
                #     'https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train_t3.tar',
                #     'https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar',
                #     'https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_test_v10102019.tar']
                #     # URLS = ['https://www.image-net.org/data/imagenet10k_eccv2010.tar']
                #     for url in URLS:
                #         print("Downloading",url)
                #         #use pysmartdl to download the files
                #         from pySmartDL import SmartDL
                #         obj=SmartDL(url,os.path.join(self.imagenet_root,url.split('/')[-1]),progress_bar=False)
                #         obj.start()
                #         if obj.isSuccessful():
                #             print("Downloaded: %s" % obj.get_dest())
                #         else:
                #             print("There were errors")
                #             print(obj.get_errors())
                #         #extract the files
                #         if url.endswith(".tar"):
                #             import tarfile
                #             with tarfile.open(obj.get_dest(), 'r') as tar_ref:
                #                 tar_ref.extractall(self.imagenet_root)
                #         elif url.endswith(".tar.gz"):
                #             import tarfile
                #             with tarfile.open(obj.get_dest(), 'r:gz') as tar_ref:
                #                 tar_ref.extractall(self.imagenet_root)
                #         else:
                #             print("Unknown file type")
                #         #load the dataset
                # test_dataset_dict.update({'ImageNet': ImageFolder(os.path.join(self.imagenet_root, 'val'), transform=preprocess224)})
                    

      
            texts_list_test = []
            for name, each in test_dataset_dict.items():
                if hasattr(each, 'clip_prompts'):
                    test_texts_tmp = each.clip_prompts
                elif hasattr(each, 'classes'):

                    test_class_names = each.classes
                    if name == 'ImageNet' or name == 'tinyImageNet':
                        test_folder2name = load_imagenet_folder2name('imagenet_classes_names.txt')
                        test_new_class_names = []
                        for class_name in test_class_names:
                            if test_folder2name.get("class_name", None) is None:
                                print(f"Class name {class_name} not found in imagenet_classes_names.txt")
                            test_new_class_names.append(folder2name.get("class_name", class_name))
                        test_class_names =test_new_class_names

                    test_texts_tmp = self.refine_classname(test_class_names)
                   
                else:
                     #print the names of the datasets that don't have classes
                    print(f"Dataset {name} does not have classes")
                    #and print it's attributes
                    print(dir(each))
                texts_list_test.append(test_texts_tmp)
            self.test_datasets_1 = [each for each in test_dataset_dict.values()]
            #print names for each dataset
        
            self.test_texts = texts_list_test
            self.test_datasets_1= [CustomtorchVisionDataset2(dataset, texts,self.default) for dataset, texts in zip(self.test_datasets_1, self.test_texts)]
 
            
            split_datasets = [torch.utils.data.random_split(v, [int(0.95 * len(v)), len(v) - int(0.95 * len(v))]) for v in self.val_datasets]
            self.test_datasets_2 = [split[0] for split in split_datasets]
            self.test_datasets = self.test_datasets_2 + self.test_datasets_1
            print(len(self.test_datasets)) 
            print("Names for test each dataset")
            print(["{}, {}".format(idx,each) for idx,each in enumerate(test_dataset_dict.keys())]) 
          
            self.val_datasets = [split[1] for split in split_datasets]   
            print(len(self.val_datasets))


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=16 if not self.ISHEC else 4 ,pin_memory=not self.ISHEC,prefetch_factor=4 if not self.ISHEC else 2,drop_last=True)

    def val_dataloader(self):
        return [DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=16 if not self.ISHEC else 4, pin_memory=not self.ISHEC,prefetch_factor=4 if not self.ISHEC else 2,drop_last=True) for dataset in self.val_datasets]

    def test_dataloader(self):
        return [DataLoader(dataset, batch_size=self.test_batch_size, shuffle=False, num_workers=4 if not self.ISHEC else 4, pin_memory=not self.ISHEC,prefetch_factor=4 if not self.ISHEC else 2,drop_last=True) for dataset in self.test_datasets]










































 
