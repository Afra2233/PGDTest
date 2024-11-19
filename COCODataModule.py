
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
from pycocotools.coco import COCO

class CustomCOCODatasetWithClasses(CocoCaptions):
    def __init__(self, root, annFile,instances_file, transform,**kwargs):
    
        super().__init__(root, annFile, transform=transform)
        self.transform = transform
        self.instances_file=instances_file
        self.cocoInst = COCO(instances_file)
        # print(self.cocoInst.anns.keys())

    def lookup_classes(self, idx):
        ann_id = self.ids[idx]
        # print("ann_id",ann_id)
        image_id = self.coco.anns[ann_id]['image_id']
        #look up the image id in the instances file
        anns = self.cocoInst.loadAnns(self.cocoInst.getAnnIds(imgIds=image_id))
        #print("anns",anns)
        target = [self.cocoInst.loadCats(ann['category_id'])[0]['name'] for ann in anns][0]
        # print("target",target)
        return target
    def __getitem__(self, idx):
        img, target = super(CocoCaptions, self).__getitem__(idx)
           
        captions=random.choice(target)
        captions=clip.tokenize(captions)
        
        try:
            classes=self.lookup_classes(idx)      
        except:
            print("Error looking up classes: ",idx)
            return self.__getitem__(random.randint(0,len(self.ids)-1))

        # if self.transform:
        #     img = self.transform(img)
        return img, classes, captions
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
           
        except:
            print("Error in getting text")
            print("label:",label)
            print("len of dataset:",len(self.dataset))
            print("len of texts:",len(self.tokenized_texts))
            # text="A picture of something"
        # text = self.tokenizer(text) #should be 77 long


        return image, label, text 

class MyDataModule(pl.LightningDataModule):
    def __init__(self,Cache_dir, dataset: str,batch_size: int,test_batch_size:int=-1,imagenet_root: str="none", tinyimagenet_root: str="none",  val_dataset_names: List[str]=None,**kwargs):
        self.cache_dir = Cache_dir
        self.imagenet_root = imagenet_root
        self.tinyimagenet_root = tinyimagenet_root
        self.datasetname = dataset    #not used any more! 
        self.test_batch_size = test_batch_size if test_batch_size>0 else batch_size

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
        self.ISHEC=os.getenv("ISHEC",False)

        self.template = 'This is a photo of a {}'
        self.preprocess = preprocess224_interpolate
        self.tokenizer=clip.tokenize
        self.default=self.tokenizer("A picture of something")
        
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
                from pySmartDL import SmartDL as SmartDL
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
                            # print("Eta: %s" % obj.get_eta(human=True))
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
                                        # print("data_dir:",self.data_dir)
                            print("Extracted: %s" % path)
                #now load the dataset
                annFile=os.path.join(self.cache_dir,"annotations","captions_train2017.json")
                instanceFile=os.path.join(self.cache_dir,"annotations","instances_train2017.json")
                root=os.path.join(self.cache_dir,"train2017")
                dataset_coco=CustomCOCODatasetWithClasses(root,annFile,instanceFile,self.preprocess)
                self.train_dataset_dict.update({"coco":dataset_coco})
                self.train_dataset =dataset_coco
            
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
                       
                if 'STL10' in self.val_dataset_names:
                    val_dataset_dict.update({'STL10': STL10(root=self.imagenet_root, split='test', transform=self.preprocess, download=download)})
                    
                if 'SUN397' in self.val_dataset_names:
                    val_dataset_dict.update({'SUN397': SUN397(root=self.imagenet_root, transform=self.preprocess, download=download)})
                       
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
                            shutil.move(os.path.join(self.tinyimagenet_root,"tiny-imagenet-200","val",'images',val_file),os.path.join(self.tinyimagenet_root,"tiny-imagenet-200","val",val_class))

                        #step four - remove the images folder
                        shutil.rmtree(os.path.join(self.tinyimagenet_root,"tiny-imagenet-200","val",'images'))
                        
                    val_dataset_dict.update({'tinyImageNet': ImageFolder(os.path.join(self.tinyimagenet_root,'tiny-imagenet-200', 'val'), transform=preprocess224)})


                    
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

                        texts_tmp = self.refine_classname(class_names)
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
                self.val_datasets= [CustomtorchVisionDataset2(dataset, texts,self.default) for dataset, texts in zip(self.val_datasets, self.val_texts)]

            #take a 90:10 split of the validation datasets
                splits=[torch.utils.data.random_split(v,[int(0.95*len(v)),len(v)-int(0.95*len(v))]) for v in self.val_datasets]
                self.test_datasets, self.val_datasets= zip(*splits)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=16 ,pin_memory=True,prefetch_factor=4,drop_last=True)

    def val_dataloader(self):
        return [DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4 if not self.ISHEC else 4, pin_memory=not self.ISHEC,prefetch_factor=4 if not self.ISHEC else 2,drop_last=True) for dataset in self.val_datasets]

    def test_dataloader(self):
        return [DataLoader(dataset, batch_size=self.test_batch_size, shuffle=False, num_workers=4 if not self.ISHEC else 4, pin_memory=not self.ISHEC,prefetch_factor=4 if not self.ISHEC else 2,drop_last=True) for dataset in self.test_datasets]






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
