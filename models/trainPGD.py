

from pytorch_lightning import LightningModule
import torch.nn as nn
import torch

from utils import cosine_lr
from utils import one_hot_embedding
from utils import accuracy,clamp,normalize
import torch.nn.functional as F
from clip import clip
from models.prompters import TokenPrompter, NullPrompter
from torchattacks import AutoAttack
from utils import clip_img_preprocessing
from sklearn.linear_model import LogisticRegression
import numpy as np


def multiGPU_CLIP(model, images, text_tokens):
   
    #images shape is (batch, 3, 224, 224)
    #text_tokens shape is (batch, 77)
    #the old shape was (C,77)
    #this is why we dont use labels, and use arange instead. 


    img_embed=model.encode_image(images)
    scale_text_embed=model.encode_text(text_tokens)
    img_embed_norm = img_embed / img_embed.norm(dim=-1, keepdim=True)
    scale_text_embed_norm = scale_text_embed / scale_text_embed.norm(dim=-1, keepdim=True)
    logits_per_image = img_embed_norm @ scale_text_embed_norm.t()
    #logits_per_text = scale_text_embed_norm @ img_embed_norm.t()
    return logits_per_image#, logits_per_text, img_embed, scale_text_embed # the shape of output WAS (C,B) but now is (B,B) as we want.


ImageNet_MEAN = (0.485, 0.456, 0.406)
ImageNet_STD = (0.229, 0.224, 0.225)


class myLightningModule(LightningModule):
    '''
    This training code follows the standard structure of Pytorch - lighthning. It's worth looking at their docs for a more in depth dive as to why it is this was
    '''
    
    def __init__(self,
                **args,
                ):

        super().__init__()
        self.save_hyperparameters()
        self.loss=torch.nn.CrossEntropyLoss()
        self.args = args
        add_prompt_len = 0 if args.get("add_prompt","none") == 'none' else 1
        self.upper_limit, self.lower_limit = 1, 0
        self.model, _ = clip.load('ViT-B/32', device=self.device, jit=False,download_root=self.args.get("imagenet_root","./data"))
        self.model_ori, _ = clip.load('ViT-B/32', device=self.device, jit=False,download_root=self.args.get("imagenet_root","./data"))
        self.model_text, _= None, None
        self.prompter = NullPrompter()
        self.add_prompter = TokenPrompter(add_prompt_len)
        '''
        To be implemented: place into the token prompter the POS embedding takedn straight fom CLIP, might make the training much faster! , or even try initiialising from random noise properly! 
        (Note, they have several different prompters in the model.prompters.py file, you can use them as a reference)
        '''

        self.criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        self.criterion_kl = nn.KLDivLoss(reduction="sum")


        '''
        Dear Afra, heres where you put you transformer decoder to build your image! 
        
        i.e  self.model_clean_image_generator = TransformerDecoder()
        
        You probably also want to add a loss function here, and you can do that by adding it to the forward pass.

        self.YourCriterion = nn.CrossEntropyLoss() ? maybe MSE? but I suspect you actually might want DICE loss/ 
        
        '''
        if args.get("norm",'l_inf')=='l_inf':
            self.init_delta=self.init_uniform
            self.clamp=self.clamp_inf
        elif  args.get("norm",'l_inf')=='l_2':
            self.init_delta=self.init_normal
            self.clamp=self.clamp_2
        else:
            raise ValueError
        if not args.get("noAttack",True):
            self.attack=self.no_attack

        self.mu_img = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).to(self.device)
        self.std_img = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).to(self.device)


        
    def init_uniform(self, X,eps):
        delta=  torch.zeros_like(X,device=self.device,).uniform_(-eps, eps)
        delta = clamp(delta, self.lower_limit - X, self.upper_limit - X)
        delta.requires_grad = True
        return delta
    
    def init_normal(self, X,eps):
            delta=torch.zeros_like(X,device=self.device)
            delta.normal_()
            d_flat = delta.view(delta.size(0), -1)
            n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r / n * eps
            delta = clamp(delta, self.lower_limit - X, self.upper_limit - X)
            delta.requires_grad = True
            return delta
    
    def clamp_inf(self,d,alpha,g,eps):
        return torch.clamp(d + alpha * torch.sign(g), min=-eps, max=eps)
    
    def clamp_2(self,d,alpha,g,eps):
        g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
        scaled_g = g / (g_norm + 1e-10)
        d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=eps).view_as(d)
        return d
    
    @torch.enable_grad()
    def attack_text_pgd(self,  X, target, text_tokens, alpha, attack_iters, restarts=1, early_stop=True, epsilon=0):
        delta=self.init_delta(text_tokens,epsilon)
        self.insert_text_model_hook()
        self.model.encode_text(text_tokens) #do this with hooks 
        clean_features=self.text_features
        for _ in range(attack_iters):

            #step 1: modify text tokens
            #step 2: pass through CLIP model module that saves features,
            #step 3: Loss= cosine similarity of clean features to dirty features. 
            #step 4: now consider loss. 
            text_tokens+=delta
            


            img_embed=self.model.encode_image(X)
            #ensure self.model has text hooks 
            self.insert_text_model_hook()
            scale_text_embed=self.model.encode_text(text_tokens)
            features=self.text_features
            #do Loss between each layer
            text_loss=torch.zeros((X.shape[0],X.shape[0]),device=self.device)
            for layer in features.keys():
                itemA=features[layer]
                itemB=clean_features[layer]
                itemA=itemA/itemA.norm(dim=-1, keepdim=True)
                itemB=itemB/itemB.norm(dim=-1, keepdim=True)
                similarities= itemA@itemB.T  # should be B,B in shape, 
                text_loss+=self.CETextLoss(similarities)
            self.log("text_loss",text_loss)

            #step 5: backpropagate, making noise closer to clean features
            text_loss.backward()
            #step 6: remove hooks and zero grad
            self.remove_text_model_hook()
            delta.grad.zero_()


            #step 7: now do attack as normal
            d = delta

            #I want to find a way to maximize the loss while minimizing text loss

            img_embed_norm = img_embed / img_embed.norm(dim=-1, keepdim=True)
            scale_text_embed_norm = scale_text_embed / scale_text_embed.norm(dim=-1, keepdim=True)
            logits_per_image = img_embed_norm @ scale_text_embed_norm.t()
            logits_per_text = scale_text_embed_norm @ img_embed_norm.t()
            # logits_per_text, img_embed, scale_text_embed


            loss = self.criterion(logits_per_text, torch.arange(X.size(0), device=self.device))
            loss.backward()
            self.log("attack_loss",loss)
            grad = delta.grad.detach()
            d = delta[:, :, :, :]
            g = grad[:, :, :, :]
            x = X[:, :, :, :]
            d=self.clamp(d,alpha,g,epsilon)
            d = clamp(d, self.lower_limit - x, self.upper_limit - x)
            delta.data[:, :, :, :] = d
            delta.grad.zero_()
        return delta
    #insert function decorator to ensure this ALWAys has grad
    @torch.enable_grad()
    def attack_pgd(self,  X, target, text_tokens, alpha, attack_iters, restarts=1, early_stop=True, epsilon=0):
        delta=self.init_delta(X,epsilon)
        losses=[]
        scale_text_embed=self.model.encode_text(text_tokens)
        scale_text_embed_norm = scale_text_embed / scale_text_embed.norm(dim=-1, keepdim=True)
        for _ in range(attack_iters):
            # output = model(normalize(X ))
            #prompted_images = self.prompter(normalize(delta + X ))
            #check prompted images has grad
            new_images = delta+X
            prompted_images = torch.div(torch.sub(new_images, self.mu_img), self.std_img) #normalize(new_images) but preserves grad
            img_embed=self.model.encode_image(prompted_images)
            img_embed_norm = img_embed / img_embed.norm(dim=-1, keepdim=True)
            output = img_embed_norm @ scale_text_embed_norm.t()
            loss = self.criterion(output, torch.arange(prompted_images.size(0), device=self.device))
            #range这个函数生成一个从0到 N-1 的整数序列，其中 N 是批次中的图像数量。这个序列在这里作为目标标签，假定每个图像的正确类别或标签就是其索引。
            #交叉熵损失有助于将模型输出（例如，从CLIP等模型获得的相似性得分）解释为概率。通过应用Softmax函数（或Log-Softmax），相似性得分被转换为一个概率分布，这个分布反映了每个类别（或标签）被预测为正确的相对概率。这种概率框架有助于进行更稳健的决策和更细致的性能评估。
            loss.backward()
            losses.append(loss.detach())
            grad = delta.grad.detach()
            d = delta[:, :, :, :]
            g = grad[:, :, :, :]
            x = X[:, :, :, :]
            
            '''
            此函数可能基于梯度g和其他参数（如学习率alpha和允许的最大扰动epsilon）计算新的扰动值。
            '''
            d=self.clamp(d,alpha,g,epsilon)
            d = clamp(d, self.lower_limit - x, self.upper_limit - x)
            delta.data[:, :, :, :] = d
            delta.grad.zero_()
        self.log("mean_attack_losses",sum(losses)/len(losses))
        self.log("max_attack_loss",max(losses))
        self.log("min_attack_loss",min(losses))
        return delta
    
    @torch.enable_grad()
    def attack_pgd_noprompt(self, X, target, text_tokens, alpha, attack_iters, restarts=1, early_stop=True, epsilon=0):
        delta=self.init_delta(X,epsilon)
        for _ in range(attack_iters):
            _images = normalize(X + delta)
            output= multiGPU_CLIP( self.model, _images, text_tokens)
            loss = self.criterion(output,  torch.arange(_images.size(0), device=self.device)) #edited from original paper to remove fixed target classes
            loss.backward()
            #Dear Afra, here is something you should probably log with self.log("attack_loss",loss)
            self.log("attack_loss",loss)
            grad = delta.grad.detach()
            d = delta[:, :, :, :]
            g = grad[:, :, :, :]
            x = X[:, :, :, :]
            d=self.clamp(d,alpha,g,epsilon)
            d = clamp(d, self.lower_limit - x, self.upper_limit - x)
            delta.data[:, :, :, :] = d
            delta.grad.zero_()

        return delta

    @torch.enable_grad()
    def attack_CW(self, X, target, text_tokens, alpha,attack_iters, restarts=1, early_stop=True, epsilon=0):
        delta=self.init_delta(X,epsilon)

        for _ in range(attack_iters):
            # output = model(normalize(X ))
            prompted_images = self.prompter(normalize(X + delta))
            # prompt_token = self.add_prompter()
            output= multiGPU_CLIP(self.model, prompted_images, text_tokens)#, prompt_token)
            '''
            X.shape[0] 获取这个张量第一个维度的大小。在处理图像数据时，这个维度通常是批次大小（batch size），即批次中包含的图像数量。
            '''
            label_mask = one_hot_embedding(torch.arange(X.shape(0),device=X.device), output.size(1)) #每个整数标签都被转换为一个全为0且只有一个位置为1的向量
            correct_logit = torch.sum(label_mask * output, dim=1)
            wrong_logit, _ = torch.max((1 - label_mask) * output - 1e4 * label_mask, axis=1)
            # loss = criterion(output, target)
            loss = - torch.sum(F.relu(correct_logit - wrong_logit + 50))

            loss.backward()
            #Dear Afra, here is something you should probably log with self.log("attack_loss",loss)
            self.log("attack_loss",loss)
            grad = delta.grad.detach()
            d = delta[:, :, :, :]
            g = grad[:, :, :, :]
            x = X[:, :, :, :]
            d=self.clamp(d,alpha,g,epsilon)
            d = clamp(d, self.lower_limit - x, self.upper_limit - x)
            delta.data[:, :, :, :] = d
            delta.grad.zero_()
        return delta
    

    @torch.enable_grad()
    def attack_CW_noprompt(self, X, target, text_tokens, alpha, attack_iters, restarts=1, early_stop=True, epsilon=0):
        delta=self.init_delta(X,epsilon)
        loss=[]
        for _ in range(attack_iters):
            # output = model(normalize(X ))
            _images = normalize(X + delta)
            # output, _ = model(_images, text_tokens)
            output= multiGPU_CLIP(self.model, _images, text_tokens)
            label_mask = one_hot_embedding(torch.arange(X.shape[0],device=X.device), output.size(1))
            correct_logit = torch.sum(label_mask * output, dim=1)
            wrong_logit, _ = torch.max((1 - label_mask) * output - 1e4 * label_mask, axis=1)
            # loss = criterion(output, target)
            loss = - torch.sum(F.relu(correct_logit - wrong_logit + 50))
            #Dear Afra, here is something you should probably log with self.log("attack_loss",loss)
            self.log("attack_loss",loss)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[:, :, :, :]
            g = grad[:, :, :, :]
            x = X[:, :, :, :]
            d=self.clamp(d,alpha,g,epsilon)
            d = clamp(d, self.lower_limit - x, self.upper_limit - x)
            delta.data[:, :, :, :] = d
            delta.grad.zero_()
        return delta

    @torch.enable_grad()
    def attack(self, images, target, text_tokens, alpha, attack_iters, epsilon=0):
            delta = self.attack_pgd( images, target, text_tokens, alpha, attack_iters, epsilon=self.args.get("train_eps",1))
            return images+delta
    
    def no_attack(self, images, *args, **kwargs):
            return images

    def forward(self,input):
        #This inference steps of a foward pass of the model 
        return self.model(input)
    def on_train_epoch_start(self):
        self.mu_img = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).to(self.device)
        self.std_img = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).to(self.device)
    
    def training_step(self, batch, batch_idx):
        #The batch is collated for you, so just seperate it here and calculate loss. 
        #By default, PTL handles optimization and scheduling and logging steps. so All you have to focus on is functionality. Here's an example...
        images, target,text = batch #label shouldnt be used here! 
        text=text.squeeze(1)
        text_embed=self.model.encode_text(text)
        # ori_text_embed=self.model_ori.encode_text(text)
        text_embed= text_embed/ text_embed.norm(dim=-1, keepdim=True)
        # ori_text_embed= ori_text_embed/ ori_text_embed.norm(dim=-1, keepdim=True)
        # images = self.prompter(images) #does nothing - its a null prompter
        Dirtyimages=self.attack(images, target, text, self.args.get("alpha",1), self.args.get("attack_iters",5), epsilon=self.args.get("train_eps",1))
        '''
        Here's where you run the dirty image through your model... first through an encoder, then through a decoder.

        output = model(normalize(images))
        rebuilt_images = model_clean_image_generator(output)
        loss2 = self.YourCriterion(rebuilt_images, images)
        #and add your loss into the total loss. 
        '''
        Dirtyimages = torch.div(torch.sub(Dirtyimages, self.mu_img), self.std_img) #normalize(Dirtyimages) but preserves grad
        # prompted_Dirtyimages = self.prompter(normalize(Dirtyimages)) #does nothing - its a null prompter
        output_of_training_model_with_dirty_images= self.model.encode_image(Dirtyimages) 
        output_of_training_model_with_dirty_images= output_of_training_model_with_dirty_images/ output_of_training_model_with_dirty_images.norm(dim=-1, keepdim=True)
        output_of_training_model_with_clean_images= self.model.encode_image(images)
        output_of_training_model_with_clean_images= output_of_training_model_with_clean_images/ output_of_training_model_with_clean_images.norm(dim=-1, keepdim=True)
        output_of_pretrained_model_with_dirty_images= self.model_ori.encode_image(Dirtyimages)
        output_of_pretrained_model_with_dirty_images= output_of_pretrained_model_with_dirty_images/ output_of_pretrained_model_with_dirty_images.norm(dim=-1, keepdim=True)
        output_of_pretrained_model_with_clean_images= self.model_ori.encode_image(images)
        output_of_pretrained_model_with_clean_images= output_of_pretrained_model_with_clean_images/ output_of_pretrained_model_with_clean_images.norm(dim=-1, keepdim=True)
        '''
        we would assume if the attack is successful, the model would be more confident in the wrong class, so we can do the following check:
        Loss_to_see_attack_success = self.CrossEntropy_loss(output_of_training_model_with_dirty_images, torch.arange(images.size(0), device=self.device))

        '''
        #This loss stops the divergence of the model from the pretrained model.
        loss_between_our_training_model_and_pretrained_on_dirty_images = self.criterion_kl(F.log_softmax(output_of_training_model_with_dirty_images, dim=1), F.softmax(output_of_pretrained_model_with_dirty_images, dim=1))
        loss_between_our_training_model_and_pretrained_on_clean_images = self.criterion_kl(F.log_softmax(output_of_training_model_with_clean_images, dim=1), F.softmax(output_of_pretrained_model_with_clean_images, dim=1))
        
        #This loss stops the divergence of the model from the clean images.
        loss_between_dirty_and_clean_images_on_training_model = self.criterion_kl(F.log_softmax(output_of_training_model_with_dirty_images, dim=1), F.softmax(output_of_training_model_with_clean_images, dim=1))
        
        #the final criterion is the loss of the model on the dirty images, towards the target.

        '''
        Dear Afra, something for you to try here, 

        I wonder whether balancing the losses using a scaling factor might help preserve overall performance
          (something to try by adding arguments to the demoparse.py file, then setting in the lightning module init.)
        
        '''
        logits_of_training_model_with_clean_images = output_of_training_model_with_clean_images @ text_embed.T

        logits_per_dirty_image = output_of_training_model_with_dirty_images @ text_embed.T
        loss_on_training_model_with_dirty_images = self.criterion(logits_per_dirty_image, torch.arange(images.size(0), device=self.device)) # the output of this is huge compared to others. 
        self.log("Loss on training model with clean images (no grad)",self.criterion(logits_of_training_model_with_clean_images, torch.arange(images.size(0), device=self.device)))
        self.log("Loss on training model with dirty images",loss_on_training_model_with_dirty_images)
        self.log("Loss between our training model and pretrained on clean images",loss_between_our_training_model_and_pretrained_on_clean_images )
        self.log("Loss on training model with dirty and clean images",loss_between_dirty_and_clean_images_on_training_model )
        self.log("Loss between our training model and pretrained on dirty images(no_grad)",loss_between_our_training_model_and_pretrained_on_dirty_images )

        loss=loss_on_training_model_with_dirty_images + loss_between_dirty_and_clean_images_on_training_model + loss_between_our_training_model_and_pretrained_on_clean_images #+ loss_between_our_training_model_and_pretrained_on_dirty_images
        
        #self.model.logit_scale.data = torch.clamp(self.model.logit_scale.data, 0, 4.6052)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        #if doing linear regression probes, you may want to have a line like 
        # self.results.append({"imfeatures":self.model(cleanimages), "dirtyfeatures":self.model(attackedImages),"classes":batch[2],"originalmodel":self.orimodel(cleanimages),"dirtyoriginalmodel":self.orimodel(attackedImages)})
        return loss
   

    def on_train_epoch_end(self):
        '''
        imfeatures=torch.nan_to_num(torch.cat([val["imfeatures"] for val in self.results],dim=0)).cpu().numpy()
        #repeat for each output. 
        
        #you can then run a linear regression probe to see how well the model is doing.
        
        #What this tells you is not just "whether the attack works" - we know the attack works!
        #  It tells you instead that the attack is fooling the entire image encoder, not just the relation to the text prompts. the text prompts rely on a template. the template looks like "a photo of ...". you could attack it by making it think its "a cartoon of...".
        #
        
        #draw lots of graphs and stuff.
        
        labels=torch.cat([val["classes"] for val in self.results],dim=0).cpu().numpy()
        if not hasattr(self,"Iclassifier"):
            self.Iclassifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1, n_jobs=-1)
        self.Iclassifier.fit(imfeatures, labels)
        self.log( "ImProbe",self.Iclassifier.score(imfeatures, labels))
        
        .parameters() 方法非常适合进行模型参数的遍历、优化器的配置或进行参数的统计分析
        p.norm(2) 计算每个参数的 L2 范数（即欧几里得范数），并将这些范数求和。这提供了一个量化模型权重总体大小的指标。
         '''
        l2_norm_obj = sum(p.norm(2) for p in self.model.module.visual.parameters())
        l2_norm_ori = sum(p.norm(2) for p in self.model_ori.module.visual.parameters())
        '''
        这行代码计算两个模型的 L2 范数之差的绝对值，然后除以原始模型的 L2 范数，得到一个相对差异比率。这个比率显示了训练模型相对于原始模型参数变化的程度。
        '''

        l2_norm_obj = sum(p.norm(2) for p in self.model.visual.parameters())
        l2_norm_ori = sum(p.norm(2) for p in self.model_ori.visual.parameters())
        ratio = abs(l2_norm_ori - l2_norm_obj) / float(l2_norm_ori)
        '''
        这行简单计算两个模型的 L2 范数之差的绝对值，提供了另一种衡量参数变化的方式。
        '''
        abs_l2 = abs(l2_norm_ori - l2_norm_obj)
        self.log('l2_norm_obj', l2_norm_obj, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('l2_norm_ori', l2_norm_ori, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('ratio', ratio, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('abs_l2', abs_l2, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    
    def on_validation_epoch_start(self):
        self.mu_img = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).to(self.device)
        self.std_img = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).to(self.device)
        self.cleanresults=[]
        self.attackedresults=[]
        self.data_loader_count = len(self.trainer.datamodule.val_dataloader())

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        images, target,text = batch
        #a is the image, b is the target
        #get the datamodule text list to lookup the text embeddings.s

        prompt_token = None
        text=text.squeeze(1)      

        
        img_embed=self.model.encode_image(images)
        scale_text_embed=self.model.encode_text(text)
        img_embed_norm = img_embed / img_embed.norm(dim=-1, keepdim=True)
        scale_text_embed_norm = scale_text_embed / scale_text_embed.norm(dim=-1, keepdim=True)
        output_prompt = img_embed_norm @ scale_text_embed_norm.t()
        # if batch_idx == 0:
        #     #save the first batch of images to disk
        #     #OR project into 2d using PCA and save that to disk
        #     #plot on graph. 
        #     #labels points by class 



        self.cleanresults.append({"logits":img_embed.detach(), "textlabels":target}) #using target like this is fine because each dataloader is tested and logged independently.
        loss = self.criterion(output_prompt, torch.arange(images.size(0), device=self.device))

        # measure accuracy and record loss
        acc1 = accuracy(output_prompt, torch.arange(images.shape[0],device=images.device), topk=(1,))
        self.log('val_clean_batch_loss', loss.detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_clean_batch_acc', acc1[0].item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if self.args.get("CW",False):
            delta_prompt = self.attack_CW(
                                    images, target, text,
                                    self.args.get("test_stepsize",2), self.args.get("test_numsteps",20), epsilon=self.args.get("test_eps",1))
        elif self.args.get("autoattack",False):#autoattack:
            def model_fn(x):
                output_a = multiGPU_CLIP(self.model, self.prompter(clip_img_preprocessing(x)),text)
                return output_a.to(torch.float32)

            adversary = AutoAttack(model_fn, norm='Linf', eps=self.args.get("test_eps",1), version='standard')
            adv_samples = adversary.run_standard_evaluation(images, target, bs=100)   ##is this correct? 
            delta_prompt = adv_samples - images
            delta_prompt = clamp(delta_prompt, self.lower_limit - images, self.upper_limit - images)
        else:
            delta_prompt = self.attack_pgd(images, target, text,self.args.get("test_stepsize",2), self.args.get("test_numsteps",20), epsilon=self.args.get("test_eps",1))

        # output_prompt_adv, _ = model(prompter(clip_img_preprocessing(images + delta_prompt)), text_tokens, prompt_token)


        img_embed=self.model.encode_image(clip_img_preprocessing(images + delta_prompt))
        scale_text_embed=self.model.encode_text(text)
        img_embed_norm = img_embed / img_embed.norm(dim=-1, keepdim=True)
        scale_text_embed_norm = scale_text_embed / scale_text_embed.norm(dim=-1, keepdim=True)
        output_prompt_adv = img_embed_norm @ scale_text_embed_norm.t()


        loss = self.criterion(output_prompt_adv, torch.arange(images.size(0),device=images.device)) #shoudl be torch arange(images.size(0), device=self.device)
        self.attackedresults.append({"logits":img_embed, "textlabels":target})
        # bl attack
        # torch.cuda.empty_cache()

        # measure accuracy and record loss
        acc1 = accuracy(output_prompt_adv, torch.arange(images.size(0),device=images.device), topk=(1,))
        self.log('val_dirty_batch_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_dirty_batch_acc', acc1[0].item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        

        return loss
    def on_validation_epoch_end(self):

        #make linear probes here, and log the results.
        
        GoodLogits=torch.nan_to_num(torch.cat([val["logits"] for val in self.cleanresults],dim=0)).cpu().numpy()
        GoodLabels=torch.cat([val["textlabels"] for val in self.cleanresults],dim=0).cpu().numpy()
        BadLogits=torch.nan_to_num(torch.cat([val["logits"] for val in self.attackedresults],dim=0)).cpu().numpy()
        BadLabels=torch.cat([val["textlabels"] for val in self.attackedresults],dim=0).cpu().numpy()



        if not hasattr(self,"Cleanclassifier"):
            self.Cleanclassifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1, n_jobs=-1)
        if not hasattr(self,"Dirtyclassifier"):
            self.Dirtyclassifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1, n_jobs=-1)
        if not hasattr(self,"generalclassifier"):
            self.generalclassifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1, n_jobs=-1)
        self.Dirtyclassifier.fit(BadLogits, BadLabels)
        self.Cleanclassifier.fit(GoodLogits, GoodLabels)

        self.log( "Clean Classifier on Dirty Features",self.Cleanclassifier.score(BadLogits, BadLabels))
        self.log( "Dirty Classifier on Clean Features",self.Dirtyclassifier.score(GoodLogits, GoodLabels))
        self.log( "Clean Classifier on Clean Features",self.Cleanclassifier.score(GoodLogits, GoodLabels))
        self.log( "Dirty Classifier on Dirty Features",self.Dirtyclassifier.score(BadLogits, BadLabels))
        
        self.generalclassifier.fit(np.concatenate([GoodLogits,BadLogits]), np.concatenate([GoodLabels,BadLabels]))
        self.log( "General Classifier on Dirty Features",self.generalclassifier.score(BadLogits, BadLabels))
        self.log( "General Classifier on Clean Features",self.generalclassifier.score(GoodLogits, GoodLabels))
        self.log( "General Classifier on All Features",self.generalclassifier.score(np.concatenate([GoodLogits,BadLogits]), np.concatenate([GoodLabels,BadLabels])))

        #this should give us PLENTY of data to write about! 
        
        #delete the results to save memory
        del self.cleanresults
        del self.attackedresults

         #You could log here the val_loss, or just print something. 
        
    def configure_optimizers(self):
        # pretty sure we probably want to use the same optimizer as the original paper: the adamw optimizer
        # https://pytorch.org/docs/stable/optim.html#torch.optim.AdamW
        # https://pytorch-lightning.readthedocs.io/en/latest/common/optimizers.html
        args={"lr":self.args.get("learning_rate",1e-5)}
        if self.args.get("optimizer","sgd") == "adamw":
            optimizer_fn=torch.optim.AdamW
            args.update({"betas":(0.9, 0.999),
                  "eps":1e-08,
                  "weight_decay":self.args.get("weight_decay",0.0001)})
        elif self.args.get("optimizer","sgd") == "sgd":
            optimizer_fn=torch.optim.SGD
            args.update({"momentum":self.args.get("momentum",0.9),
                  "weight_decay":self.args.get("weight_decay",0.0001)})

        elif self.args.get("optimizer","sgd") == "adam":
            optimizer_fn=torch.optim.Adam
            args.update({"betas":(0.9, 0.999),
                  "eps":1e-08,
                  "weight_decay":self.args.get("weight_decay",0.0001)})
        else:
            raise ValueError

        #note we've adjusted this to allow the text module to move too! 
        parameters = list(self.model.visual.parameters()) if self.args.get("freeze_text",True) else list(self.model.parameters())
        optimizer = optimizer_fn(parameters,
                                        **args)
        

        if self.args.get("last_num_ft",-1) != -1:
            optimizer = optimizer_fn(parameters[-self.args.last_num_ft:], # remember to add the parameters of your model decoder into this line!! 
                                        **args)
        #scheduler = cosine_lr(optimizer, self.args.get("learning_rate",1e-5), self.args.get("warmup",1000), self.args.get("total_steps",100000))
        return optimizer#([optimizer],[scheduler])
