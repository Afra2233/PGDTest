

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


def multiGPU_CLIP(model, images, text_tokens):
    # prompt_token = prompt_token.repeat(images.size(0), 1, 1)
    #print images without shape
    img_embed=model.encode_image(images)
    scale_text_embed=model.encode_text(text_tokens)
    img_embed_norm = img_embed / img_embed.norm(dim=-1, keepdim=True)
    scale_text_embed_norm = scale_text_embed / scale_text_embed.norm(dim=-1, keepdim=True)
    logits_per_image = img_embed_norm @ scale_text_embed_norm.t() #cosine similarity
    #logits_per_text = scale_text_embed_norm @ img_embed_norm.t()#图像嵌入和文本嵌入之间的相似度得分矩阵
    return logits_per_image#, logits_per_text, img_embed, scale_text_embed



# def multiGPU_CLIP_NP(model, images, text_tokens):
#     img_embed, scale_text_embed = model(images, text_tokens, None)
#     img_embed_norm = img_embed / img_embed.norm(dim=-1, keepdim=True)
#     scale_text_embed_norm = scale_text_embed / scale_text_embed.norm(dim=-1, keepdim=True)
#     logits_per_image = img_embed_norm @ scale_text_embed_norm.t()
#     #logits_per_text = scale_text_embed_norm @ img_embed_norm.t()
#     return logits_per_image#, logits_per_text, img_embed, scale_text_embed

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
        self.model, _ = clip.load('ViT-B/32', device=self.device, jit=False)
        self.model_ori, _ = clip.load('ViT-B/32', device=self.device, jit=False)
        self.model_text, _= None, None
        self.prompter = NullPrompter()
        self.add_prompter = TokenPrompter(add_prompt_len)
        '''
        To be implemented: place into the token prompter the POS embedding takedn straight fom CLIP, might make the training much faster! , or even try initiialising from random noise properly! 
        (Note, they have several different prompters in the model.prompters.py file, you can use them as a reference)
        '''

        self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion_kl = nn.KLDivLoss(reduction="sum") # mean？why sum?


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
    
    
    #insert function decorator to ensure this ALWAys has grad
    @torch.enable_grad()
    def attack_pgd(self,  X, target, text_tokens, alpha, attack_iters, restarts=1, early_stop=True, epsilon=0):
        delta=self.init_delta(X,epsilon)
        for _ in range(attack_iters):
            # output = model(normalize(X ))
            #prompted_images = self.prompter(normalize(delta + X ))
            #check prompted images has grad
            new_images = delta+X
          
            prompted_images = torch.div(torch.sub(new_images, self.mu_img), self.std_img) #normalize(new_images) but preserves grad
            assert prompted_images.requires_grad
            
            #prompt_token = self.add_prompter()
            output = multiGPU_CLIP(self.model, prompted_images, text_tokens)#, prompt_token)
            loss = self.criterion(output, torch.arange(prompted_images.size(0), device=self.device))
            #range这个函数生成一个从0到 N-1 的整数序列，其中 N 是批次中的图像数量。这个序列在这里作为目标标签，假定每个图像的正确类别或标签就是其索引。
            #交叉熵损失有助于将模型输出（例如，从CLIP等模型获得的相似性得分）解释为概率。通过应用Softmax函数（或Log-Softmax），相似性得分被转换为一个概率分布，这个分布反映了每个类别（或标签）被预测为正确的相对概率。这种概率框架有助于进行更稳健的决策和更细致的性能评估。
            loss.backward()

            #Dear Afra, here is something you should probably log with self.log("attack_loss",loss)
            grad = delta.grad.detach()#这里从delta中提取梯度，并使用.detach()将其从当前计算图中分离出

            '''
            下面的代码是这些行通过直接索引来复制张量数据，
            虽然这种写法在Python中通常不是必需的，
            因为d, g, x和delta, grad, X已经是指向同一数据的不同名称。
            '''
            d = delta[:, :, :, :]
            g = grad[:, :, :, :]
            x = X[:, :, :, :]
            
            '''
            此函数可能基于梯度g和其他参数（如学习率alpha和允许的最大扰动epsilon）计算新的扰动值。
            '''
            d=self.clamp(d,alpha,g,epsilon)
            d = clamp(d, self.lower_limit - x, self.upper_limit - x)
            delta.data[:, :, :, :] = d #计算后的扰动d赋值回delta的数据部分
            delta.grad.zero_() #梯度清零
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
            prompt_token = self.add_prompter()
            output= multiGPU_CLIP(self.model, prompted_images, text_tokens)#, prompt_token)
            label_mask = one_hot_embedding(torch.arange(X.shape(0),device=X.device), output.size(1))
            correct_logit = torch.sum(label_mask * output, dim=1)
            wrong_logit, _ = torch.max((1 - label_mask) * output - 1e4 * label_mask, axis=1)
            # loss = criterion(output, target)
            loss = - torch.sum(F.relu(correct_logit - wrong_logit + 50))
            loss.backward()
            #Dear Afra, here is something you should probably log with self.log("attack_loss",loss)

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

        prompted_clean_images = self.prompter(images) #does nothing - its a null prompter
        Dirtyimages=self.attack(images, target, text, self.args.get("alpha",1), self.args.get("attack_iters",5), epsilon=self.args.get("train_eps",1))
        '''
        Here's where you run the dirty image through your model... first through an encoder, then through a decoder.

        output = model(normalize(images))
        rebuilt_images = model_clean_image_generator(output)
        loss2 = self.YourCriterion(rebuilt_images, images)
        #and add your loss into the total loss. 
        '''

        prompted_Dirtyimages = self.prompter(normalize(Dirtyimages)) #does nothing - its a null prompter
        output_of_training_model_with_dirty_images= multiGPU_CLIP( self.model, prompted_Dirtyimages, text)
        output_of_pretrained_model_with_dirty_images= multiGPU_CLIP( self.model_ori, prompted_Dirtyimages, text)
        '''
        we would assume if the attack is successful, the model would be more confident in the wrong class, so we can do the following check:
        Loss_to_see_attack_success = self.CrossEntropy_loss(output_of_training_model_with_dirty_images, torch.arange(images.size(0), device=self.device))

        '''
        output_of_training_model_with_clean_images = multiGPU_CLIP( self.model, prompted_clean_images, text)
        #This loss stops the divergence of the model from the pretrained model.
        loss_between_our_training_model_and_pretrained_on_dirty_images = self.criterion_kl(F.log_softmax(output_of_training_model_with_dirty_images, dim=1), F.softmax(output_of_pretrained_model_with_dirty_images, dim=1))
        
        #This loss stops the divergence of the model from the clean images.
        loss_between_dirty_and_clean_images_on_training_model = self.criterion_kl(F.log_softmax(output_of_training_model_with_dirty_images, dim=1), F.softmax(output_of_training_model_with_clean_images, dim=1))
        
        #the final criterion is the loss of the model on the dirty images, towards the target.

        '''
        Dear Afra, something for you to try here, 

        I wonder whether balancing the losses using a scaling factor might help preserve overall performance
          (something to try by adding arguments to the demoparse.py file, then setting in the lightning module init.)
        
        '''

        loss_on_training_model_with_dirty_images = self.criterion(output_of_training_model_with_dirty_images, torch.arange(images.size(0), device=self.device)) # 
        loss=loss_on_training_model_with_dirty_images + loss_between_dirty_and_clean_images_on_training_model + loss_between_our_training_model_and_pretrained_on_dirty_images
        
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
        '''
        l2_norm_obj = sum(p.norm(2) for p in self.model.module.visual.parameters())
        l2_norm_ori = sum(p.norm(2) for p in self.model_ori.module.visual.parameters())
        ratio = abs(l2_norm_ori - l2_norm_obj) / float(l2_norm_ori)
        abs_l2 = abs(l2_norm_ori - l2_norm_obj)
        self.log('l2_norm_obj', l2_norm_obj, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('l2_norm_ori', l2_norm_ori, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('ratio', ratio, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('abs_l2', abs_l2, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    
    def on_validation_epoch_start(self):
        self.mu_img = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).to(self.device)
        self.std_img = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).to(self.device)
    def validation_step(self, batch, batch_idx, *args, **kwargs):
        images, target,text = batch
        #a is the image, b is the target
        #get the datamodule text list to lookup the text embeddings.s
     
        prompt_token = None
        print("images shape",images.shape)
        text=text.squeeze(1)
        print("text shape",text.shape)

        print("target shape",target.shape)

        output_prompt= multiGPU_CLIP(self.model, self.prompter(images), text)

        loss = self.criterion(output_prompt, torch.arange(images.size(0), device=self.device))

        # measure accuracy and record loss
        acc1 = accuracy(output_prompt, target, topk=(1,))
        print("acc1",acc1)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc1[0].item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)

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

        prompt_token = self.add_prompter()
        # output_prompt_adv, _ = model(prompter(clip_img_preprocessing(images + delta_prompt)), text_tokens, prompt_token)
        output_prompt_adv = multiGPU_CLIP( self.model,
                                                    self.prompter(clip_img_preprocessing(images + delta_prompt)),
                                                    text) #prommpt token is not used here.maybe should be?

        loss = self.criterion(output_prompt_adv, torch.arange(images.size(0),device=images.device)) #shoudl be torch arange(images.size(0), device=self.device)

        # bl attack
        # torch.cuda.empty_cache()

        # measure accuracy and record loss
        acc1 = accuracy(output_prompt_adv, torch.arange(images.size(0),device=images.device), topk=(1,))
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc1[0].item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        

        return loss

         #You could log here the val_loss, or just print something. 
        
    def configure_optimizers(self):
        # pretty sure we probably want to use the same optimizer as the original paper: the adamw optimizer
        # https://pytorch.org/docs/stable/optim.html#torch.optim.AdamW
        # https://pytorch-lightning.readthedocs.io/en/latest/common/optimizers.html

        optimizer = torch.optim.SGD(list(self.model.visual.parameters()),
                                        lr=self.args.get("learning_rate",1e-5),
                                        momentum=self.args.get("momentum",0.99),
                                        weight_decay=self.args.get("weight_decay",0))
        

        if self.args.get("last_num_ft",-1) != -1:
            optimizer = torch.optim.SGD(self.model.visual.parameters()[-self.args.last_num_ft:], # remember to add the parameters of your model decoder into this line!! 
                                        lr=self.args.get("learning_rate",1e-5),
                                        momentum=self.args.get("momentum",0.99),
                                        weight_decay=self.args.get("weight_decay",0))
        #scheduler = cosine_lr(optimizer, self.args.get("learning_rate",1e-5), self.args.get("warmup",1000), self.args.get("total_steps",100000))
        return optimizer#([optimizer],[scheduler])
