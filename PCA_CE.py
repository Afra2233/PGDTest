import json
import os
import clip
import torch
from torchvision import transforms
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
import torch.nn.functional as F
from PIL import Image
pca = PCA(n_components=2)
class_names = {}
with open(os.path.join(".","train_class_names.json"),'r') as f:
    class_names = json.load(f)
Loss=torch.nn.CrossEntropyLoss()

tokens={}
model, preprocess = clip.load("ViT-B/32",device='cuda')
with torch.inference_mode(True):
    for key in class_names.keys():
        names=class_names[key]
        print("datasets: ",key)
        print("names: ",names)
        names=clip.tokenize(names).to('cuda')
        tokens.update({key:model.encode_text(names).cpu()})
fullpoints=torch.cat(tuple(list(tokens.values())),axis=0).to(torch.float)
# optimumscore=fullpoints/torch.norm(fullpoints,dim=-1,keepdim=True)
# optimumscore=optimumscore
# optimumscore=optimumscore@optimumscore.T
# LossLabels=torch.arange(0,optimumscore.shape[0],device=optimumscore.device)
# loss=Loss(optimumscore,LossLabels)

transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor() 
])
#load the datasets
cifar100 = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
image, label = cifar100[37] #37,52
class_names_100 = cifar100.classes
print("class_names_100:",class_names_100)

#pick one sample from the dataset and get the ground truth label.
image = preprocess(transforms.ToPILImage()(image)).unsqueeze(0).to('cuda')
text_prompts = ["This is a photo of a {}".format(class_names_100[label])]
print("text_prompts:",text_prompts)
text_inputs =clip.tokenize(text_prompts).to('cuda')
text_embedding = model.encode_text(text_inputs)

#predicted label
text_predict = ["This is a photo of a {}".format(class_name) for class_name in class_names_100]
text_inputs_predict = clip.tokenize(text_predict).to('cuda')
text_embeddings_predict = model.encode_text(text_inputs_predict)
image_embedding = model.encode_image(image)
similarity = (image_embedding @ text_embeddings_predict.T).softmax(dim=-1).cpu().detach().numpy()
print("similarity :",similarity[:10])
best_match_index = similarity.argmax().item()
print("best_match_index:",best_match_index)
print("similarity.argmax:",similarity.argmax())
predict_prompts = ["This is a photo of a {}".format(class_names_100[best_match_index])]
print(predict_prompts)
predict_inputs =clip.tokenize(predict_prompts).to('cuda')
predict_embedding = model.encode_text(predict_inputs).cpu()

def pgd_attack(model, image, label, eps, alpha, num_steps):
    
    perturbed_image = image.clone().detach().to('cuda')
    perturbed_image.requires_grad = True
    
  
    for _ in range(num_steps):
        with torch.enable_grad():
            image_embedding_attack = model.encode_image(perturbed_image)
            loss = -F.cosine_similarity(image_embedding_attack, label, dim=-1).mean()

          
            model.zero_grad()
            loss.backward(retain_graph=True)
            
            
            perturbed_image = perturbed_image + alpha * perturbed_image.grad.sign()
            
           
            perturbation = torch.clamp(perturbed_image - image, min=-eps, max=eps)
            perturbed_image = torch.clamp(image + perturbation, min=0, max=1).detach()
            perturbed_image.requires_grad = True
    
    return perturbed_image
    
epsilons = [1/255,4/255,8/255]   
alphas = [64/255,32/255,125/255]           
num_steps = 10               
attack_point = {}
for eps in epsilons:
    for alpha in alphas:
        
        
        adv_image = pgd_attack(model, image, text_embedding, eps, alpha, num_steps)
        adv_image_squeezed = adv_image.squeeze(0)
           
        with torch.no_grad():
            image_features = model.encode_image(preprocess(transforms.ToPILImage()(adv_image_squeezed)).unsqueeze(0).to('cuda'))
            # text_features = model.encode_text(text_inputs)
            similarity = (image_features @ text_embeddings_predict.T).softmax(dim=-1).cpu().numpy()
            
        print("similarity:",similarity.argmax())
        best_match_index_acctack = similarity.argmax().item()
        predict_prompts_attack = ["This is a photo of a {}".format(class_names_100[best_match_index_acctack])]
        predict_inputs_attack =clip.tokenize(predict_prompts_attack).to('cuda')
    
        attack_point.update({(eps, alpha):model.encode_text(predict_inputs_attack).cpu()})
print("Keys in attack_point:", list(attack_point.keys()))
for key, embedding in attack_point.items():
    print(f"{key}: {embedding.flatten()[:10]}") 

        # acctack_point[(eps, alpha)] = model.encode_text(predict_inputs_attack).cpu()


        
#draw the picture
X_pca = pca.fit_transform(fullpoints.detach().cpu().numpy())
text_pac =pca.transform(text_embedding.detach().cpu().numpy())
predict_pac =pca.transform(predict_embedding.detach().cpu().numpy())
# optimumscore=fullpoints

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111)
for i, key in enumerate(tokens.keys()):
    points=pca.transform(tokens[key])
    ax.scatter(points[:,0],points[:,1], label=key, alpha=0.5)
add_attack_label = True
for (eps, alpha), embedding in attack_point.items():
    point = pca.transform(embedding.detach().cpu().numpy().reshape(1, -1))
    if add_attack_label:
        ax.scatter(point[:, 0], point[:, 1], color='red', marker='x', label='Attacked Point')
        add_attack_label = False  
    else:
        ax.scatter(point[:, 0], point[:, 1], color='red', marker='x')  

    # ax.scatter(point[:, 0], point[:, 1], label=f"eps={eps}, alpha={alpha}", color='red', marker='x')
ax.scatter(text_pac[:,0],text_pac[:,1],color='Black',marker="x", label='Target Annotation')
ax.scatter(predict_pac[:,0],predict_pac[:,1],color='Yellow',marker="*", label='Prediction Point')

ax.set_title('2D PCA of Text Embeddings for each class')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.legend()
plt.show() 

#save the pca plt
fig.savefig("PCA.png")
