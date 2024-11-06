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
cifar100 = datasets.cifar100(root='./data', train=False, download=True, transform=transform)
image, label = cifar100[50]
class_names = cifar100.classes

#pick one sample from the dataset and get the ground truth label.
image = preprocess(transforms.ToPILImage()(image)).unsqueeze(0).to('cuda')
text_prompts = ["This is a photo of a {}".format(class_names[label])]
text_inputs =clip.tokenize(text_prompts).to('cuda')
text_embedding = model.encode_text(text_inputs).cpu()

#predicted label
image_embedding = model.encode_image(image)
similarity = (image_features @ text_features.T).softmax(dim=-1).cpu().numpy()
best_match_index = similarity.argmax()
predict_prompts = ["This is a photo of a {}".format(class_names[best_match_index])]
predict_inputs =clip.tokenize(predict_prompts).to('cuda')
predict_embedding = model.encode_text(predict_inputs).cpu()

#draw the picture
X_pca = pca.fit_transform(fullpoints.detach().cpu().numpy())
text_pac =pca.transform(text_embedding.detach().numpy())
predict_pac =pca.transform(predict_embedding.detach().numpy())
optimumscore=fullpoints
#normalise the optimum score

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111)
for i, key in enumerate(tokens.keys()):
    points=pca.transform(tokens[key])
    ax.scatter(points[:,0],points[:,1], label=key, alpha=0.5)
ax.scatter(text_pac[:,0],text_pac[:,1],color='Green',marker="x", label='Target Annotation')
ax.scatter(predict_pac[:,0],predict_pac[:,1],color='Yellow',marker="*", label='Prediction Point')

ax.set_title('2D PCA of Text Embeddings for each class')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.legend()
plt.show() 

#save the pca plt
fig.savefig("PCA.png")
