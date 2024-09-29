import torch
from flask import Flask, render_template, request, jsonify, send_file, make_response
from nargsLossCalculation import get_loss_fn
from functools import reduce
from io import BytesIO
import zipfile
from PIL import Image

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

app = Flask(__name__,template_folder='.')



@app.route("/2d") 
def index():
    return render_template("./index.html") #the index page is a menu with 2 buttons that lead to the 2 pages

@app.route("/ImagePage")
def ImagePage():
    return render_template("./ImagePage.html") 



def pgd_attack(model, images, labels, eps=0.4, alpha=0.2, iters=2, clip_min=-1, clip_max=1):
    """
    Performs a PGD attack on a batch of points.

    Args:
        model: A function that takes a batch of points and returns predictions (e.g., a linear model).
        images: The batch of points (numpy array).
        labels: The true labels for the points (list).
        eps: Maximum perturbation magnitude.
        alpha: Step size for the attack.
        iters: Number of attack iterations.
        clip_min: Minimum value for each coordinate.
        clip_max: Maximum value for each coordinate.

    Returns:
        A batch of adversarially perturbed points.
    """
    perturbed_images = np.copy(images)
    for _ in range(iters):
        perturbed_images_tensor = torch.tensor(perturbed_images, dtype=torch.float,requires_grad=True)
        model_output = model(perturbed_images_tensor)  # Placeholder - replace with your actual model

        # Placeholder: Replace with your loss function based on model output and labels
        loss = torch.nn.CrossEntropyLoss()(model_output, torch.tensor(labels))

        loss.backward()
        grad = perturbed_images_tensor.grad.detach().numpy()
        perturbation = alpha * np.sign(grad)
        perturbed_images = perturbed_images + perturbation
        perturbed_images = np.clip(perturbed_images, images - eps, images + eps)
        perturbed_images = np.clip(perturbed_images, clip_min, clip_max)

    return perturbed_images

@app.route('/2d/data', methods=['GET','POST'])
async def perturbPoints():
    data=request.get_json()
    wh=torch.tensor([[data['width'],data['height']]])
    x=[float(x[:-2]) for x in filter(lambda a: a != '',data['x'])]
    y=[float(y[:-2]) for y in filter(lambda a: a != '',data['y'])]
    labels=[int(l) for l in data['labels']]
    xys=[(torch.tensor([[x,y]],requires_grad=False)-(wh/2))/wh for x,y in zip(x,y)]
    
    normed=data['norm']
    alpha=data['alpha']
    epsilon=data['epsilon']
    iters=data['iters']
    if normed:
        xys=xys/torch.norm(xys,keepdim=True,dim=1)
    classes=torch.unique(labels)
    #make linear model with 2 inputs and number of classes
    model=torch.nn.Linear(2,classes.size(0))
    #fit model to points
    optimizer=torch.optim.Adam(model.parameters(),lr=0.01)
    criterion=torch.nn.CrossEntropyLoss()

    for _ in range(100):
        optimizer.zero_grad()
        output=model(xys)
        loss=criterion(output,labels)
        loss.backward()
        optimizer.step()
    perturbed_points=pgd_attack(model, xys, labels, eps=epsilon, alpha=alpha, iters=iters, clip_min=-1, clip_max=1)
    # is a stack of xys, we need to unstack it, multiply by width and height and add width/2 and height/2
    perturbed_points=perturbed_points*wh+wh/2
    #add original labels
    return_dict={'labels':labels,'x':perturbed_points[:,0].tolist(),'y':perturbed_points[:,1].tolist()}
    return return_dict

@app.route('/2d/classifyPoints', methods=['GET','POST'])
async def fitmodeltoPoints():
    data=request.get_json()
    #which was a stringified {'x': x, 'y': y, 'labels': labels, 'method': method};
    x=[float(x[:-2]) for x in filter(lambda a: a != '',data['x'])]
    y=[float(y[:-2]) for y in filter(lambda a: a != '',data['y'])]
    labels=torch.tensor(data['labels'])
    method=data['method']
    if method=="KMeans":
        #fit kmeans to points
        from sklearn.cluster import KMeans
        kmeans=KMeans(n_clusters=len(torch.unique(labels)))
        kmeans.fit(np.array(list(zip(x,y))))
        return kmeans.labels_.tolist()
    elif method=="Linear":
        #fit linear model to points
        from sklearn.linear_model import LogisticRegression
        model=LogisticRegression(max_iter=20)
        model.fit(np.array(list(zip(x,y))),labels)
        return model.predict(np.array(list(zip(x,y)))).tolist()
    else:
        return "Invalid method"




#################IMAGE PAGE####################
import clip
clip_model,preprocess = clip.load('ViT-B/32', device='cpu', jit=False)
template="A photo of a {}"
# in this part we use the model to allow users to upload an image and then we show the attacked image with parameters based on the clip model
@app.route('/ImagePage/data', methods=['GET','POST'])
def ImagePageData():
    data=request.get_json()
    #get the image
    image = Image.open(BytesIO(data['image']))
    #get the model
    #get the epsilon
    epsilon = data['epsilon']
    #get the alpha
    alpha = data['alpha']
    #get the iters
    iters = data['iters']
    #get the clip_min
    clip_min = data['clip_min']
    #get the clip_max
    clip_max = data['clip_max']
    #get the image tensor
    image_tensor = preprocess(image).unsqueeze(0)
    
    #do PGD attack using the class labels from imagenet. 
    perturbed_image = doPGDAttack(image_tensor, epsilon, alpha, iters, clip_min, clip_max)


    return send_file('perturbed_image.png', mimetype='image/png')
t_emb = None
@app.route('/ImagePage/labels', methods=['GET','POST'])
def get_labels():
    global labels
    global template
    global t_emb
    #get the class labels for imagenet
    #get number of classes
    data=request.get_json()
    count=100
    if data.get('number_classes',None) is not None:
        count=data['number_classes']
    with open('imagenet_classes_names.txt') as f:
        labels = [line.strip().split()[2] for line in f]
    labels=labels[:count]
    #encode the labels
    Tokenizedlabels = torch.cat([clip.tokenize(template.format(l)) for l in labels],dim=0).to('cpu')
    t_emb = clip_model.encode_text(Tokenizedlabels)

    return labels

class__idx=0
@app.route('/ImagePage/class', methods=['GET'])
def get_class():
    global class__idx
    return class__idx


def doPGDAttack( image, epsilon, alpha, iters, clip_min, clip_max):
    #get the class labels
    global clip_model
    global labels
    global t_emb
    global class__idx
    #get the perturbed image
    image_clean_emb=clip_model.encode_image(image)
    image_clean_emb = image_clean_emb / image_clean_emb.norm(dim=-1, keepdim=True)
    labels=image_clean_emb@t_emb.T
    delta=torch.zeros_like(image)
    for _ in range(iters):
        
        delta.requires_grad=True
        perturbed_image=image+delta
        perturbed_image_emb=clip_model.encode_image(perturbed_image)
        perturbed_image_emb = perturbed_image_emb / perturbed_image_emb.norm(dim=-1, keepdim=True)
        logits=perturbed_image_emb@t_emb.T

        # Placeholder: Replace with your loss function based on model output and labels
        loss = torch.nn.CrossEntropyLoss()(logits, labels)

        loss.backward()
        grad = delta.grad.detach()
        d = delta[:, :, :, :]
        g = grad[:, :, :, :]
        x = image[:, :, :, :]
        d=torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
        d = torch.clamp(d, clip_min - x, clip_max - x)
        delta.data[:, :, :, :] = d
        delta.grad.zero_()
        class_selection=torch.argmax(logits,dim=1)

    perturbed_image = image + delta
    perturbed_image = perturbed_image.squeeze(0)
    perturbed_image = perturbed_image.permute(1, 2, 0)
    perturbed_image = perturbed_image.cpu().numpy()
    plt.imsave('perturbed_image.png', perturbed_image)
    return perturbed_image











#################debugging####################
if __name__ == "__main__":

        
    app.run(host="0.0.0.0", port=5000 )



