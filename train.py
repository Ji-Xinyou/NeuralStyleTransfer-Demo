from load_data import load_images
from losses import content_loss, style_loss
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from transforms import prep, post
from torch.autograd import Variable
from Macros import *

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device('cuda:0')
print('Current Device:', device)

###### image preparations #####
imgs = load_images()
imgs = [prep(img) for img in imgs]
imgs = [Variable(img.unsqueeze(0).to(device)) for img in imgs]

img_con, img_sty = imgs
opt_img = Variable(img_con.data.clone(), requires_grad=True)
###### image preparations #####

###### model preparations #####
print('Optimizing from content image.... Using pretained model vgg19_bn')
model = models.vgg19_bn(pretrained=True).to(device)
# freeze the value, we dont need to update the weights
for param in model.parameters():
    param.requires_grad = False
###### model preparations #####

##### feature containers (through hook) #####
class FeatureSaver(nn.Module):
    feature = None
    def __init__(self, layer):
        self.hook = layer.register_forward_hook(self.hook_func)
    def hook_func(self, module, input, output):
        self.feature = output
    def close(self):
        self.hook.remove()

content_feature_savers = [FeatureSaver(model.features[layer]) for layer in content_layers]
model(Variable(img_con))
content_features = [saver.feature.clone() for saver in content_feature_savers]

print('Saved content features from layer {} of the model'.format(content_layers[0]))

style_feature_savers  = [FeatureSaver(model.features[layer]) for layer in style_layers]
model(Variable(img_sty))
style_features = [saver.feature.clone() for saver in style_feature_savers]

print('Saved style features from layer {} of the model'.format(layer) for layer in style_layers)
##### feature containers #####


optimizer = optim.LBFGS([opt_img])
# optimizer stepping closure
def closure():
    global i
    model(opt_img)
    gen_content_feats = [saver.feature.clone() for saver in content_feature_savers]
    gen_style_feats = [saver.feature.clone() for saver in style_feature_savers]

    contentloss = WEIGHT_CONTENT * content_loss(gen_content_feats, content_features)
    styleloss = style_loss(gen_style_feats, style_features, WEIGHTS_STYLE)
    loss = contentloss + styleloss

    optimizer.zero_grad()
    loss.backward()

    if i % show_iter == 0:
        print('Epoch: {}, Content loss: {}, Style loss: {}, loss: {}'.format\
            (i, contentloss, styleloss, loss))
    i += 1

    return loss

##### Training Section
i = 0
print('Start Training...')
while(i < max_iter):
    optimizer.step(closure)

out_img = post(opt_img.data[0].cpu().squeeze())
out_img.save('.\outputs\\result.png', format='png')