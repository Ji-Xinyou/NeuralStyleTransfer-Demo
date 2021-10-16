import torchvision.transforms as transforms
import torch

img_size = 512
prep = transforms.Compose([transforms.Scale(img_size),
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
                           transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],
                                                std=[1,1,1]),
                           transforms.Lambda(lambda x: x.mul_(255)),
                          ])
recover = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1./255)),
                           transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961],
                                                std=[1,1,1]),
                           transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to RGB
                           ])
toPIL = transforms.Compose([transforms.ToPILImage()])

def post(tensor): 
    tensor = recover(tensor)
    tensor[tensor > 1] = 1    
    tensor[tensor < 0] = 0
    tensor = toPIL(tensor)
    return tensor