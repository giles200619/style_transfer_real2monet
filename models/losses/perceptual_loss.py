import torch
import torch.nn as nn
from models.perceptual_model import PerceptualVGG19


class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()

        self.perceptual_loss_module = PerceptualVGG19(feature_layers=[21], use_normalization=False)

    def forward(self, input, target):
        fake_features = self.perceptual_loss_module(input)
        real_features = self.perceptual_loss_module(target)
        vgg_tgt = ((fake_features - real_features) ** 2).mean()
        return vgg_tgt


class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()

        self.perceptual_loss_module = PerceptualVGG19(feature_layers=[0, 5, 10, 19, 28], 
                                                      use_normalization=False, is_style = True)
        self.weights = [0.75, 0.5, 0.2, 0.2, 0.2]

    def forward(self, input, target):
        fake_features = self.perceptual_loss_module(input)
        real_features = self.perceptual_loss_module(target)
        style_loss = 0
        for i in range(len(fake_features)):
            fake_gram = gram_matrix(fake_features[i])
            real_gram = gram_matrix(real_features[i])
            layer_style_loss = torch.mean((fake_gram - real_gram) ** 2) * self.weights[i]
            style_loss += layer_style_loss #/ (fake_features[i].shape[1]*fake_features[i].shape[2]*fake_features[i].shape[3])
        return style_loss
    
def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(b * c * d)