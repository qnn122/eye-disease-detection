import torchvision
import torch.nn as nn

#Backbone
class Backbone(nn.Module): 
  def __init__(self, num_classes, is_trained=True):
    super().__init__()
    ##### TODO: Customize your backbone model here######
    #...

    # Load the Resnet50 from ImageNet
    self.net = torchvision.models.resnet.resnet50(pretrained=is_trained)
    
    # Get the input dimension of last layer
    classifier_input_size = self.net.fc.in_features

    # Replace last layer with new layer that have num_classes nodes, after that apply Sigmoid to the output
    self.net.fc = nn.Sequential(nn.Linear(classifier_input_size, num_classes), nn.Sigmoid())

  def forward(self, images):
      return self.net(images)


class RetinaModel(nn.Module): 
  def __init__(self, num_classes):
    super().__init__()
    ##### TODO: Develop your retina model here######
    #...
    self.backbone = Backbone(num_classes,True)

  def forward(self, images):
      x=self.backbone(images)
      return x