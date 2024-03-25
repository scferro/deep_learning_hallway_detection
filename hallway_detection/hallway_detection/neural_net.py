import torch

class NeuralNet(torch.nn.Module):

  def __init__(self, n_classes):
    super(NeuralNet, self).__init__()
    # encoder layers
    self.enc_conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding='same')
    self.enc_conv1_2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding='same')
    self.enc_bn1 = torch.nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True)
    self.enc_pool1 = torch.nn.MaxPool2d((2,2), stride=2)
    self.enc_relu1 = torch.nn.ReLU()
    self.enc_conv2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding='same')
    self.enc_conv2_2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding='same')
    self.enc_bn2 = torch.nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True)
    self.enc_pool2 = torch.nn.MaxPool2d((2,2), stride=2)
    self.enc_relu2 = torch.nn.ReLU()
    self.enc_conv3 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding='same')
    self.enc_conv3_2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding='same')
    self.enc_bn3 = torch.nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True)
    self.enc_pool3 = torch.nn.MaxPool2d((2,2), stride=2)
    self.enc_relu3 = torch.nn.ReLU()
    self.enc_conv4 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding='same')
    self.enc_conv4_2 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding='same')
    self.enc_bn4 = torch.nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.1, affine=True)
    self.enc_pool4 = torch.nn.MaxPool2d((2,2), stride=2)
    self.enc_relu4 = torch.nn.ReLU()
    # decoder layers
    self.dec_up1 = torch.nn.UpsamplingNearest2d(scale_factor=2)
    self.dec_conv1 = torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5, stride=1, padding='same')
    self.dec_conv1_2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding='same')
    self.dec_bn1 = torch.nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True)
    self.dec_relu1 = torch.nn.ReLU()
    self.dec_up2 = torch.nn.UpsamplingNearest2d(scale_factor=2)
    self.dec_conv2 = torch.nn.Conv2d(in_channels=128, out_channels=32, kernel_size=5, stride=1, padding='same')
    self.dec_conv2_2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding='same')
    self.dec_bn2 = torch.nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True)
    self.dec_relu2 = torch.nn.ReLU()
    self.dec_up3 = torch.nn.UpsamplingNearest2d(scale_factor=2)
    self.dec_conv3 = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding='same')
    self.dec_conv3_2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding='same')
    self.dec_bn3 = torch.nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True)
    self.dec_relu3 = torch.nn.ReLU()
    self.dec_up4 = torch.nn.UpsamplingNearest2d(scale_factor=2)
    self.dec_conv4 = torch.nn.Conv2d(in_channels=64, out_channels=3, kernel_size=5, stride=1, padding='same')
    self.dec_conv4_2 = torch.nn.Conv2d(in_channels=3, out_channels=n_classes, kernel_size=5, stride=1, padding='same')
    self.dec_bn4 = torch.nn.BatchNorm2d(num_features=n_classes, eps=1e-05, momentum=0.1, affine=True)
    self.dec_relu4 = torch.nn.ReLU()
    #self.dec_soft = torch.nn.Softmax(dim=1)

  def forward(self, x):
    # encoder forward pass
    a = self.enc_conv1(x)
    a = self.enc_conv1_2(a)
    a = self.enc_bn1(a)
    a = self.enc_relu1(a)
    res1 = self.enc_pool1(a)

    a = self.enc_conv2(res1)
    a = self.enc_conv2_2(a)
    a = self.enc_bn2(a)
    a = self.enc_relu2(a)
    res2 = self.enc_pool2(a)

    a = self.enc_conv3(res2)
    a = self.enc_conv3_2(a)
    a = self.enc_bn3(a)
    a = self.enc_relu3(a)
    res3 = self.enc_pool3(a)

    a = self.enc_conv4(res3)
    a = self.enc_conv4_2(a)
    a = self.enc_bn4(a)
    a = self.enc_relu4(a)
    res4 = self.enc_pool4(a)

    # decoder forward pass
    a = self.dec_up1(res4)
    a = self.dec_conv1(a)
    a = self.dec_conv1_2(a)
    a = self.dec_bn1(a)
    a = self.dec_relu1(a)

    a = torch.cat((a,res3),1)

    a = self.dec_up2(a)
    a = self.dec_conv2(a)
    a = self.dec_conv2_2(a)
    a = self.dec_bn2(a)
    a = self.dec_relu2(a)

    a = torch.cat((a,res2),1)

    a = self.dec_up3(a)
    a = self.dec_conv3(a)
    a = self.dec_conv3_2(a)
    a = self.dec_bn3(a)
    a = self.dec_relu3(a)

    a = torch.cat((a,res1),1)

    a = self.dec_up4(a)
    a = self.dec_conv4(a)
    a = self.dec_conv4_2(a)
    a = self.dec_bn4(a)
    a = self.dec_relu4(a)

    return a #self.dec_soft(a)