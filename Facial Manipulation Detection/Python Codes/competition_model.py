"""Define your architecture here."""
import torch
import torchvision
from models import SimpleNet
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def my_competition_model():
    """Override the model initialization here.

    Do not change the model load line.
    """
    # initialize your model:
    model = get_competition_model()
    # load your model using exactly this line (don't change it):
    model.load_state_dict(torch.load('checkpoints/competition_model.pt')['model'])
    return model


def get_competition_model() -> nn.Module:
    # Building a MobileNet pre-trained and hold it as `my_model`
    my_model = torchvision.models.mobilenet_v2(pretrained=True)
    # Overriding `my_model`'s classifier attribute with the binary classification head stated in the exercise.
    my_model.classifier = MLP_head_MobileNet(my_model.last_channel)
    return my_model


class MLP_head_googleNet(nn.Module):
    def __init__(self):
        super(MLP_head_googleNet, self).__init__()
        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        fully_connected_first_out = F.relu(self.fc1(x))
        fully_connected_second_out = F.relu(self.fc2(fully_connected_first_out))
        two_way_output = self.fc3(fully_connected_second_out)
        return two_way_output


class MLP_head_AlexNet(nn.Module):
    def __init__(self):
        super(MLP_head_AlexNet, self).__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2))

    def forward(self, x):
        two_way_output = self.classifier(x)
        return two_way_output


class MLP_head_ResNet18(nn.Module):
    def __init__(self):
        super(MLP_head_ResNet18, self).__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        fully_connected_first_out = F.relu(self.fc1(x))
        fully_connected_second_out = F.relu(self.fc2(fully_connected_first_out))
        two_way_output = self.fc3(fully_connected_second_out)
        return two_way_output


class MLP_head_MobileNet(nn.Module):
    def __init__(self, last_channel: int):
        super(MLP_head_MobileNet, self).__init__()
        self.fc1 = nn.Linear(last_channel, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        fully_connected_first_out = F.relu(self.fc1(x))
        fully_connected_second_out = F.relu(self.fc2(fully_connected_first_out))
        two_way_output = self.fc3(fully_connected_second_out)
        return two_way_output


class MLP_head_efficientnet_b4(nn.Module):
    def __init__(self):
        super(MLP_head_efficientnet_b4, self).__init__()
        self.fc1 = nn.Linear(1792, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        fully_connected_first_out = F.relu(self.fc1(x))
        fully_connected_second_out = F.relu(self.fc2(fully_connected_first_out))
        two_way_output = self.fc3(fully_connected_second_out)
        return two_way_output


class MLP_head_efficientnet_b3(nn.Module):
    def __init__(self):
        super(MLP_head_efficientnet_b3, self).__init__()
        self.fc1 = nn.Linear(1536, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        fully_connected_first_out = F.relu(self.fc1(x))
        fully_connected_second_out = F.relu(self.fc2(fully_connected_first_out))
        two_way_output = self.fc3(fully_connected_second_out)
        return two_way_output


class MLP_head_efficientnet_b2(nn.Module):
    def __init__(self):
        super(MLP_head_efficientnet_b2, self).__init__()
        self.fc1 = nn.Linear(1408, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        fully_connected_first_out = F.relu(self.fc1(x))
        fully_connected_second_out = F.relu(self.fc2(fully_connected_first_out))
        two_way_output = self.fc3(fully_connected_second_out)
        return two_way_output


class MLP_head_efficientnet_b1(nn.Module):
    def __init__(self):
        super(MLP_head_efficientnet_b1, self).__init__()
        self.fc1 = nn.Linear(1280, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        fully_connected_first_out = F.relu(self.fc1(x))
        fully_connected_second_out = F.relu(self.fc2(fully_connected_first_out))
        two_way_output = self.fc3(fully_connected_second_out)
        return two_way_output


class MLP_head_RegNet_y_1_6gf(nn.Module):
    def __init__(self):
        super(MLP_head_RegNet_y_1_6gf, self).__init__()
        self.fc1 = nn.Linear(888, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        fully_connected_first_out = F.relu(self.fc1(x))
        fully_connected_second_out = F.relu(self.fc2(fully_connected_first_out))
        two_way_output = self.fc3(fully_connected_second_out)
        return two_way_output


class MLP_head_RegNet_y_3_2gf(nn.Module):
    def __init__(self):
        super(MLP_head_RegNet_y_3_2gf, self).__init__()
        self.fc1 = nn.Linear(1512, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        fully_connected_first_out = F.relu(self.fc1(x))
        fully_connected_second_out = F.relu(self.fc2(fully_connected_first_out))
        two_way_output = self.fc3(fully_connected_second_out)
        return two_way_output


def get_based_model_temp(model_name: str) -> nn.Module:
    if model_name == "GoogleNet":
        # Building a GoogleNet pre-trained and hold it as `my_model`
        my_model = torchvision.models.googlenet(pretrained=True)
        # Overriding `my_model`'s fc attribute with the binary classification head stated in the exercise.
        my_model.fc = MLP_head_googleNet()
    elif model_name == "AlexNet":
        # Building a AlexNet pre-trained and hold it as `my_model`
        my_model = torchvision.models.alexnet(pretrained=True)
        # Overriding `my_model`'s fc attribute with the binary classification head stated in the exercise.
        my_model.classifier = MLP_head_AlexNet()
    elif model_name == "ResNet18":
        # Building a ResNet18 pre-trained and hold it as `my_model`
        my_model = torchvision.models.resnet18(pretrained=True)
        # Overriding `my_model`'s fc attribute with the binary classification head stated in the exercise.
        my_model.fc = MLP_head_ResNet18()
    elif model_name == 'MobileNet':
        # Building a MobileNet pre-trained and hold it as `my_model`
        my_model = torchvision.models.mobilenet_v2(pretrained=True)
        # Overriding `my_model`'s fc attribute with the binary classification head stated in the exercise.
        my_model.classifier = MLP_head_MobileNet(my_model.last_channel)
    elif model_name == 'efficientnet_b4':
        # Building a efficientnet_b4 pre-trained and hold it as `my_model`
        my_model = torchvision.models.efficientnet_b4(pretrained=True)
        my_model.classifier = MLP_head_efficientnet_b4()
    elif model_name == 'efficientnet_b3':
        # Building a efficientnet_b3 pre-trained and hold it as `my_model`
        my_model = torchvision.models.efficientnet_b3(pretrained=True)
        my_model.classifier = MLP_head_efficientnet_b3()
    elif model_name == 'efficientnet_b2':
        # Building a efficientnet_b2 pre-trained and hold it as `my_model`
        my_model = torchvision.models.efficientnet_b2(pretrained=True)
        my_model.classifier = MLP_head_efficientnet_b2()
    elif model_name == 'efficientnet_b1':
        # Building a efficientnet_b1 pre-trained and hold it as `my_model`
        my_model = torchvision.models.efficientnet_b1(pretrained=True)
        my_model.classifier = MLP_head_efficientnet_b1()
    elif model_name == 'RegNet_y_1_6gf':
        # Building a RegNet_y_1_6gf pre-trained and hold it as `my_model`
        my_model = torchvision.models.regnet_y_1_6gf(pretrained=True)
        my_model.fc = MLP_head_RegNet_y_1_6gf()
    elif model_name == 'RegNet_y_3_2gf':
        # Building a RegNet_y_3_2gf pre-trained and hold it as `my_model`
        my_model = torchvision.models.regnet_y_3_2gf(pretrained=True)
        my_model.fc = MLP_head_RegNet_y_3_2gf()
    return my_model