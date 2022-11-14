import torch
from .model import Net
from pathlib import Path

actions = ['kick', 'punch', 'squat', 'stand', 'attention', 'cancel', 'walk', 'Sit', 'Direction', 'PhoneCall']


def init_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net()
    model.to(device)
    model.load_state_dict(torch.load(Path(Path(__file__).parent.absolute(), "model_best.pth")))
    model.eval()
    return model



def detect(model, input_tensor):
    output = model(input_tensor)
    # pred = output.data.max(1, keepdim=True)[1] 
    pred = output.data.max(0, keepdim=True)[1]
    # print(pred)
    return actions[pred]
