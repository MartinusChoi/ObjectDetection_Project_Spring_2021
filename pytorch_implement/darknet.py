"""
Darknet : YOLO 구조의 이름

YOLO 네트워크를 구현하는 코드를 작성
"""

"""
batch의 각 이미지당 10647 x 85 크기의 테이블을 가짐

10674 = 테이블의 행 => bounding box를 나타냄

85 = (4 bbox attributes, 1 object score, 80 class score)
"""
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
import numpy as np

import cv2

from utils.utils import predict_transform, weights_init_normal
from utils.parse_config import parse_cfg
from utils.layers import EmptyLayer, DetectionLayer, Upsample

def create_modules(module_defs):
    """
    return : nn.ModuleList

    nn.ModuleList : nn.Module object를 담은 일반 리스트에 거의 흡사
    -> 다만 nn.Module object를 nn.ModuleList에 추가할 때,
    -> nn.ModuleList 안에 있는 nn.Module object(modules)들의 모든 parameter들이 
    -> nn.Module object의 parameter로 추가됨.

    """

    """
    새로운 convolutional layer를 정의할 때, 반드시 kernel의 dimension을 정의해야함.

    cfg파일에서 kernel의 width와 height의 정보를 알 수 있고,

    kernel의 깊이 (feature map의 깊이)는 이전 layer의 filter의 수에서 얻을 수 있음.
    => convolutional layer가 적용되는 layer의 filter 수를 추적해야함을 의미
    => prev_filter 변수로 해당 작업 진행

    이미지가 RGB channel에 대응되는 3개의 filter를 가지기에 default 값이 3임.
    """

    """
    Route layer : 이전 layer들에서 feature map을 가져옴.(concat되었을 가능성 높음)

    Route layer 바로 앞에 convoulutional layer가 있는 경우 kernel은
    route layer가 가지고 온 이전 layer들의 feature map들에 적용됨.

    따라서, 이전 layer의 filter 수 뿐만 아니라 route layer로 부터 온 feature map의 filter수도 파악해야함.
    
    iterate 마다 각 block의 output filter 수를 output_filters 리스트에 append.
    """

    """
    EmptyLayer

    다른 layer들 처럼 작업을 수행함.
    => previous layer 앞으로 가져오기 / concatenation

    PyTorch에서 새로운 layer를 정의할 때 nn.Module을 subclass하고,
    nn.Module object의 foward 함수에 layer가 수행하는 작업들을 작성함.

    Route block을 위한 layer를 디자인하기 위해, 
    멤버로서 attribute layers의 값으로 초기화된 nn.Module object를 만들어야함.
    => 그 다음 foward 함수에 concatenate/feature map 앞으로 가져오는 작업에 대한 코드를 작성할 수 있다.
    => 마지막으로, 네트워크의 foward 함수에서 해당 layer를 실행함.

    하지만 주어진 concatenate된 코드는 간결함 => feature map에 torch.cat 호출

    위와 같이 layer를 디자인하는 것은 불필요한 추상화 발생 => boiler plate code만 증가시킴.

    하지만 route layer 자리에 dummy layer를 두게 되면 
    darknet을 나타내는 nn.Module object의 forward 함수에서 바로 concatenate 할 수 있게 됨.

    """

    net_info = module_defs[0] # 첫 번째 block의 pre-processing(전처리)와 입력에 관한 정보 저장(dictionary)
    
    net_info.update({
        'batch' : int(net_info['batch']),
        'subdivisions' : int(net_info['subdivisions']),
        'width' : int(net_info['width']),
        'height' : int(net_info['height']),
        'channels' : int(net_info['channels']),
        'optimizer' : net_info.get('optimizer'),
        'momentum' : float(net_info['momentum']),
        'decay' : float(net_info['decay']),
        'learning_rate' : float(net_info['learning_rate']),
        'burn_in' : int(net_info['burn_in']),
        'max_batches' : int(net_info['max_batches']),
        'policy' : net_info['policy'],
        'lr_steps' : list(zip(map(int, net_info['steps'].split(',')), \
            map(float, net_info['scales'].split(','))))
    })

    # assert -> 값을 보증 하기 위해 사용 ; height와 width의 크기가 같아야 하기에 이를 보장하기 위한 방어적 프로그래밍 기법
    # python '\' -> 구문이 길어질때 사용하면 구문이 이어진다는 것을 의미함.
    assert net_info['height'] == net_info['width'], \
        "Invalid size, Height and Width should be same! (Non square images are padded with zeros.)"
    
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []

    for index, module_def in enumerate(module_defs[1:]):
        module = nn.Sequential()

        # block의 type 확인
        # block에 대한 new module 생성
        # 생성한 module을 module_list에 append

        # If it is convolutional layer
        if module_def["type"] == "convolutional":
            # Get the info about the layer
            activation = module_def["activation"]

            try:
                batch_normalize = int(module_def["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True
            
            filters = int(module_def["filters"])
            padding = int(module_def["pad"])
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0
            
            # Add the convolutional layer
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            module.add_module(f"conv_{index}", conv)

            # Add the Batch Norm layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module(f"batch_norm_{index}", bn)
            
            # Check the activation
            # YOLO는 Linear 혹은 Leaky ReLU를 사용함.
            # Linear는 선형 함수이므로 따로 activation module이 없는 것과 같음.
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace = True)
                module.add_module(f"leaky_{index}", activn)
            
        # If it's an upsampling layer
        # We use Bilinear2dUpsampling
        elif module_def["type"] == "upsample":
            stride = int(module_def["stride"])
            upsample = Upsample(scale_factor = stride, mode = "nearest")
            module.add_module(f"upsample_{index}", upsample)
        
        # If it is a Route layer
        elif module_def["type"] == "route":
            module_def["layers"] = module_def["layers"].split(',')

            # Start of a route
            start = int(module_def["layers"][0]) # 몇 index 앞에서 feature map을 가져올 지 에 대한 값
            # end, if there exists one.
            try:
                end = int(module_def["layers"][1])
            except:
                end = 0
            
            # Positive anotation (양수 값이 넘어온 경우 앞에서 부터 index 탐색)
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index
            
            route = EmptyLayer()
            module.add_module(f"route_{index}", route)

            # route layer에서 나온 filter의 수를 담도록 filters 변수를 업데이트
            # shortcut layer도 간단한 작업(addition)을 수행하기에 empty layer를 사용하지만
            # 이전 layer의 feature map에 바로 뒤 layer의 feature map을 더하는 작업을 하기 때문에
            # filters 변수를 업데이트 할 필요가 없음. => 몇 layer 전이 아닌 단지 이전만 addition
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]
        
        # shortcut corresponds to skip connection
        elif (module_def["type"] == "shortcut") :
            shortcut = EmptyLayer()
            module.add_module(f"shortcut_{index}", shortcut)
        
        # YOLO is the detection layer
        elif (module_def["type"] == "yolo"):
            anchor_idxs = module_def["mask"].split(",")
            anchor_idxs = [int(x) for x in anchor_idxs]

            anchors = module_def["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
            anchors = [anchors[i] for i in anchor_idxs]

            num_classes = int(module_def['classes'])

            # Bounding box를 detect하는데 사용되는 anchors들을 가지고 있는 DetectionLayer
            detection = DetectionLayer(anchors, num_classes)
            module.add_module(f"Detection_{index}", detection)
        
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return net_info, module_list

class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.module_defs = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.module_defs)
        self.yolo_layers = [layer[0]\
            for layer in self.module_list if isinstance(layer, DetectionLayer)]
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)
    
    def forward(self, x, CUDA):
        modules = self.module_defs[1:] # blocks[0] -> net_info
        outputs = {} # cache all the outputs for the route layer (Key : index, value : feature map)
        
        write = 0
        
        for i, module in enumerate(modules):
            module_type = (module["type"])

            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)
            
            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]

                if (layers[0]) > 0:
                    layers[0] = layers[0] - i
                
                if len(layers) == 1:
                    x = outputs[i + (layers[0])]
                
                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i
                    
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)
            
            elif module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i-1] + outputs[i+from_] # 이전 layer의 output과 from_ layer의 output을 더함 => shortcut
            
            elif module_type == 'yolo':
                anchors = self.module_list[i][0].anchors
                #Get the input dimensions
                inp_dim = int (self.net_info["height"])
        
                #Get the number of classes
                num_classes = int (module["classes"])
        
                #Transform 
                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                if not write:              #if no collector has been intialised. 
                    detections = x
                    write = 1
        
                else:
                    detections = torch.cat((detections, x), 1)
        
            outputs[i] = x
        
        return detections

    def load_darknet_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        with open(weights_path, "rb") as f:
            # First five are header values
            header = np.fromfile(f, dtype=np.int32, count=5)
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        cutoff = None
        if "darknet53.conv.74" in weights_path:
            cutoff = 75

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if i == cutoff:
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(
                    weights[ptr: ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w
    
    def save_darknet_weights(self, path, cutoff=-1):
        """
            @:param path    - path of the new weights file
            @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()

def get_test_input():
    img = cv2.imread("./dog-cycle-car.png")
    img = cv2.resize(img, (416,416))          #Resize to the input dimension (cfg file과 동일하게 맞추기!! net_info)
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W X C -> C X H X W / ::-1 -> 해당 axis에서 순서를 반대로 바꿈(channel의 순서를 뒤집기) 
    img_ = img_[np.newaxis,:,:,:]/255.0       # Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     # Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    return img_

def load_model(model_path, weights_path=None):
    """file로 부터 yolo model을 load

    : param model_path ; model definition file(.cfg) 까지의 경로
    : type model_path : str
    : param weights_path ; weigths 혹은 checkpoint file(.weights or .pth) 파일 까지의 경로
    : type weigth_paht ; str
    : return ; Returns model
    : rtype ; Darknet
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Darknet(model_path).to(device)

    model.apply(weights_init_normal)

    if weights_path:
        if weights_path.endswith(".pth"):
            model.load_state_dict(torch.load(weights_path, map_location=device))
        else:
            model.load_darknet_weights(weights_path)
    
    return model

model = Darknet("./cfg/yolov3.cfg")
inp = get_test_input()
pred = model(x = inp, CUDA = torch.cuda.is_available())
print(pred)
print(pred.shape)