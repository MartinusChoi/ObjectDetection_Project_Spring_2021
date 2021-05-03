"""
Darknet : YOLO 구조의 이름

YOLO 네트워크를 구현하는 코드를 작성
"""

# enable use python 3 code in python 2
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from util import *

def parse_cfg(cfgfile) :
    """
    Configuration 파일을 인자로 받음.

    Blocks 리스트를 리턴함.
    각 block은 neural network를 어떻게 빌드하는지에 대해 나타냄.
    Blocks는 dictionary들의 리스트임.

    함수의 목적 : cfg를 파싱 => 모든 block을 dict 형식으로 저장하는 것.

    Block들의 attribute들과 value들은 dictionary에 key-value 형식으로 저장됨.

    code : cfg 파싱 => 이러한 dicts들을 block 변수에 저장 => blocks라는 list에 dicts들을 append.

    return : list
    """

    file = open(cfgfile, 'r') # file open
    lines = file.read().split('\n') # 개행(line) 기준으로 나누어 list로 저장
    lines = [line for line in lines if len(line) > 0] # 비어있지 않은 line들만 추출 (빈 line 제거)
    lines = [line for line in lines if line[0] != '#'] # 주석이 아닌 line들만 추출 (주석 line 제거)
    lines = [line.rstrip().lstrip() for line in lines] # white space 제거

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":      # new block start
            if len(block) != 0:     # block is not empty => append it next the previous block
                blocks.append(block)        # append to block list
                block = {}      # empty block
            block["type"] = line[1:-1].rstrip()         # 대괄호를 제외한 block의 이름을 dict로 생성
            # ie : block = {"type" : 'net'}
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()
            # ie : block = {'batch' : 64, 'subdivisions'=16, ...}
    blocks.append(block)
    # ie : blocks = [{"type" : 'net', 'batch' : 64, ...}, {"type" : 'convolution', 'filters' : 32, ...}]

    return blocks

def create_modules(blocks):
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

    net_info = blocks[0] # 첫 번째 block의 pre-processing(전처리)와 입력에 관한 정보 저장
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []

    for index, block in enumerate(blocks[1:]):
        module = nn.Sequential()

        # block의 type 확인
        # block에 대한 new module 생성
        # 생성한 module을 module_list에 append

        # If it is convolutional layer
        if (block["type"] == "convolutional"):
            # Get the info about the layer
            activation = block["activation"]

            try:
                batch_normalize = int(block["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True
            
            filters = int(block["filters"])
            padding = int(block["pad"])
            kernel_size = int(block["size"])
            stride = int(block["stride"])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0
            
            # Add the convolutional layer
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            module.add_module("conv_{0}".format(index), conv)

            # Add the Batch Norm layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)
            
            # Check the activation
            # YOLO는 Linear 혹은 Leaky ReLU를 사용함.
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace = True)
                module.add_module("leaky_{0}".format(index), activn)
            
        #If it's an upsampling layer
        #We use Bilinear2dUpsampling
        elif (block["type"] == "upsample"):
            stride = int(block["stride"])
            upsample = nn.Upsample(scale_factor = 2, mode = "bilinear")
            module.add_module("upsample_{}".format(index), upsample)
        
        # If it is a Route layer
        elif (block["type"] == "route"):
            block["layers"] = block["layers"].split(',')

            # Start of a route
            start = int(block["layers"][0]) # 몇 index 앞에서 feature map을 가져올 지 에 대한 값
            # end, if there exists one.
            try:
                end = int(block["layers"][1])
            except:
                end = 0
            
            # Positive anotation (양수 값이 넘어온 경우 앞에서 부터 index 탐색)
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index
            
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)

            # route layer에서 나온 filter의 수를 담도록 filters 변수를 업데이트
            # shortcut layer도 간단한 작업(addition)을 수행하기에 empty layer를 사용하지만
            # 이전 layer의 feature map에 바로 뒤 layer의 feature map을 더하는 작업을 하기 때문에
            # filters 변수를 업데이트 할 필요가 없음. => 몇 layer 전이 아닌 단지 이전만 addition
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]
        
        # shortcut corresponds to skip connection
        elif (block["type"] == "shortcut") :
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)
        
        # YOLO is the detection layer
        elif (block["type"] == "yolo"):
            mask = block["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = block["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
            anchors = [anchors[i] for i in mask]

            # Bounding box를 detect하는데 사용되는 anchors들을 가지고 있는 DetectionLayer
            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)
        
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return (net_info, module_list)

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)
    
    def forward(self, x, CUDA):
        modules = self.blocks[1:] # blocks[0] -> net_info
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
                x = outputs[i-1] + outputs[i+from_]
            
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
            
def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (416,416))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W 
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    return img_

model = Darknet("cfg/yolov3.cfg")
inp = get_test_input()
pred = model(inp, torch.cuda.is_available())
print (pred)