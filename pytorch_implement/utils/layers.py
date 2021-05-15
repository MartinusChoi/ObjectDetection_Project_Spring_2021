"""
define layer classes what use in 'create_modules'

"""
# itertools 자신만의 반복자를 만들게 해주는 모듈
from itertools import chain


import torch
import torch.nn as nn
import torch.nn.functional as F

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class Upsample(nn.Module):
    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x

class DetectionLayer(nn.Module):
    def __init__(self, anchors, num_classes):
        super(DetectionLayer, self).__init__()
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.no = num_classes + 5 # number of outputs per anchor
        self.grid = torch.zeros(1)

        # *anchors -> anchors를 unpacking 하여 값을 풀어서 전달.
        # register_buffer -> pytorch에서 parameter가 아닌 하나의 상태로써 사용하기 위한 것.
        # ==> optimizer가 업데이트 하지 않음. 하지만 값은 존재(하나의 layer로써 작용, 학습되지 않는 layer)
        # ==> state_dict()로 확인 가능
        # ==> GPU 연산 가능
        anchors = torch.tensor(list(chain(*anchors))).float().view(-1, 2)
        self.register_buffer('anchors', anchors)
        self.register_buffer(
            'anchor_grid', anchors.clone().view(1, -1, 1, 1, 2)
        )
        self.stride = None
    
    def forward(self, x, img_size):
        stride = img_size // x.size(2)
        self.stride = stride
        bs, _, ny, nx = x.shape # x(batch_size, 255, 20, 20) to x(batch_size, 3, 20, 20, 85)
        # 차원 순서 바꿀 때 permute 사용 => permute(1, 0, 2).contiguous() 와 같이 붙여서 사용
        # view => 붙어있는 차원을 떼어낼 때 사용 => [B*2, C, D] --> [B, 2, C, D]
        # 1 인 차원 생성/제거할 때는 unsqueeze, squeeze
        x = x.view(bs, self.num_anchors, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

        if not self.training : # inference
            if self.grid.shape[2:4] != x.shape[2:4]:
                self.grid = self._make_grid(nx, ny).to(x.device)
            
            # pytorch tensor -> ... 로 슬라이싱 == : 처럼 모든 값을 가져오라는 의미
            x[..., 0:2] = (x[..., 0:2].sigmoid() + self.grid) * stride  # xy
            x[..., 2:4] = torch.exp(x[..., 2:4]) * self.anchor_grid # wh
            x[..., 4:] = x[..., 4:].sigmoid()
            x = x.view(bs, -1, self.no)
        
        return x
    
    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(ny)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()