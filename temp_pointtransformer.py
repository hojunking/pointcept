import torch
import torch.optim as optim

import torch
import torch.nn as nn
import pointops
import einops
import os
if __name__ == '__main__':
    #os.sys.path.append('./pytorch_geometric/torch_geometric')
    os.sys.path.append('./pointcept/models')
from builder import MODELS
from point_transformer.utils import LayerNorm1d

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TORCH_USE_CUDA_DSA"] = '1'

class PointTransformer(nn.Module):
    def __init__(self, block, blocks, in_channels=6, out_channels=1024):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_planes, planes = in_channels, [32, 64, 128, 256, out_channels]
        fpn_planes, fpnhead_planes, share_planes = 128, 64, 8
        stride, nsample = [1, 4, 4, 4, 4], [8, 16, 16, 16, 16]
        self.enc1 = self._make_enc(block, planes[0], blocks[0], share_planes, stride=stride[0], nsample=nsample[0])
        self.enc2 = self._make_enc(block, planes[1], blocks[1], share_planes, stride=stride[1], nsample=nsample[1])
        self.enc3 = self._make_enc(block, planes[2], blocks[2], share_planes, stride=stride[2], nsample=nsample[2])
        self.enc4 = self._make_enc(block, planes[3], blocks[3], share_planes, stride=stride[3], nsample=nsample[3])
        self.enc5 = self._make_enc(block, planes[4], blocks[4], share_planes, stride=stride[4], nsample=nsample[4])

    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
        layers = [TransitionDown(self.in_planes, planes * block.expansion, stride, nsample)]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def forward(self, x):
        B, N, _ = x.size()
        coord = x[:, :, :3]
        feat = x[:, :, 3:]
        x0 = torch.cat((coord, feat), dim=2)

        # Flatten the input for processing
        x0 = einops.rearrange(x0, 'b n c -> (b n) c')
        p0 = einops.rearrange(coord, 'b n c -> (b n) c')
        o0 = torch.arange(0, B * N + 1, N).to(x.device).int()

        p1, x1, o1 = self.enc1([p0, x0, o0])
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])
        p5, x5, o5 = self.enc5([p4, x4, o4])

        x = []
        for i in range(o5.shape[0]):
            if i == 0:
                s_i, e_i, cnt = 0, o5[0], o5[0]
            else:
                s_i, e_i, cnt = o5[i - 1], o5[i], o5[i] - o5[i - 1]
            x_b = x5[s_i:e_i, :].sum(0, True) / cnt
            x.append(x_b)
        x = torch.cat(x, 0)
        return x

class PointTransformerLayer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        self.linear_p = nn.Sequential(
            nn.Linear(3, 3),
            LayerNorm1d(3),
            nn.ReLU(inplace=True),
            nn.Linear(3, out_planes),
        )
        self.linear_w = nn.Sequential(
            LayerNorm1d(mid_planes),
            nn.ReLU(inplace=True),
            nn.Linear(mid_planes, out_planes // share_planes),
            LayerNorm1d(out_planes // share_planes),
            nn.ReLU(inplace=True),
            nn.Linear(out_planes // share_planes, out_planes // share_planes),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pxo) -> torch.Tensor:
        p, x, o = pxo  # (n, 3), (n, c), (b)
        x_q, x_k, x_v = self.linear_q(x), self.linear_k(x), self.linear_v(x)
        x_k, idx = pointops.knn_query_and_group(x_k, p, o, new_xyz=p, new_offset=o, nsample=self.nsample, with_xyz=True)
        x_v, _ = pointops.knn_query_and_group(x_v, p, o, new_xyz=p, new_offset=o, idx=idx, nsample=self.nsample, with_xyz=False)
        p_r, x_k = x_k[:, :, 0:3], x_k[:, :, 3:]
        p_r = self.linear_p(p_r)
        r_qk = x_k - x_q.unsqueeze(1) + einops.reduce(p_r, "n ns (i j) -> n ns j", reduction="sum", j=self.mid_planes)
        w = self.linear_w(r_qk)  # (n, nsample, c)
        w = self.softmax(w)
        x = torch.einsum(
            "n t s i, n t i -> n s i",
            einops.rearrange(x_v + p_r, "n ns (s i) -> n ns s i", s=self.share_planes),
            w,
        )
        x = einops.rearrange(x, "n s i -> n (s i)")
        return x

class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, share_planes=8, nsample=16):
        super(Bottleneck, self).__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.transformer = PointTransformerLayer(planes, planes, share_planes, nsample)
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        identity = x
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.transformer([p, x, o])))
        x = self.bn3(self.linear3(x))
        x += identity
        x = self.relu(x)
        return [p, x, o]
    
class TransitionDown(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, nsample=16):
        super().__init__()
        self.stride, self.nsample = stride, nsample
        if stride != 1:
            self.linear = nn.Linear(3 + in_planes, out_planes, bias=False)
            self.pool = nn.MaxPool1d(nsample)
        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        if self.stride != 1:
            n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
            for i in range(1, o.shape[0]):
                count += (o[i].item() - o[i - 1].item()) // self.stride
                n_o.append(count)
            n_o = torch.cuda.IntTensor(n_o)
            idx = pointops.farthest_point_sampling(p, o, n_o)  # (m)
            n_p = p[idx.long(), :]  # (m, 3)
            x, _ = pointops.knn_query_and_group(x, p, offset=o, new_xyz=n_p, new_offset=n_o, nsample=self.nsample, with_xyz=True)
            x = self.relu(self.bn(self.linear(x).transpose(1, 2).contiguous()))  # (m, c, nsample)
            x = self.pool(x).squeeze(-1)  # (m, c)
            p, o = n_p, n_o
        else:
            x = self.relu(self.bn(self.linear(x)))  # (n, c)
        return [p, x, o]

# #@MODELS.register_module("PointTransformer-Cls26")
# class PointTransformerCls26(PointTransformer):
#     def __init__(self, **kwargs):
#         super(PointTransformerCls26, self).__init__(Bottleneck, [1, 1, 1, 1, 1], **kwargs)

# #@MODELS.register_module("PointTransformer-Cls38")
# class PointTransformerCls38(PointTransformer):
#     def __init__(self, **kwargs):
#         super(PointTransformerCls38, self).__init__(Bottleneck, [1, 2, 2, 2, 2], **kwargs)

# #@MODELS.register_module("PointTransformer-Cls50")
# class PointTransformerCls50(PointTransformer):
#     def __init__(self, **kwargs):
#         super(PointTransformerCls50, self).__init__(Bottleneck, [1, 2, 3, 5, 2], **kwargs)

# 모델 초기화
def initialize_point_transformer():
    in_channels = 3  # 입력 포인트 차원 (x, y, z)
    out_channels = 1024  # 최종 출력 특징 벡터 차원
    num_blocks = [1, 2, 3, 5, 2]  # 각 레벨의 블록 수

    model = PointTransformer(Bottleneck, num_blocks, in_channels=in_channels, out_channels=out_channels)
    return model

# 모델 학습 설정
def setup_training(model):
    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    return optimizer, criterion

def main():
    # 모델 초기화
    model = initialize_point_transformer().to('cuda')

    # 학습 설정 초기화
    optimizer, criterion = setup_training(model)

    # 더미 입력 데이터 생성 (예: 배치 크기 32, 포인트 수 1024, 차원 3)
    batch_size = 32
    num_points = 1024
    input_data = torch.tensor(torch.randn(batch_size, num_points, 3), dtype=torch.float32, device='cuda') 

    # 모델 출력
    output = model(input_data)

    print(f"Output shape: {output.shape}")


if __name__ == '__main__':
    main()