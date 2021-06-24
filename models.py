import torch
import torch.nn as nn

def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

class BoxModel1(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'box_1'
        self.cnn_block_1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=(1, 1), stride=(2, 2)),
            nn.BatchNorm2d(8)
        )
        self.cnn_block_2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(1, 1), stride=(2, 2)),
            nn.BatchNorm2d(16)
        )
        self.cnn_block_3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(1, 1), stride=(2, 2)),
            nn.BatchNorm2d(32)
        )
        self.cnn_block_4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1), stride=(2, 2)),
            nn.BatchNorm2d(64)
        )
        self.cnn_block_5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2)),
            nn.BatchNorm2d(128)
        )
        self.lin_block_1 = nn.Sequential(
            nn.Linear(128 * 7 * 7, 128),
            nn.ReLU()
        )
        self.lin_block_2 = nn.Sequential(
            nn.Linear(128, 4),
            nn.ReLU()
        )

    def forward(self, x):
        y = self.cnn_block_1(x)
        y = self.cnn_block_2(y)
        y = self.cnn_block_3(y)
        y = self.cnn_block_4(y)
        y = self.cnn_block_5(y)
        y = y.view(y.shape[0], -1)
        y = self.lin_block_1(y)
        y = self.lin_block_2(y)
        return y

    @staticmethod
    def loss_func(y_true, y_pred):
        loss = nn.MSELoss()
        # loss = nn.BCELoss(reduction='sum')
        return loss(y_pred, y_true)

    def loss_area(self, y_true, y_pred):
        y_pred = torch.round(y_pred) + torch.ones(size=y_pred.shape)
        inter_x_start = torch.max(y_true[:, 0], y_pred[:, 0]).unsqueeze(1)
        inter_y_start = torch.max(y_true[:, 1], y_pred[:, 1]).unsqueeze(1)
        inter_start = torch.cat([inter_x_start, inter_y_start], dim=1)
        inter_x_end = torch.min(y_true[:, 0] + y_true[:, 2], y_pred[:, 0] + y_pred[:, 2]).unsqueeze(1)
        inter_y_end = torch.min(y_true[:, 1] + y_true[:, 3], y_pred[:, 1] + y_pred[:, 3]).unsqueeze(1)
        inter_end = torch.cat([inter_x_end, inter_y_end], dim=1)
        inter_dim = torch.max(inter_end - inter_start, torch.zeros(size=inter_end.shape))
        inter_area = inter_dim[:, 0] * inter_dim[:, 1]
        union_area = y_true[:, 2] * y_true[:, 3] + y_pred[:, 2] * y_pred[:, 3] - inter_area
        res = torch.exp(- inter_area / union_area)
        return torch.sum(res) / res.shape[0]

    def score_area(self, y_true, y_pred):
        y_pred = torch.round(y_pred)
        inter_x_start = torch.max(y_true[:, 0], y_pred[:, 0]).unsqueeze(1)
        inter_y_start = torch.max(y_true[:, 1], y_pred[:, 1]).unsqueeze(1)
        inter_start = torch.cat([inter_x_start, inter_y_start], dim=1)
        inter_x_end = torch.min(y_true[:, 0] + y_true[:, 2], y_pred[:, 0] + y_pred[:, 2]).unsqueeze(1)
        inter_y_end = torch.min(y_true[:, 1] + y_true[:, 3], y_pred[:, 1] + y_pred[:, 3]).unsqueeze(1)
        inter_end = torch.cat([inter_x_end, inter_y_end], dim=1)
        inter_dim = torch.max(inter_end - inter_start, torch.zeros(size=inter_end.shape))
        inter_area = inter_dim[:, 0] * inter_dim[:, 1]
        union_area = y_true[:, 2] * y_true[:, 3] + y_pred[:, 2] * y_pred[:, 3] - inter_area
        res = inter_area / union_area
        return torch.sum(res) / res.shape[0]


class BoxModel2(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'box_2'
        self.cnn_block_1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=(1, 1)),
            # nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.cnn_block_2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(1, 1), stride=(2, 2)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.cnn_block_3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(1, 1), stride=(2, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.cnn_block_4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1), stride=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.cnn_block_5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.lin_block_1 = nn.Sequential(
            nn.Linear(128 * 7 * 7, 128),
            nn.ReLU()
        )
        self.lin_block_2 = nn.Sequential(
            nn.Linear(128, 4),
            nn.Sigmoid()
        )
        self.cnn_block_1.apply(init_weights)

    def forward(self, x):
        y = self.cnn_block_1(x)
        y = self.cnn_block_2(y)
        y = self.cnn_block_3(y)
        y = self.cnn_block_4(y)
        y = self.cnn_block_5(y)
        y = y.view(y.shape[0], -1)
        y = self.lin_block_1(y)
        y = self.lin_block_2(y)
        return y

    @staticmethod
    def loss_func(y_true, y_pred):
        w_intersection = torch.min(y_true[:, 0] + y_true[:, 2], y_pred[:, 0] + y_pred[:, 2]) \
                         - torch.max(y_true[:, 0], y_pred[:, 0])
        h_intersection = torch.min(y_true[:, 1] + y_true[:, 3], y_pred[:, 1] + y_pred[:, 3]) \
                         - torch.max(y_true[:, 1], y_pred[:, 1])
        w_intersection = torch.max(torch.zeros(size=w_intersection.shape), w_intersection)
        h_intersection = torch.max(torch.zeros(size=h_intersection.shape), h_intersection)
        inter_area = w_intersection * h_intersection
        union = y_true[:, 2] * y_true[:, 3] + y_pred[:, 2] * y_pred[:, 3] - inter_area
        res = inter_area / union
        iou = torch.sum(res) / res.shape[0]
        # loss = nn.BCELoss(reduction='sum')
        return 1 - iou



        # inter_x_start = torch.max(y_true[:, 0], y_pred[:, 0]).unsqueeze(1)
        # inter_y_start = torch.max(y_true[:, 1], y_pred[:, 1]).unsqueeze(1)
        # inter_start = torch.cat([inter_x_start, inter_y_start], dim=1)
        # inter_x_end = torch.min(y_true[:, 0] + y_true[:, 2], y_pred[:, 0] + y_pred[:, 2]).unsqueeze(1)
        # inter_y_end = torch.min(y_true[:, 1] + y_true[:, 3], y_pred[:, 1] + y_pred[:, 3]).unsqueeze(1)
        # inter_end = torch.cat([inter_x_end, inter_y_end], dim=1)
        # inter_dim = torch.max(inter_end - inter_start, torch.zeros(size=inter_end.shape))
        # inter_area = inter_dim[:, 0] * inter_dim[:, 1]
        # union_area = y_true[:, 2] * y_true[:, 3] + y_pred[:, 2] * y_pred[:, 3] - inter_area
        # res = torch.exp(- inter_area / union_area)
        # return torch.sum(res) / res.shape[0]


class MaskModel1(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'mask'
        self.cnn_block_1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=(1, 1), stride=(2, 2)),
            nn.BatchNorm2d(8)
        )
        self.cnn_block_2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(1, 1), stride=(2, 2)),
            nn.BatchNorm2d(16)
        )
        self.cnn_block_3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(1, 1), stride=(2, 2)),
            nn.BatchNorm2d(32)
        )
        self.cnn_block_4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1), stride=(2, 2)),
            nn.BatchNorm2d(64)
        )
        self.cnn_block_5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2)),
            nn.BatchNorm2d(128)
        )
        self.lin_block_1 = nn.Sequential(
            nn.Linear(128 * 7 * 7, 128),
            nn.ReLU()
        )
        self.lin_block_2 = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.cnn_block_1(x)
        y = self.cnn_block_2(y)
        y = self.cnn_block_3(y)
        y = self.cnn_block_4(y)
        y = self.cnn_block_5(y)
        y = y.view(y.shape[0], -1)
        y = self.lin_block_1(y)
        y = self.lin_block_2(y)
        return y

    @ staticmethod
    def loss_func(y_true, y_pred):
        loss = nn.BCELoss()
        return loss(y_pred, y_true.float().unsqueeze(1))


class MaskModel2(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'mask'
        self.cnn_block_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.cnn_block_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.cnn_block_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        )
        self.cnn_block_4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.cnn_block_5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=1)
        )
        self.lin_block_1 = nn.Sequential(
            nn.Linear(512 * 14 * 14, 128),
            nn.ReLU()
        )
        self.lin_block_2 = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.cnn_block_1(x)
        y = self.cnn_block_2(y)
        y = self.cnn_block_3(y)
        y = self.cnn_block_4(y)
        y = self.cnn_block_5(y)
        y = y.view(y.shape[0], -1)
        y = self.lin_block_1(y)
        y = self.lin_block_2(y)
        return y

    @ staticmethod
    def loss_func(y_true, y_pred):
        loss = nn.BCELoss()
        return loss(y_pred, y_true.float().unsqueeze(1))