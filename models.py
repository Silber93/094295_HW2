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


class MaskModel1(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'mask_1'
        self.cnn_block_1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=(1, 1), stride=(2, 2)),
            # nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        self.cnn_block_2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(1, 1), stride=(2, 2)),
            # nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.cnn_block_3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(1, 1), stride=(2, 2)),
            # nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.cnn_block_4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1), stride=(2, 2)),
            # nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.cnn_block_5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
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
        return loss(y_pred, y_true.float())


class PlainNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(2, 2)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        )

        self.cnn2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        )
        self.cnn3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1)),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=(1, 1)),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1)),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        )

        self.cnn4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1)),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=(1, 1)),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1)),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=(1, 1)),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1)),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=(1, 1)),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1)),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=(1, 1)),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(1, 1)),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3), padding=(1, 1)),

            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), stride=(2, 2))
        )

        self.cnn5 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(1, 1)),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3), padding=(1, 1)),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(1, 1)),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3), padding=(1, 1)),

            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), padding=(1, 1)),

            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        )

        self.cnn6 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), padding=(1, 1)),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), padding=(1, 1)),
            nn.AvgPool2d(kernel_size=(2, 2))
        )

        self.fc = nn.Sequential(
            nn.Linear(1024, 1000),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = self.cnn5(x)
        x = self.cnn6(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class PlainNetBox(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'box_plainnet'
        self.base_model = PlainNet()

        self.lin_bbox = nn.Sequential(
            nn.Linear(1000, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.ReLU()
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.lin_bbox(x)
        x = torch.min(x, torch.Tensor([224]).cuda())
        return x

    @ staticmethod
    def loss_func(bbox_real, bbox_pred):
        bbox_loss = nn.MSELoss()
        return bbox_loss(bbox_pred.squeeze(1), bbox_real)


class PlainNetMask(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'mask_plainnet'
        self.base_model = PlainNet()

        self.lin_bbox = nn.Sequential(
            nn.Linear(1000, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.lin_bbox(x)
        return x

    @ staticmethod
    def loss_func(mask_real, mask_pred):
        mask_loss = nn.BCELoss()
        return mask_loss(mask_pred, mask_real)


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'box_resnet'
        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(2, 2)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        )

        self.res_1_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1)),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=(1, 1)),
        )

        self.res_1_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1)),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=(1, 1)),
        )

        self.res_1_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1)),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=(1, 1)),
        )

        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.res_2_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=(1, 1)),
        )

        self.res_2_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=(1, 1)),
        )

        self.res_2_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=(1, 1)),
        )

        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.res_3_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1)),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=(1, 1)),
        )

        self.res_3_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1)),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=(1, 1)),
        )

        self.res_3_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1)),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=(1, 1)),
        )

        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.res_4_1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1)),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=(1, 1)),
        )

        self.res_4_2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(1, 1)),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=(1, 1)),
        )

        self.res_4_3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(1, 1)),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=(1, 1)),
        )

        self.pool4 = nn.AvgPool2d(kernel_size=(2, 2), stride=2)

        self.fc = nn.Sequential(
            nn.Linear(4608, 1000),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.cnn1(x)
        x = x + self.res_1_1(x)
        x = x + self.res_1_2(x)
        x = x + self.res_1_3(x)
        x = self.pool1(x)
        x = self.res_2_1(x)
        x = x + self.res_2_2(x)
        x = x + self.res_2_3(x)
        x = self.pool2(x)
        x = self.res_3_1(x)
        x = x + self.res_3_2(x)
        x = x + self.res_3_3(x)
        x = self.pool3(x)
        x = self.res_4_1(x)
        x = x + self.res_4_2(x)
        x = x + self.res_4_3(x)
        x = self.pool4(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ResNetBox(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'box_resnet'
        self.base_model = ResNet()

        self.lin_bbox = nn.Sequential(
            nn.Linear(1000, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.ReLU()
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.lin_bbox(x)
        x = torch.min(x, torch.Tensor([224]).cuda())
        return x

    @ staticmethod
    def loss_func(bbox_real, bbox_pred):
        bbox_loss = nn.MSELoss()
        return bbox_loss(bbox_pred, bbox_real)


class ResNetMask(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'mask_resnet'
        self.base_model = ResNet()

        self.lin_mask = nn.Sequential(
            nn.Linear(1000, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.lin_mask(x)
        return x

    @ staticmethod
    def loss_func(mask_real, mask_pred):
        mask_loss = nn.BCELoss()
        return mask_loss(mask_pred, mask_real)


class ResNetCombined(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'combined_resnet'
        self.base_model = ResNet()

        self.lin_bbox = nn.Sequential(
            nn.Linear(1000, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Sigmoid()
        )

        self.lin_mask = nn.Sequential(
            nn.Linear(1000, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.base_model(x)
        mask = self.lin_mask(x)
        bbox = self.lin_bbox(x)
        return bbox, mask

    @ staticmethod
    def loss_func(mask_real, mask_pred, bbox_real, bbox_pred):
        mask_loss = nn.BCELoss()(mask_pred, mask_real)
        bbox_loss = nn.BCELoss()(bbox_pred, bbox_real)
        return mask_loss + bbox_loss