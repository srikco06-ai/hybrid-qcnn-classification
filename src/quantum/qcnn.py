import torch
import torch.nn as nn

from pennylane import qnn

from .circuits import (
    qcnn_circuit,
    weight_shapes,
    N_QUBITS
)

qcnn_layer = qnn.TorchLayer(
    qcnn_circuit,
    weight_shapes
)


class QCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc_in = nn.Linear(
            784,
            N_QUBITS
        )

        self.qcnn = qcnn_layer

        self.fc_out = nn.Linear(
            N_QUBITS // 4,
            10
        )

    def forward(self, x):

        x = x.view(
            x.size(0),
            -1
        )

        x = torch.tanh(
            self.fc_in(x)
        )

        x = self.qcnn(x)

        x = torch.tanh(x)

        return self.fc_out(x)