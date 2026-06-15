import pennylane as qml

N_QUBITS = 12

dev = qml.device(
    "default.qubit",
    wires=N_QUBITS
)


def qconv_block(weights, wires):

    qml.RY(weights[0], wires=wires[0])
    qml.RY(weights[1], wires=wires[1])

    qml.CNOT(wires=wires)


@qml.qnode(dev, interface="torch")
def qcnn_circuit(
    inputs,
    conv1,
    conv2,
    ent
):

    qml.AngleEmbedding(
        inputs,
        wires=range(N_QUBITS)
    )

    for i in range(0, N_QUBITS, 2):
        qconv_block(
            conv1[0],
            [i, i + 1]
        )

    active = [
        i for i in range(
            0,
            N_QUBITS,
            2
        )
    ]

    for i in range(0, len(active), 2):

        qconv_block(
            conv2[0],
            [active[i], active[i + 1]]
        )

    pooled = active[::2]

    qml.templates.StronglyEntanglingLayers(
        ent,
        wires=pooled
    )

    return [
        qml.expval(qml.PauliZ(w))
        for w in pooled
    ]


weight_shapes = {
    "conv1": (1, 2),
    "conv2": (1, 2),
    "ent": (2, N_QUBITS // 4, 3)
}