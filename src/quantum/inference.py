import torch


def predict(model, image):

    model.eval()

    with torch.no_grad():

        output = model(image)

        prediction = torch.argmax(
            output,
            dim=1
        )

    return prediction