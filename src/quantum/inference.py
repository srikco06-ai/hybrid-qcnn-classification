import torch


def predict(model, image, device="cpu"):

    model.eval()

    image = image.to(device)

    with torch.no_grad():

        output = model(image)

        prediction = torch.argmax(
            output,
            dim=1
        )

    return prediction.item()


def predict_proba(model, image, device="cpu"):

    model.eval()

    image = image.to(device)

    with torch.no_grad():

        output = model(image)

        probabilities = torch.softmax(
            output,
            dim=1
        )

    return probabilities.cpu().numpy()[0]