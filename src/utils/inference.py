import torch


def predict(model, image):

    model.eval()

    with torch.no_grad():

        output = model(
            image.unsqueeze(0)
        )

        probs = torch.softmax(
            output,
            dim=1
        )

    return probs