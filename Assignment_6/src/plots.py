import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch


class Plots:
    def __init__(self):
        pass

    def sampleVisual(dataset):
        batch = next(iter(dataset))
        images, labels = batch
        batch_grid = make_grid(images)
        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)
        return plt.imshow(batch_grid[0].squeeze(), cmap='gray_r')

    def plotting(model, loader, device):
        wrong_images = []
        wrong_label = []
        correct_label = []

        with torch.no_grad():
            for img, label in loader:
                img, label = img.to(device), label.to(device)
                pred_label = model(img.to(device))
                pred = pred_label.argmax(dim=1, keepdim=True)

                wrong_pred = (pred.eq(target.view_as(pred)) == False)
                wrong_images.append(data[wrong_pred])
                wrong_label.append(pred[wrong_pred])
                correct_label.append(target.view_as(pred)[wrong_pred])

                wrong_predictions = list(
                    zip(torch.cat(wrong_images), torch.cat(wrong_label), torch.cat(correct_label)))
                fig = plt.figure(figsize=(8, 10))
                fig.tight_layout()
                for i, (img, pred, correct) in enumerate(wrong_predictions[:10]):
                    img, pred, target = img.cpu().numpy(), pred.cpu(), correct.cpu()
                    ax = fig.add_subplot(5, 2, i+1)
                    ax.axis('off')
                    ax.set_title(
                        f'\nactual {target.item()}\npredicted {pred.item()}', fontsize=10)
                    ax.imshow(img.squeeze(), cmap='gray_r')

                plt.show()
            return len(wrong_predictions)
