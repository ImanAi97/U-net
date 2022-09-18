import torch
import torchvision
from tqdm import tqdm

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device='cpu'
):  
    # evaluation mode:    
    model.eval()
    for idx, (x, y) in enumerate(loader):

        x = x.to(device = device)
        with torch.no_grad():

            # sigmoid for binary class segmentation
            # output of the model is in shape [batch_size, channels, width, height] 
            preds = torch.sigmoid(model(x))

            # binariez the output:   0.0 : background   1.0 : mask (float type becuase we want to calc loss)
            preds = (preds > 0.5).float()
        

        # Save prediction Tensor into an image file.
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    """Save Checkpoints in training 

    Args:
        state (dict): dictionary with (model state_dict) and (optimizer state_dict)
        filename (str, optional): .pth or .pt.tar. Defaults to "my_checkpoint.pth.tar".
    """
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    # only model state_dict
    model.load_state_dict(checkpoint["state_dict"])


def check_accuracy(loader, model, device="cpu"):

    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x)) 
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum() # 2D matrices comparisons elementwise

            # torch.numel ==> Returns the total number of elements in the input tensor.
            num_pixels += torch.numel(preds)

            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.3f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()

def pixel_accuracy(prediction, target):
    preds = torch.sigmoid(prediction)
    preds = (preds > 0.5).float()
    return ((preds == target).sum() / torch.numel(preds)).item()


def dice_accuracy(prediction, target):
    preds = torch.sigmoid(prediction)
    preds = (preds > 0.5).float()

    dice_score = (2 * (preds * target).sum()) / (
                (preds + target).sum() + 1e-8
            )
    return dice_score.item()



from collections import defaultdict

class MetricMonitor():
    def __init__(self, precision = 3):
        self.float_precision = precision
        self.reset()

    def reset(self):
        ## For every key assigned, this lambda function with no param associated.
        self.metrics = defaultdict(lambda: {"value": 0, "count": 0, "avg": 0})


    def update(self, metric_name, value):
        metric = self.metrics[metric_name]

        metric["value"] += value
        metric["count"] += 1
        metric["avg"] = metric["value"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"], float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )
            

def train_epoch(loader, model, optimizer, loss_fn, epoch, DEVICE, scaler):
    metric_monitor = MetricMonitor()
    model.train()
    stream = tqdm(loader)
    # data, targets = next(iter(loader))
    for batch_idx, (data, targets) in enumerate(stream, start = 1):
        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE).float().unsqueeze(1) # [batch, channel, height, width]

        # forward
        # using automatic mixed precision for faster training on NVIDIA GPUs instead of torch.grad()
        if DEVICE == 'cuda':
            with torch.cuda.amp.autocast():
                predictions = model(data)
                loss = loss_fn(predictions, targets)
                pixel_accu = pixel_accuracy(predictions, targets)
                dice_accu = dice_accuracy(predictions, targets)
                metric_monitor.update("Loss", loss.item())
                metric_monitor.update("pixel_accu", pixel_accu)
                metric_monitor.update("dice_accu", dice_accu)

        # backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # stream.set_description(
            #     "Epoch: {epoch}. Train. {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
            # )
        else:
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            pixel_accu = pixel_accuracy(predictions, targets)
            dice_accu = dice_accuracy(predictions, targets)
            metric_monitor.update("Loss", loss.item())
            metric_monitor.update("pixel_accu", pixel_accu)
            metric_monitor.update("dice_accu", dice_accu)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    # update tqdm loop
        stream.set_description(
                "Epoch: {epoch}. Train. {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor))       


def validation_epoch(loader, model, loss_fn ,epoch, DEVICE, scaler = None):
    metric_monitor = MetricMonitor()
    model.eval()
    stream = tqdm(loader)
    # data, targets = next(iter(loader))
    for batch_idx, (data, targets) in enumerate(stream, start = 1):
        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE).float().unsqueeze(1) # [batch, channel, height, width]

        # forward
        # using automatic mixed precision for faster training on NVIDIA GPUs instead of torch.grad()
        predictions = model(data)
        pixel_accu = pixel_accuracy(predictions, targets)
        dice_accu = dice_accuracy(predictions, targets)
        loss = loss_fn(predictions, targets)
        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("pixel_accu", pixel_accu)
        metric_monitor.update("dice_accu", dice_accu)




    # update tqdm loop
        stream.set_description(
                "Epoch: {epoch}. Valid. {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
            )