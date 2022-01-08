import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
from skimage import transform as tf
import cv2
from metrics import StreamSegMetrics # TODO add new
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1

    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        #jc = metric.binary.jc(pred, gt)
        #asd = metric.binary.asd(pred, gt)
        #assd = metric.binary.asd(pred, gt)
        #precision = metric.binary.precision(pred, gt)
        #recall = metric.binary.recall(pred, gt)
        #sensitivity = metric.binary.sensitivity(pred, gt)
        #specificity = metric.binary.specificity(pred, gt)
        #print (iou_mean(pred, gt)) # TODO delete later
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0

def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1,
                       dataset_name='others'):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    input_shape= 3 if dataset_name == 'Synapse' else 2 # TODO len(image.shape) == 3 if data is synapse else 2
    if len(image.shape) == input_shape:
        prediction = np.zeros_like(label)
        #image = image.transpose(2, 0, 1)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().to(device) # cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred[ind]
    else:
        #input = torch.from_numpy(image).unsqueeze(
        #    0).unsqueeze(0).float().to(device) # cuda()
        # TODO add new
        input_w, input_h, input_c=image.shape
        input = tf.resize(image, (patch_size[0], patch_size[1], 3), order=3)
        input = input.transpose(2, 0, 1)
        input = torch.from_numpy(input).unsqueeze(
            0).float().to(device)
        # TODO add new

        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()

            # TODO add new
            if prediction.shape[0] != input_w or prediction.shape[1].cpu() != input_h:
                prediction = zoom(prediction, ( input_w/prediction.shape[0], input_h/prediction.shape[1]), order=0)

    metrics = StreamSegMetrics(classes) if dataset_name != 'Synapse' else None# TODO add new
    metrics.update(prediction, label) if dataset_name != 'Synapse' else None# TODO add new
            # cv2.imwrite(test_save_path + '/' + case + "_pred.jpg", prediction)
    score = metrics.get_results() if dataset_name != 'Synapse' else []

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz") if dataset_name=='Synapse' else cv2.imwrite(test_save_path + '/' + case + "_pred.jpg", prediction)
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz") if dataset_name=='Synapse' else cv2.imwrite(test_save_path + '/' + case + "_img.jpg", image)
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz") if dataset_name=='Synapse' else cv2.imwrite(test_save_path + '/' + case + "_gt.jpg", label)
    return metric_list, score