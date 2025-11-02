import torch
from dataset.dataset_rgbn import MVTecADRGBNDataset
from torch.utils.data import DataLoader
import numpy as np
import random
from torchvision import transforms
from models.encoders import ResNet50Encoder
from models.decoders import ResNet50DualModalDecoder
from utils.losses import *
from evaluation.eval_utils import cal_anomaly_map
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import precision_recall_curve
from evaluation.metrics_utils import calculate_au_pro


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def f1_score_max(y_true, y_score):
    precs, recs, thrs = precision_recall_curve(y_true, y_score)

    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    return f1s.max()


def train_one_epoch(teacher_rgb, teacher_depth, student_rgb, student_depth, train_dataloader, optimizer_rgb, optimizer_depth, device, log, epoch):
    loss_rgb_list = []
    loss_depth_list = []
    for batch_idx, data in enumerate(train_dataloader, 0):
        rgb_image, depth_image, _, _, _ = data
        rgb_image = rgb_image.to(device)
        depth_image = depth_image.to(device)

        with torch.no_grad():
            output_Tr = teacher_rgb(rgb_image)
            output_Td = teacher_depth(depth_image)

            output_Tr_detach = [output_Tr[0].detach(), output_Tr[1].detach(), output_Tr[2].detach()]
            output_Td_detach = [output_Td[0].detach(), output_Td[1].detach(), output_Td[2].detach()]

        proj_d, proj_d_amply, output_Sr, output_Sr_am = student_rgb(output_Tr_detach, output_Td_detach)
        loss_rgb = loss_distil(output_Sr, output_Tr) + loss_distil(proj_d, output_Tr) + loss_distil(proj_d_amply, output_Tr) + loss_distil(output_Sr_am, output_Tr)
        loss_rgb_list.append(loss_rgb.item())
        optimizer_rgb.zero_grad()
        loss_rgb.backward()
        optimizer_rgb.step()

        proj_r, proj_r_amply, output_Sd, output_Sd_am = student_depth(output_Td_detach, output_Tr_detach)
        loss_depth = loss_distil(output_Sd, output_Td) + loss_distil(proj_r, output_Td) + loss_distil(proj_r_amply, output_Td) + loss_distil(output_Sd_am, output_Td)
        loss_depth_list.append(loss_depth.item())
        optimizer_depth.zero_grad()
        loss_depth.backward()
        optimizer_depth.step()

    print('epoch %d, loss_rgb: %.10f, loss_depth: %.10f' % (epoch + 1, np.mean(loss_rgb_list), np.mean(loss_depth_list)))
    print(
        'epoch %d, loss_rgb: %.10f, loss_depth: %.10f' % (epoch + 1, np.mean(loss_rgb_list), np.mean(loss_depth_list)), file=log)

    return


def train(device, classname, data_root, log, epochs, learning_rate, batch_size, img_size, ckp):
    if ckp is not None:
        ckp_path = ckp + classname + '.pth'
    else:
        ckp_path = None

    train_mean = [0.485, 0.456, 0.406]
    train_std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(train_mean, train_std)])
    depth_transform = transforms.Compose([
        transforms.Resize((img_size, img_size), antialias=True),
        transforms.Normalize(train_mean, train_std)])

    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(train_mean, train_std)])
    gt_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()])

    train_dir = data_root + classname + '/train'
    valid_dir = data_root + classname + '/validation'
    test_dir = data_root + classname + '/test'

    k_dict = {'cookie': 3, 'dowel' : 7, 'foam' : 7, 'tire' : 7}
    if classname in k_dict:
        k = k_dict[classname]
    else:
        k = 5
    print(k)
    print(k, file=log)

    train_data = MVTecADRGBNDataset(data_dir=train_dir, transform=transform, depth_transform=depth_transform, k=k)
    valid_data = MVTecADRGBNDataset(data_dir=valid_dir, transform=transform, depth_transform=depth_transform, k=k)
    test_data = MVTecADRGBNDataset(data_dir=test_dir, transform=test_transform, depth_transform=depth_transform,
                                   test=True, gt_transform=gt_transform, k=k)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=16)
    valid_dataloader = DataLoader(valid_data, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

    teacher_rgb = ResNet50Encoder()
    teacher_rgb.to(device)
    teacher_rgb.eval()

    teacher_depth = ResNet50Encoder()
    teacher_depth.to(device)
    teacher_depth.eval()

    student_depth = ResNet50DualModalDecoder(pretrained=False)
    student_depth.to(device)
    student_depth.train()

    student_rgb = ResNet50DualModalDecoder(pretrained=False)
    student_rgb.to(device)
    student_rgb.train()

    optimizer_rgb = torch.optim.Adam(student_rgb.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_depth = torch.optim.Adam(student_depth.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    scheduler_rgb = torch.optim.lr_scheduler.ExponentialLR(optimizer_rgb, gamma=0.95)
    scheduler_depth = torch.optim.lr_scheduler.ExponentialLR(optimizer_depth, gamma=0.95)

    for epoch in range(epochs):
        student_rgb.train()
        student_depth.train()
        train_one_epoch(teacher_rgb, teacher_depth, student_rgb, student_depth, train_dataloader, optimizer_rgb, optimizer_depth, device, log, epoch)
        student_rgb.eval()
        student_depth.eval()

        scheduler_rgb.step()
        scheduler_depth.step()

    params = valid(teacher_rgb, teacher_depth, student_rgb, student_depth, valid_dataloader, device)
    print(params)
    print(params,file=log)
    test(teacher_rgb, teacher_depth, student_rgb, student_depth, test_dataloader, device, log, epochs - 1, classname,
         data_root, ckp_path, params)


def test(teacher_rgb, teacher_depth, student_rgb, student_depth, test_dataloader, device, log, epoch, classname, data_root, ckp_path, params=None):
    if ckp_path is not None:
        torch.save({'student_rgb': student_rgb.state_dict(),
                'student_depth': student_depth.state_dict()}, ckp_path)

    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    pr_list_px_r = []
    pr_list_sp_r = []
    pr_list_px_d = []
    pr_list_sp_d = []
    gts = []
    predictions = []
    predictions_r = []
    predictions_d = []

    with torch.no_grad():
        for i, data in enumerate(test_dataloader, 0):
            rgb_image, depth_image, gt, ad_label, ad_type = data
            rgb_img = rgb_image.to(device)
            depth_img = depth_image.to(device)

            output_Trgb = teacher_rgb(rgb_img)
            output_Td = teacher_depth(depth_img)

            _, _, _, output_Srgb = student_rgb(output_Trgb, output_Td)
            _, _, _, output_Sd = student_depth(output_Td, output_Trgb)

            anomaly_map_rgb, _ = cal_anomaly_map(output_Srgb, output_Trgb, out_size=img_size, amap_mode='add')
            anomaly_map_depth, _ = cal_anomaly_map(output_Sd, output_Td, out_size=img_size, amap_mode='add')

            if params is not None:
                anomaly_map_rgb = (anomaly_map_rgb - params[0]) / params[1]
                anomaly_map_depth = (anomaly_map_depth - params[2]) / params[3]

            anomaly_map = anomaly_map_rgb + anomaly_map_depth

            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.ravel())
            gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
            pr_list_sp.append(np.max(anomaly_map))
            gts.append(gt.squeeze().cpu().detach().numpy())  # * (256,256)
            predictions.append(anomaly_map)  # * (256,256)

            anomaly_map_rgb = gaussian_filter(anomaly_map_rgb, sigma=4)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            pr_list_px_r.extend(anomaly_map_rgb.ravel())
            pr_list_sp_r.append(np.max(anomaly_map_rgb))
            predictions_r.append(anomaly_map_rgb)  # * (256,256)

            anomaly_map_depth = gaussian_filter(anomaly_map_depth, sigma=4)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            pr_list_px_d.extend(anomaly_map_depth.ravel())
            pr_list_sp_d.append(np.max(anomaly_map_depth))
            predictions_d.append(anomaly_map_depth)  # * (256,256)

        print('-----------------------testing %d epoch-----------------------' % (epoch + 1))
        print('-----------------------testing %d epoch-----------------------' % (epoch + 1), file=log)


        print('add:')
        print('add:', file=log)
        test_metric(gt_list_px, pr_list_px, gt_list_sp, pr_list_sp, predictions, gts)

        print('rgb:')
        print('rgb:', file=log)
        test_metric(gt_list_px, pr_list_px_r, gt_list_sp, pr_list_sp_r, predictions_r, gts)

        print('depth:')
        print('depth:', file=log)
        test_metric(gt_list_px, pr_list_px_d, gt_list_sp, pr_list_sp_d, predictions_d, gts)


    return


def test_metric(gt_list_px, pr_list_px, gt_list_sp, pr_list_sp, predictions, gts):
    auroc_px = roc_auc_score(gt_list_px, pr_list_px)
    auroc_sp = roc_auc_score(gt_list_sp, pr_list_sp)

    ap_px = average_precision_score(gt_list_px, pr_list_px)
    ap_sp = average_precision_score(gt_list_sp, pr_list_sp)

    f1_px = f1_score_max(gt_list_px, pr_list_px)
    f1_sp = f1_score_max(gt_list_sp, pr_list_sp)

    au_pros, _ = calculate_au_pro(gts, predictions)
    pro = au_pros[0]
    pro_10 = au_pros[1]
    pro_5 = au_pros[2]
    pro_1 = au_pros[3]

    print(" I-AUROC | P-AUROC | AUPRO@30% | AUPRO@10% | AUPRO@5% | AUPRO@1% |   I-AP   |   P-AP   |   I-F1   |   P-F1")
    print(
        f'  {auroc_sp:.3f}  |  {auroc_px:.3f}  |   {pro:.3f}   |   {pro_10:.3f}   |   {pro_5:.3f}  |   {pro_1:.3f}  |   {ap_sp:.3f}  |   {ap_px:.3f}  |   {f1_sp:.3f}  |   {f1_px:.3f}',
        end='\n')

    print(" I-AUROC | P-AUROC | AUPRO@30% | AUPRO@10% | AUPRO@5% | AUPRO@1% |   I-AP   |   P-AP   |   I-F1   |   P-F1", file=log)
    print(
        f'  {auroc_sp:.3f}  |  {auroc_px:.3f}  |   {pro:.3f}   |   {pro_10:.3f}   |   {pro_5:.3f}  |   {pro_1:.3f}  |   {ap_sp:.3f}  |   {ap_px:.3f}  |   {f1_sp:.3f}  |   {f1_px:.3f}',
        end='\n', file=log)


def valid(teacher_rgb, teacher_depth, student_rgb, student_depth, valid_dataloader, device):
    a_rgb = []
    a_depth = []
    with torch.no_grad():
        for i, data in enumerate(valid_dataloader, 0):
            rgb_image, depth_image, _, _, _ = data
            rgb_img = rgb_image.to(device)
            depth_img = depth_image.to(device)

            with torch.no_grad():
                output_Trgb = teacher_rgb(rgb_img)
                output_Td = teacher_depth(depth_img)

                _, _, _, output_Srgb = student_rgb(output_Trgb, output_Td)
                _, _, _, output_Sd = student_depth(output_Td, output_Trgb)

                anomaly_map_rgb, _ = cal_anomaly_map(output_Srgb, output_Trgb, out_size=img_size, amap_mode='add')
                anomaly_map_depth, _ = cal_anomaly_map(output_Sd, output_Td, out_size=img_size, amap_mode='add')

                a_rgb.append(anomaly_map_rgb)
                a_depth.append(anomaly_map_depth)

    a_rgb_array = np.array(a_rgb)
    a_depth_array = np.array(a_depth)

    mean_r = np.mean(a_rgb_array)
    std_r = np.std(a_rgb_array)

    mean_d = np.mean(a_depth_array)
    std_d = np.std(a_depth_array)

    return [mean_r, std_r, mean_d, std_d]


if __name__ == "__main__":

    classnames = ['bagel', 'cable_gland', 'carrot', 'cookie', 'dowel', 'foam', 'peach', 'potato', 'rope', 'tire']

    learning_rate = 0.005
    batch_size = 16
    img_size = 256
    data_root = '../data/mvtec_3d_anomaly_detection/'
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    log = open("./logs/CRD_mvtec3d_rgb_normal_seed42_200e.txt",'a')
    ckp = None

    for i in range(len(classnames)):
        setup_seed(42)
        classname = classnames[i]
        epochs_i = 200
        print('-----------------------training on ' + classname + '[%d / %d]-----------------------' % (
            i + 1, len(classnames)))
        print('-----------------------training on ' + classname + '[%d / %d]-----------------------' % (
            i + 1, len(classnames)), file=log)
        train(device, classname, data_root, log, epochs_i, learning_rate, batch_size, img_size, ckp)
