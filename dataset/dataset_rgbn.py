from torch.utils.data import Dataset
import os
from PIL import Image

from .geo_utils import *
from .mvtec3d_utils import *

import tifffile


def get_max_min_depth_img(image_path):
    image = tifffile.imread(image_path).astype(np.float32)
    image_t = (
        np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
    )
    image = image_t[:, :, 2]
    zero_mask = np.where(image == 0, np.ones_like(image), np.zeros_like(image))
    im_max = np.max(image)
    im_min = np.min(image * (1.0 - zero_mask) + 1000 * zero_mask)
    return im_min, im_max


def depth_to_normal_map(depth, k=5, mask=None):
    """
    Convert depth map (H, W) to normal map (H, W, 3)
    depth: numpy array, depth in float (missing background可为0)
    mask: optional foreground mask (1 for foreground)
    """
    dzdx = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=k)
    dzdy = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=k)

    normal = np.dstack((-dzdx, -dzdy, np.ones_like(depth)))
    norm = np.linalg.norm(normal, axis=2, keepdims=True)
    normal /= (norm + 1e-8)

    normal_img = (normal + 1) / 2.0

    if mask is not None:
        normal_img *= mask[:, :, None]

    return normal_img


class MVTecADRGBNDataset(Dataset):
    def __init__(self, data_dir, transform=None, depth_transform=None, test=False, gt_transform=None, test_min=0.0, test_max=1.0, k=5):
        self.transform = transform
        self.gt_transform = gt_transform
        self.data_info = self.get_data_info(data_dir)
        self.depth_transform = depth_transform

        self.global_max, self.global_min = 1, 0
        if not test:
            for data_info_i in self.data_info:
                rgb_path, depth_path, gt, ad_label, ad_type = data_info_i
                im_min, im_max = get_max_min_depth_img(depth_path)
                self.global_min = min(self.global_min, im_min)
                self.global_max = max(self.global_max, im_max)
            self.global_min = self.global_min * 0.9
            self.global_max = self.global_max * 1.1
        else:
            self.global_max = test_max
            self.global_min = test_min

        self.k = k

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, index):
        rgb_path, depth_path, gt, ad_label, ad_type = self.data_info[index]
        rgb_img = Image.open(rgb_path).convert('RGB')
        if self.transform is not None:
            rgb_img = self.transform(rgb_img)

        normal_img, plane_mask = self.get_normal_image(depth_path, rgb_img.size()[-2], self.k)

        if self.depth_transform is not None:
            normal_img = self.depth_transform(normal_img)

        if gt == 0:
            gt = torch.zeros([1, rgb_img.size()[-2], rgb_img.size()[-2]])
        else:
            gt = Image.open(gt).convert('L')
            if self.gt_transform is not None:
                gt = self.gt_transform(gt)

        return rgb_img, normal_img, gt, ad_label, ad_type

    def get_data_info(self, data_dir):
        data_info = list()

        for root, dirs, _ in os.walk(data_dir):
            # print(dirs)
            for sub_dir in dirs:
                rgb_names = os.listdir(os.path.join(root, sub_dir, 'rgb'))
                rgb_names = list(filter(lambda x: x.endswith('.png'), rgb_names))
                for rgb_name in rgb_names:
                    rgb_path = os.path.join(root, sub_dir, 'rgb', rgb_name)
                    depth_name = rgb_name.replace(".png", ".tiff")
                    depth_path = os.path.join(root, sub_dir, 'xyz', depth_name)
                    if sub_dir == 'good':
                        data_info.append((rgb_path, depth_path, 0, 0, sub_dir))
                    else:
                        gt_name = rgb_name
                        gt_path = os.path.join(root, sub_dir, 'gt', gt_name)
                        data_info.append((rgb_path, depth_path, gt_path, 1, sub_dir))

            break

        np.random.shuffle(data_info)

        return data_info

    def get_normal_image(self, file, target_size=None, k=5):
        xyz_data = tifffile.imread(file).astype(np.float32)
        H_orig, W_orig, C = xyz_data.shape

        size = self.image_size if target_size is None else target_size

        original_depth = xyz_data[:, :, 2]
        image = original_depth
        image_t = xyz_data

        zero_mask = np.where(image == 0, np.ones_like(image), np.zeros_like(image))
        plane_mask = get_plane_mask(image_t)  # 0 is background, 1 is foreground
        plane_mask[:, :, 0] = plane_mask[:, :, 0] * (1.0 - zero_mask)
        plane_mask = fill_plane_mask(plane_mask)
        plane_mask_2d = plane_mask[:, :, 0]

        image = image * plane_mask[:, :, 0]
        zero_mask = np.where(image == 0, np.ones_like(image), np.zeros_like(image))
        im_min = np.min(image * (1.0 - zero_mask) + 1000 * zero_mask)
        im_max = np.max(image)
        image = (image - im_min) / (im_max - im_min)
        # image = image * 0.8 + 0.1
        image = image * (1.0 - zero_mask)
        image = fill_depth_map(image)

        filled_normalized_depth = image

        normals_map_raw = depth_to_normal_map(filled_normalized_depth, k=k)  # HxWx3

        normals_map_processed = normals_map_raw

        plane_mask_3d = np.expand_dims(plane_mask_2d, axis=2)
        final_normals = normals_map_processed * plane_mask_3d

        final_normals_resized = cv2.resize(
            final_normals, (size, size),
            interpolation=cv2.INTER_LINEAR
        )

        normal_img = final_normals_resized.transpose((2, 0, 1))

        plane_mask_resized = cv2.resize(
            plane_mask[:, :, 0], (size, size),
            interpolation=cv2.INTER_NEAREST
        )
        plane_mask_resized = np.expand_dims(plane_mask_resized, 2)

        return torch.FloatTensor(normal_img), torch.FloatTensor(
            np.squeeze(plane_mask_resized)
        )
