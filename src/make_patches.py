from typing import Tuple, List

import concurrent.futures
import glob
import math
import os
import pathlib
import random
import time
import xml.etree.ElementTree

import numpy as np
import pandas as pd
import PIL.Image
import PIL.ImageFilter
import PIL.ImageEnhance

dll_path = pathlib.Path(r'lib\openslide\bin')
os.add_dll_directory(dll_path.resolve())
import openslide

tma_dir = r'data\raw\tma'
tma_list_file = r'data\tma_wsi_data.xlsx'
tma_sheet_name = 'list'
tcga_dir = r'data\raw\tcga'
tcga_list_file = r'data\tcga_wsi_data.xlsx'
tcga_sheet_name = 'list'

max_process_num = 8
image_w = 224
image_h = 224
microns_per_pixel = 0.91
tcga_annotation_num = 4

blur_rate = 0.20
max_blur_radius = 1.2
epsilon = 1.0

# Optical density of hematoxylin and eosin
h_od = np.array([0.6972, 0.6620, 0.2752])
e_od = np.array([0.0005, 0.9508, 0.3097])


class Annotation:
    """
    Single annotation.
    """
    def __init__(self):
        self.title: str
        self.type: str
        self.closed: bool
        self.pointlist: List[Tuple[float, float]]
        self.left: float
        self.top: float
        self.right: float
        self.bottom: float

    def encloses(self, point: Tuple[float, float]) -> bool:
        """
        Check whether a certain point is included in this annotation
        """
        if self.type != 'freehand' or not self.closed:
            raise RuntimeError('Invalid annotation type: Annotation.encloses')
        
        c = 0
        px, py = point
        p0x, p0y = self.pointlist[-1]
        for p1x, p1y in self.pointlist:
            if (p0y <= py < p1y) or (p0y > py >= p1y):
                if px > (py - p0y) * (p1x - p0x) / (p1y - p0y) + p0x:
                    c += 1
            p0x, p0y = p1x, p1y

        return (c % 2) == 1

    def pixel_area(self) -> float:
        """
        Area of this annotation at level0 [pixels]
        """
        if self.type != 'freehand' or not self.closed:
            raise RuntimeError('Invalid annotation type: Annotation.pixel_area')
        
        s = 0.0
        p0x, p0y = self.pointlist[-1]
        for p1x, p1y in self.pointlist:
            s += (p0x * p1y - p1x * p0y)
            p0x, p0y = p1x, p1y

        return abs(s / 2.0)


class AnnotationList:
    """
    List of annotations included in a WSI.
    """
    def __init__(self, file_name: str, associated_slide: openslide.OpenSlide):
        self.annotations: List[Annotation] = []
        self.mppx: float
        self.mppy: float

        vendor = associated_slide.properties[openslide.PROPERTY_NAME_VENDOR]
        self.mppx = float(associated_slide.properties[openslide.PROPERTY_NAME_MPP_X])
        self.mppy = float(associated_slide.properties[openslide.PROPERTY_NAME_MPP_Y])
        if vendor == 'hamamatsu':
            offset_x = float(associated_slide.properties['hamamatsu.XOffsetFromSlideCentre'])
            offset_y = float(associated_slide.properties['hamamatsu.YOffsetFromSlideCentre'])
            dimension_x, dimension_y = associated_slide.dimensions

        tree = xml.etree.ElementTree.parse(file_name)
        root = tree.getroot()

        for e_ndpviewstate in root.findall('ndpviewstate'):
            annotation = Annotation()
            for e_title in e_ndpviewstate.findall('title'):
                annotation.title = e_title.text
            for e_annotation in e_ndpviewstate.findall('annotation'):
                annotation.type = e_annotation.get('type')
                for e_closed in e_annotation.findall('closed'):
                    annotation.closed = (e_closed.text == '1')

                if annotation.type == 'freehand' and annotation.closed == True:
                    annotation.pointlist = []
                    for e_pointlist in e_annotation.findall('pointlist'):
                        for e_point in e_pointlist.findall('point'):
                            for e_x in e_point.findall('x'):
                                x = float(e_x.text)
                            for e_y in e_point.findall('y'):
                                y = float(e_y.text)
                            if vendor == 'hamamatsu':
                                x = (x - offset_x) / (self.mppx * 1000.0) + dimension_x / 2.0
                                y = (y - offset_y) / (self.mppy * 1000.0) + dimension_y / 2.0
                            elif vendor == 'aperio':
                                x = x / (self.mppx * 1000.0)
                                y = y / (self.mppy * 1000.0)
                            annotation.pointlist.append((x, y))
                    annotation.left = min(annotation.pointlist, key=(lambda x: x[0]))[0]
                    annotation.top = min(annotation.pointlist, key=(lambda x: x[1]))[1]
                    annotation.right = max(annotation.pointlist, key=(lambda x: x[0]))[0]
                    annotation.bottom = max(annotation.pointlist, key=(lambda x: x[1]))[1]
            self.annotations.append(annotation)

    def exist(self, title: str) -> bool:
        """
        Check whether a certain annotation exists.
        """
        for annotation in self.annotations:
            if annotation.title == title:
                return True
    
        return False

    def boundary(self, title: str) -> Tuple[float, float, float, float]:
        """
        Returns the bounding box.
        """
        items = [item for item in self.annotations if item.title == title]

        if len(items) != 0:
            left = min(items, key=(lambda x: x.left)).left
            top = min(items, key=(lambda x: x.top)).top
            right = max(items, key=(lambda x: x.right)).right
            bottom = max(items, key=(lambda x: x.bottom)).bottom
            return left, top, right, bottom
        else:
            raise RuntimeError('No valid region: AnnotationList.boundary')

    def encloses(self, point: Tuple[float, float], title: str) -> bool:
        """
        Check whether a certain point is included in any of the annotations.
        """
        for annotation in self.annotations:
            if annotation.title == title and annotation.encloses(point):
                return True
        
        return False

    def area(self, title: str) -> float:
        """
        Total area of the annotations [mm2].
        """
        s = 0.0
        for annotation in self.annotations:
            if annotation.title == title:
                s += annotation.pixel_area()

        return s * (self.mppx / 1000.0) * (self.mppy / 1000.0)


def sample_from_tma(output_dir: str, target_groups: List[int],
        augmentation: bool, image_num: int) -> None:
    """
    Make patches from TMA WSI.
    image_num: Number of images per mm2.
    """
    case_num = 0
    tma, case, group, amount, focus = [], [], [], [], []
    types = {'N_TMA': int, 'N_Case': int, 'Group': int,
            'Amount1': int, 'Amount2': int, 'Focus1': int, 'Focus2': int}
    df = pd.read_excel(tma_list_file, tma_sheet_name, dtype=types)

    for index, row in df.iterrows():
        case_num += 1
        tma.append(row['N_TMA'])
        case.append(row['N_Case'])
        group.append(row['Group'])
        amount.append((row['Amount1'], row['Amount2']))
        focus.append((row['Focus1'], row['Focus2']))

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_process_num) as executor:
        for index in range(case_num):
           if group[index] not in target_groups:
               continue
           
           for d in range(0, 2):  # Two cores are included in a case.
                if amount[index][d] != 0:  # When the amount of the tumor is not sufficient
                    continue

                file = tma_dir + fr'\TMA{tma[index]}_HE.ndpi'
                if not os.path.exists(file):
                    print('File not found: ' + file)
                    continue

                case_str = f'{case[index]:0>2}'
                title = case_str + f'-{d + 1}-T'
                output_subdir = output_dir + fr'\{tma[index]}' + '\\' + case_str

                executor.submit(sample_images, file, title, augmentation,
                        image_num, focus[index][d], output_subdir, f'_{d + 1}')
                

def sample_from_tcga(output_dir, target_groups: List[int],
        augmentation: bool, image_num: int) -> None:
    """
    Make patches from TCGA WSI.
    image_num: Number of images per mm2.
    """
    case_num = 0
    code, group = [], []

    types = {'TCGA barcode': str, 'Group': int}
    df = pd.read_excel(tcga_list_file, tcga_sheet_name, dtype=types)

    for index, row in df.iterrows():
        case_num += 1
        code.append(row['TCGA barcode'])
        group.append(row['Group'])

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_process_num) as executor:
        for index in range(case_num):
            if group[index] not in target_groups:
                continue

            path_list = glob.glob(tcga_dir + '\\' + code[index] + '*.svs')
            if len(path_list) == 0:
                print('File not found: ' + code[index])
                continue
            elif len(path_list) > 1:
                print('Multiple files found: ' + code[index])
                continue

            output_subdir = output_dir + '\\' + code[index]

            for i in range(tcga_annotation_num):
                executor.submit(sample_images, path_list[0], f'T{i}', augmentation,
                        image_num, 0, output_subdir, f'_{i}')


def sample_images(file: str, annotation_title: str, augmentation: bool, image_num: int,
        focus: int, output_dir: str, suffix: str) -> None:
    """
    Sample images from a WSI.
    file: WSI file name.
    annotation_title: Target annotation title.
    focus: Values other than zero represents that the WSI is not in focus
           (additional blurring will not be applied).
    """
    with openslide.OpenSlide(file) as slide:
        annotation_file = file + '.ndpa'
        if not os.path.exists(annotation_file):
            print('Annotation file not found: ' + annotation_file)
            return

        annotations = AnnotationList(annotation_file, slide)
        if not annotations.exist(annotation_title):
            print('Annotation not found: ' + annotation_file)
            return

        base_rect = annotations.boundary(annotation_title)
        margin = math.sqrt(image_w ** 2 + image_h ** 2) / 2.0 + 10.0
        target_img, target_factor, target_left, target_top = read_region_from_slide(
                slide, base_rect, margin)

        if augmentation:
            if not annotations.exist('B'):
                print('Background annotation not found: ' + annotation_file)
                return

            background_rect = annotations.boundary('B')
            background_img, _, _, _ = read_region_from_slide(slide, background_rect, 0.0)
            ary = img_to_ary(background_img)
            ary = rgb_to_od(ary)
            background = np.mean(ary, axis=(0, 1))

            ary = img_to_ary(target_img)
            ary = rgb_to_od(ary)
            ary = ary - background

            # Calculates the deconvolution matrix and coefficients for normalization.
            outlier_th = 0.1
            valid_th = 0.5
            target_mean = np.array([0.97, 0.77, 0.0])  # TMA298 Core11-1
            norm_factor = np.array([1.0, 1.0, 1.0])  # Coefficients for normalization [H, E, R].
            conv_m = get_optimal_convolution_matrix(ary)
            deconv_m = np.linalg.inv(conv_m)
            temp1 = np.dot(ary, deconv_m)
            for i in range(10):
                mask = np.logical_and(temp1[:, :, 2] > -outlier_th, temp1[:, :, 2] < outlier_th)
                mask = np.logical_and(mask, temp1[:, :, 0] > -outlier_th / norm_factor[0])
                mask = np.logical_and(mask, temp1[:, :, 1] > -outlier_th / norm_factor[1])
                conv_m = get_optimal_convolution_matrix(ary[mask])
                deconv_m = np.linalg.inv(conv_m)
                temp1 = np.dot(ary, deconv_m)

                temp2 = (temp1[mask])[:, 0]
                norm_factor[0] = target_mean[0] / np.mean(temp2[temp2 > valid_th / norm_factor[0]])
                temp2 = (temp1[mask])[:, 1]
                norm_factor[1] = target_mean[1] / np.mean(temp2[temp2 > valid_th / norm_factor[1]])
            
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num = math.ceil(image_num * annotations.area(annotation_title))
    coverage = 1.0
    cov_margin = 0.1
    cov_dec = 0.01
    duration = 0.2
    fixed = False

    for i in range(num):
        t = time.time()
        while True:
            # Find a window covered by tumor area.
            cx = random.uniform(base_rect[0], base_rect[2])
            cy = random.uniform(base_rect[1], base_rect[3])

            if not annotations.encloses((cx, cy), annotation_title):
                continue

            angle = random.uniform(0.0, math.pi * 2.0)

            # Point counting method.
            div = 11
            cos, sin = math.cos(angle), math.sin(angle)
            center = np.array([cx, cy])
            ex = np.array([cos, sin]) * (image_w * target_factor / div)
            ey = np.array([-sin, cos]) * (image_h * target_factor / div)
            count = 0

            for y in range(-(div // 2), div // 2 + 1):
                for x in range(-(div // 2), div // 2 + 1):
                    p = center + x * ex + y * ey
                    if annotations.encloses(p, annotation_title):
                        count += 1

            if count / (div ** 2) >= coverage:
                if not fixed:
                    coverage -= cov_margin
                    fixed = True
                break

            t2 = time.time()
            if t2 - t > duration:
                coverage -= cov_dec
                t = t2

        cx = cx / target_factor - target_left
        cy = cy / target_factor - target_top
        flip = random.random() < 0.5

        mt1 = np.array([[1.0, 0.0, -image_w / 2.0], [0.0, 1.0, -image_h / 2.0], [0.0, 0.0, 1.0]])
        ms = np.array([[(-1.0 if flip else 1.0), 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        mr = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
        mt2 = np.array([[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1.0]])
        m = np.dot(mt2, np.dot(mr, np.dot(ms, mt1)))

        img = target_img.transform((image_w, image_h),
                PIL.Image.AFFINE, m[:2].flatten(), PIL.Image.BICUBIC)

        if augmentation:
            img = color_augmentation(img, background, norm_factor, deconv_m)

            if focus == 0:  # When the original image is in focus
                if random.random() < blur_rate:
                    img = img.filter(PIL.ImageFilter.GaussianBlur(
                            random.uniform(0.0, max_blur_radius)))
        else:
            img = PIL.Image.fromarray(np.uint8(img_to_ary(img)))  # Delete alpha channel.

        img.save(output_dir + fr'\{i:0>8}' + suffix + '.png')

    print(output_dir + f' {suffix}: finished ({num}/{num}), coverage: {coverage}')


def read_region_from_slide(slide: openslide.OpenSlide, rect: Tuple[float, float, float, float],
                           margin: float) -> Tuple[PIL.Image.Image, float, float, float]:
    """
    Read region from a WSI.
    Returns: (image, target_factor, target_left, target_top)
    """
    base_mpp = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
    target_factor = microns_per_pixel / base_mpp
    best_level = slide.get_best_level_for_downsample(target_factor)
    best_factor = slide.level_downsamples[best_level]
        
    base_left, base_top, base_right, base_bottom = rect
    target_left = base_left / target_factor - margin
    target_top = base_top / target_factor - margin
    target_right = base_right / target_factor + margin
    target_bottom = base_bottom / target_factor + margin
    target_w = round(target_right - target_left)
    target_h = round(target_bottom - target_top)

    best_left = target_left * target_factor / best_factor
    best_top = target_top * target_factor / best_factor
    best_right = target_right * target_factor / best_factor
    best_bottom = target_bottom * target_factor / best_factor
    best_w = round(best_right - best_left)
    best_h = round(best_bottom - best_top)

    best_img = slide.read_region((int(best_left * best_factor), int(best_top * best_factor)),
                                 best_level, (best_w, best_h))

    if target_w == best_w and target_h == best_h:
        target_img = best_img
    else:
        target_img = best_img.resize((target_w, target_h), PIL.Image.BICUBIC)

    return target_img, target_factor, target_left, target_top


def get_optimal_convolution_matrix(ary: np.ndarray) -> np.ndarray:
    """
    Find a plane formed by RGB ODs by least squares method. Then, project base vectors of
    hematoxylin and eosin, and calculate the convolution matrix.
    """
    ary = ary.reshape(-1, 3)
    x = ary[:, 0]
    y = ary[:, 1]
    z = ary[:, 2]
    
    m = np.array([[1.0, np.mean(x), np.mean(y)],
                  [np.mean(x), np.mean(x * x), np.mean(x * y)],
                  [np.mean(y), np.mean(x * y), np.mean(y * y)]])
    v = np.array([np.mean(z), np.mean(x * z), np.mean(y * z)])
    m = np.linalg.inv(m)
    a, b, c = np.dot(v, m)
    n = np.array([b, c, -1.0])

    h = h_od - (np.dot(h_od, n) + a) / np.linalg.norm(n) ** 2 * n
    h = h / np.linalg.norm(h)

    e = e_od - (np.dot(e_od, n) + a) / np.linalg.norm(n) ** 2 * n
    e = e / np.linalg.norm(e)

    r = np.cross(h, e)
    r = r / np.linalg.norm(r)

    return np.array([h, e, r])


def color_augmentation(img: PIL.Image.Image, background: np.ndarray, norm_factor: np.ndarray,
                       deconv_m: np.ndarray) -> PIL.Image.Image:
    """
    Data augmentation by random color change.
    """
    ary = img_to_ary(img)
    ary = rgb_to_od(ary)
    ary = ary - background
    ary = np.dot(ary, deconv_m)

    def get_random_param(min, max):
        return np.random.rand(3) * (max - min) + min

    stain_factor_c = random.uniform(-0.8, 0.8)
    stain_factor = get_random_param(-0.4, 0.4) + stain_factor_c
    stain_factor = np.exp(stain_factor)
    result_background = get_random_param(0.0, 0.4)

    conv_m_h = np.exp(get_random_param(-1.0, 1.0))
    conv_m_h = h_od * conv_m_h
    conv_m_h = conv_m_h / np.linalg.norm(conv_m_h)
    conv_m_e = np.exp(get_random_param(-1.0, 1.0))
    conv_m_e = e_od * conv_m_e
    conv_m_e = conv_m_e / np.linalg.norm(conv_m_e)
    conv_m_r = np.cross(conv_m_h, conv_m_e)
    conv_m_r = conv_m_r / np.linalg.norm(conv_m_r)
    conv_m = np.array([conv_m_h, conv_m_e, conv_m_r])
    
    ary = ary * norm_factor * stain_factor
    ary = np.dot(ary, conv_m)
    ary = ary + result_background
    ary = np.clip(od_to_rgb(ary), 0.0, 255.0)
    img = PIL.Image.fromarray(np.uint8(ary))

    saturation = math.exp(random.uniform(-0.3, 0.3))
    contrast = math.exp(random.uniform(-0.3, 0.3))
    brightness = math.exp(random.uniform(-0.3, 0.3))
    img = PIL.ImageEnhance.Color(img).enhance(saturation)
    img = PIL.ImageEnhance.Contrast(img).enhance(contrast)
    img = PIL.ImageEnhance.Brightness(img).enhance(brightness)

    return img


def rgb_to_od(ary: np.ndarray) -> np.ndarray:
    """
    Convert RGB value [0-255] to optical density
    """
    return -np.log((ary + epsilon) / (255.0 + epsilon))


def od_to_rgb(ary: np.ndarray) -> np.ndarray:
    """
    Convert optical density to RGB value [0-255]
    """
    return np.exp(-ary) * (255.0 + epsilon) - epsilon


def img_to_ary(img: PIL.Image.Image) -> np.ndarray:
    """
    Convert a PIL image to a numpy array
    """
    ary = np.array(img, dtype=np.float32)
    if ary.shape[2] == 4:
        ary = np.delete(ary, 3, 2)
    return ary


if __name__ == '__main__':
    # Make patches for training.
    sample_from_tma(r'data\patches\tma_with_aug', [1, 2, 3, 4], True, 1500)
    sample_from_tma(r'data\patches\tma_no_aug', [1, 2, 3, 4], False, 1500)
    sample_from_tcga(r'data\patches\tcga_with_aug', [0], True, 500)
    sample_from_tcga(r'data\patches\tcga_no_aug', [0], False, 500)

    # Make patches for test.
    sample_from_tma(r'data\patches\tma_test', [0], False, 150)
    sample_from_tcga(r'data\patches\tcga_test', [0, 1, 2, 3, 4], False, 50)
