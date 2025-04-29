import os
import cv2
import numpy as np
from typing import List
from shapely import affinity
from shapely.geometry import Polygon
from tqdm import tqdm

# from .ballon_extractor import extract_ballon_region
from . import text_render
from .text_render_eng import render_textblock_list_eng
from ..utils import (
    BASE_PATH,
    TextBlock,
    color_difference,
    get_logger,
    rotate_polygons,
)
from ..config import Config

logger = get_logger('render')

def parse_font_paths(path: str, default: List[str] = None) -> List[str]:
    if path:
        parsed = path.split(',')
        parsed = list(filter(lambda p: os.path.isfile(p), parsed))
    else:
        parsed = default or []
    return parsed

def fg_bg_compare(fg, bg):
    fg_avg = np.mean(fg)
    if color_difference(fg, bg) < 30:
        bg = (255, 255, 255) if fg_avg <= 127 else (0, 0, 0)
    return fg, bg

def resize_regions_to_font_size(img: np.ndarray, text_regions: List['TextBlock'], font_size_fixed: int, font_size_offset: int, font_size_minimum: int):  
    """  
    调整文本区域大小以适应字体大小和翻译文本长度。  

    此函数会根据以下逻辑调整每个文本区域的边界框：  
    1. 确定最小字体大小。  
    2. 根据原始字体大小、固定字体大小和偏移量计算目标字体大小。  
    3. **如果翻译文本长度大于原文文本，则根据长度比例计算一个缩放因子，并将缩放因子限制在1.1到1.4倍之间。**  
    4. **使用计算出的缩放因子（如果应用）和目标字体大小来调整原始边界框的大小。**  
    5. 将调整后的边界框裁剪到图像边界内。  
    6. 更新 TextBlock 对象的字体大小。  

    Args:  
        img (np.ndarray): 输入图像。  
        text_regions (List[TextBlock]): 待处理的文本区域列表。  
        font_size_fixed (int): 固定字体大小 (如果提供，忽略其他字体大小参数)。  
        font_size_offset (int): 字体大小偏移量。  
        font_size_minimum (int): 最小字体大小。如果为 -1，则根据图像尺寸自动计算。  

    Returns:  
        List[np.ndarray]: 调整后的文本区域边界框列表，每个边界框是一个 (4, 2) 的 NumPy 数组。  
    """  
    # 1. 定义最小字体大小  
    if font_size_minimum == -1:  
        font_size_minimum = round((img.shape[0] + img.shape[1]) / 200)  
    logger.debug(f'font_size_minimum {font_size_minimum}')  
    font_size_minimum = max(1, font_size_minimum)  

    dst_points_list = []  
    for region in text_regions:  
        # 存储该区域的原始字体大小  
        original_region_font_size = region.font_size  
        # 确保原始字体大小有效  
        if original_region_font_size <= 0:  
            logger.warning(f"原始字体大小无效 ({original_region_font_size}) 对于文本 '{region.translation[:10]}'. 使用默认值 {font_size_minimum}。")  
            original_region_font_size = font_size_minimum # 使用最小值作为默认值  

        # 2. 确定目标字体大小  
        current_base_font_size = original_region_font_size  
        if font_size_fixed is not None:  
            target_font_size = font_size_fixed  
        else:  
            # 将偏移量应用于原始字体大小  
            target_font_size = current_base_font_size + font_size_offset  

        # 应用最小字体大小限制  
        target_font_size = max(target_font_size, font_size_minimum)  
        # 确保字体大小至少为 1  
        target_font_size = max(1, target_font_size)  
        # logger.debug(f"计算目标字体大小: {target_font_size} 对于文本 '{region.translation[:10]}'")  

        # 3. 计算基于文本长度比例的缩放因子  
        #char_count_orig = len(region.text.strip())
        orig_text = getattr(region, "text_raw", region.text)  # 若没保存则退回现有 text
        char_count_orig = len(orig_text.strip())        
        char_count_trans = len(region.translation.strip())  
        length_ratio = 1.0 # 默认缩放因子为 1.0  

        if char_count_orig > 0 and char_count_trans > char_count_orig:  
             # 翻译文本更长，计算长度比例  
            length_ratio = char_count_trans / char_count_orig  
            logger.debug(f"文本长度比例: {length_ratio:.2f} ({char_count_trans} / {char_count_orig}) 对于文本 '{region.translation}'")  
            # 将缩放因子限制在 1.1 到 1.4 倍之间  
            target_scale = max(1.1, min(length_ratio, 1.4))  
            logger.debug(f"应用长度比例缩放，目标缩放因子 (限制后): {target_scale:.2f}")  
        else:  
            # 翻译文本不长于原文，不应用长度比例缩放，只考虑字体大小调整  
            target_scale = 1.1  # 不能是1，当长度比原文短也可能缩小字体，还没找到原因。
            print("-" * 50)  
            logger.debug(f"翻译文本不长于原文 ({char_count_trans} <= {char_count_orig}) 或原文长度为0，不应用长度比例缩放。目标缩放因子: {target_scale:.2f}") 


        # 4. 根据目标字体大小和长度比例（如果应用）计算最终的缩放因子  
        # 我们需要一个缩放因子来调整原始边界框，使其能够容纳新的字体大小和可能的更长文本  
        # 一个简单的方法是综合考虑字体大小变化和长度比例  
        # 如果原始字体大小有效且目标字体大小不同，首先考虑字体大小的缩放  
        font_size_scale = target_font_size / original_region_font_size if original_region_font_size > 0 else 1.0  
        # 如果应用了长度比例缩放，取字体大小缩放和长度比例缩放的最大值  
        # 这样可以确保区域至少能容纳更长的文本或更大的字体  
        final_scale = max(font_size_scale, target_scale) # 使用之前计算的target_scale (考虑长度比例)  
        # 确保最终缩放因子至少为 1.0  
        final_scale = max(1.0, final_scale)  

        logger.debug(f"字体大小缩放因子: {font_size_scale:.2f}")  
        logger.debug(f"最终边界框缩放因子: {final_scale:.2f}")  


        # 5. 缩放边框，旋转回，并裁剪  
        if final_scale > 1.001:  
            logger.debug(f"需要缩放边框: 文本='{region.translation}', 缩放={final_scale:.2f}")  
            try:  
                # 使用 unrotated_min_rect 进行缩放  
                poly = Polygon(region.unrotated_min_rect[0])  
                # 从中心缩放  
                poly = affinity.scale(poly, xfact=final_scale, yfact=final_scale, origin='center')  
                scaled_unrotated_points = np.array(poly.exterior.coords[:4])  

                # 将缩放后的点旋转回原始方向  
                # 使用 to_int=False 以保留精度进行裁剪  
                dst_points = rotate_polygons(region.center, scaled_unrotated_points.reshape(1, -1), -region.angle, to_int=False).reshape(-1, 4, 2)  

                # 将坐标裁剪到图像边界内  
                # 使用 img.shape[1]-1 和 img.shape[0]-1 避免 off-by-one 问题  
                dst_points[..., 0] = dst_points[..., 0].clip(0, img.shape[1] - 1)  
                dst_points[..., 1] = dst_points[..., 1].clip(0, img.shape[0] - 1)  

                # 裁剪后转换为 int64  
                dst_points = dst_points.astype(np.int64)  

                # 检查最终形状 (以防万一)  
                dst_points = dst_points.reshape((-1, 4, 2))  
                logger.debug(f"计算缩放后的 dst_points 完成。")  

            except Exception as e:  
                # 如果在缩放/旋转几何形状时发生错误，使用原始的 min_rect  
                logger.error(f"缩放/旋转几何形状时出错对于文本 '{region.translation}': {e}. 使用原始 min_rect。")  
                dst_points = region.min_rect # 错误时使用原始值  
        else:  
            # 无需显著缩放，使用原始的 min_rect  
            logger.debug(f"无需显著缩放对于文本 '{region.translation}'. 使用原始 min_rect。")  
            dst_points = region.min_rect  

        # 6. 存储最终的 dst_points 并更新 region 的字体大小  
        dst_points_list.append(dst_points)  
        region.font_size = int(target_font_size) # 将 TextBlock 的字体大小更新为计算出的目标字体大小  

    return dst_points_list  

async def dispatch(
    img: np.ndarray,
    text_regions: List[TextBlock],
    font_path: str = '',
    font_size_fixed: int = None,
    font_size_offset: int = 0,
    font_size_minimum: int = 0,
    hyphenate: bool = True,
    render_mask: np.ndarray = None,
    line_spacing: int = None,
    disable_font_border: bool = False,
    config: Config = None
    ) -> np.ndarray:

    text_render.set_font(font_path)
    text_regions = list(filter(lambda region: region.translation, text_regions))

    # Resize regions that are too small
    dst_points_list = resize_regions_to_font_size(img, text_regions, font_size_fixed, font_size_offset, font_size_minimum)

    # TODO: Maybe remove intersections

    # Render text
    for region, dst_points in tqdm(zip(text_regions, dst_points_list), '[render]', total=len(text_regions)):
        if render_mask is not None:
            # set render_mask to 1 for the region that is inside dst_points
            cv2.fillConvexPoly(render_mask, dst_points.astype(np.int32), 1)
        img = render(img, region, dst_points, hyphenate, line_spacing, disable_font_border, config)
    return img

def render(
    img,
    region: TextBlock,
    dst_points,
    hyphenate,
    line_spacing,
    disable_font_border,
    config: Config
):
    fg, bg = region.get_font_colors()
    fg, bg = fg_bg_compare(fg, bg)
    if disable_font_border:
        bg = None

    middle_pts = (dst_points[:, [1, 2, 3, 0]] + dst_points) / 2
    norm_h = np.linalg.norm(middle_pts[:, 1] - middle_pts[:, 3], axis=1)
    norm_v = np.linalg.norm(middle_pts[:, 2] - middle_pts[:, 0], axis=1)
    r_orig = np.mean(norm_h / norm_v)

    # 如果配置中设定了非自动模式，则直接使用配置决定方向
    forced_direction = region._direction if hasattr(region, "_direction") else region.direction
    if forced_direction != "auto":
        if forced_direction in ["horizontal", "h"]:
            render_horizontally = True
        elif forced_direction in ["vertical", "v"]:
            render_horizontally = False
        else:
            render_horizontally = region.horizontal
    else:
        render_horizontally = region.horizontal

    if render_horizontally:
        temp_box = text_render.put_text_horizontal(
            region.font_size,
            region.get_translation_for_rendering(),
            round(norm_h[0]),
            round(norm_v[0]),
            region.alignment,
            region.direction == 'hl',  # 强制水平排版
            fg,
            bg,
            region.target_lang,
            hyphenate,
            line_spacing,
        )
    else:
        temp_box = text_render.put_text_vertical(
            region.font_size,
            region.get_translation_for_rendering(),
            round(norm_v[0]),
            region.alignment,
            fg,
            bg,
            line_spacing,
            config
        )
    h, w, _ = temp_box.shape
    r_temp = w / h

    # # Extend temporary box so that it has same ratio as original
    # if r_temp > r_orig:
        # h_ext = int(w / (2 * r_orig) - h / 2)
        # box = np.zeros((h + h_ext * 2, w, 4), dtype=np.uint8)
        # box[h_ext:h + h_ext, 0:w] = temp_box
    # else:
        # w_ext = int((h * r_orig - w) / 2)
        # box = np.zeros((h, w + w_ext * 2, 4), dtype=np.uint8)
        # box[0:h, w_ext:w_ext+w] = temp_box

    # --- Start Modification ---
    box = None  
    #print("\n" + "="*50)  
    #print(f"Processing text: \"{region.get_translation_for_rendering()}\"")  
    #print(f"Text direction: {'Horizontal' if region.horizontal else 'Vertical'}")  
    #print(f"Font size: {region.font_size}, Alignment: {region.alignment}")  
    #print(f"Target language: {region.target_lang}")      
    #print(f"Region horizontal: {region.horizontal}")  
    #print(f"Starting image adjustment: r_temp={r_temp}, r_orig={r_orig}, h={h}, w={w}")  
    if region.horizontal:  
        #print("Processing HORIZONTAL region")  
        
        if r_temp > r_orig:   
            #print(f"Case: r_temp({r_temp}) > r_orig({r_orig}) - Need vertical padding")  
            h_ext = int((w / r_orig - h) // 2) if r_orig > 0 else 0  
            #print(f"Calculated h_ext = {h_ext}")  
            
            if h_ext >= 0:  
                #print(f"Creating new box with dimensions: {h + h_ext * 2}x{w}")  
                box = np.zeros((h + h_ext * 2, w, 4), dtype=np.uint8)  
                #print(f"Placing temp_box at position [h_ext:h_ext+h, :w] = [{h_ext}:{h_ext+h}, 0:{w}]")  
                # 列已排满，行居中
                box[h_ext:h_ext+h, 0:w] = temp_box  
            else:  
                #print("h_ext < 0, using original temp_box")  
                box = temp_box.copy()  
        else:   
            #print(f"Case: r_temp({r_temp}) <= r_orig({r_orig}) - Need horizontal padding")  
            w_ext = int((h * r_orig - w) // 2)  
            #print(f"Calculated w_ext = {w_ext}")  
            
            if w_ext >= 0:  
                #print(f"Creating new box with dimensions: {h}x{w + w_ext * 2}")  
                box = np.zeros((h, w + w_ext * 2, 4), dtype=np.uint8)  
                #print(f"Placing temp_box at position [:, :w] = [0:{h}, 0:{w}]")  
                # 当前气泡检测的问题：
                # 1.对纯色背景的误判（核心问题）：
                # 原因：代码计算文本框边缘2像素区域的黑/白像素比例。如果文本框在一个大的纯白背景上（比如白纸黑字），边缘绝大部分是白色，ratio会很低（接近0），低于ignore_bubble阈值。代码会认为这是一个“正常的白色气泡背景”，从而错误地不忽略它（即，认为它是需要翻译的普通气泡内文字）。虽然这些文字确实要翻译，但是它们并不是气泡内文字。
                # 根本缺陷：这种方法没有检测气泡的边界/轮廓，只是在检查局部背景色。
                # 2.无法识别气泡边界：
                # 原因：代码不涉及任何形状或轮廓检测。它不知道是否存在一个封闭的、颜色相对均匀的线条包围着文本框。
                # 后果：无法区分真正的气泡（有边界）和仅仅是背景颜色恰好符合比例的情况。
                # 3.对气泡大小和相对位置不敏感：
                # 原因：只看紧邻的2像素，不考虑气泡整体的大小、形状，以及文本框在气泡内的相对位置。
                # 后果：无法利用“气泡通常会包围文本框，且大小适中。”这一常识性特征。
                # 4.连通气泡问题：
                # 原因：当前逻辑完全基于单个文本框的局部环境，无法感知是否存在一个跨越多个文本框的共享气泡结构。
                # 后果：无法处理一个大或形状复杂的气泡包含多个独立文本块的情况，也无法判断哪个文本块对应气泡的哪一部分。
            
                # 行已排满，文字左侧不留空列，否则当存在多个文本框的左边线处于一条线上时译后文本无法对齐。常见场景：无框漫画、漫画后记
                # 当前页面存在气泡时则可改为居中：box[0:h, w_ext:w_ext+w] = temp_box，需更准确的气泡检测。              
                box[0:h, 0:w] = temp_box  
            else:  
                #print("w_ext < 0, using original temp_box")  
                box = temp_box.copy()  
    else:  
        #print("Processing VERTICAL region")  
        
        if r_temp > r_orig:   
            #print(f"Case: r_temp({r_temp}) > r_orig({r_orig}) - Need vertical padding")  
            h_ext = int(w / (2 * r_orig) - h / 2) if r_orig > 0 else 0   
            #print(f"Calculated h_ext = {h_ext}")  
            
            if h_ext >= 0:   
                #print(f"Creating new box with dimensions: {h + h_ext * 2}x{w}")  
                box = np.zeros((h + h_ext * 2, w, 4), dtype=np.uint8)  
                #print(f"Placing temp_box at position [0:h, 0:w] = [0:{h}, 0:{w}]")  
                # 列已排满，文字的上方不留空行，否则当存在多个文本框的上边线在一条线上时文本无法对齐，常见场景：无框漫画、CG
                # 当前页面存在气泡时则可改为居中：box[h_ext:h_ext+h, 0:w] = temp_box，需更准确的气泡检测。
                box[0:h, 0:w] = temp_box  
            else:   
                #print("h_ext < 0, using original temp_box")  
                box = temp_box.copy()   
        else:   
            #print(f"Case: r_temp({r_temp}) <= r_orig({r_orig}) - Need horizontal padding")  
            w_ext = int((h * r_orig - w) / 2)  
            #print(f"Calculated w_ext = {w_ext}")  
            
            if w_ext >= 0:  
                #print(f"Creating new box with dimensions: {h}x{w + w_ext * 2}")  
                box = np.zeros((h, w + w_ext * 2, 4), dtype=np.uint8)  
                #print(f"Placing temp_box at position [0:h, w_ext:w_ext+w] = [0:{h}, {w_ext}:{w_ext+w}]") 
                # 行已排满，列居中                
                box[0:h, w_ext:w_ext+w] = temp_box  
            else:   
                #print("w_ext < 0, using original temp_box")  
                box = temp_box.copy()   

    #print(f"Final box dimensions: {box.shape if box is not None else 'None'}")  

    src_points = np.array([[0, 0], [box.shape[1], 0], [box.shape[1], box.shape[0]], [0, box.shape[0]]]).astype(np.float32)
    M, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    rgba_region = cv2.warpPerspective(box, M, (img.shape[1], img.shape[0]),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=0)
    x, y, w, h = cv2.boundingRect(dst_points.astype(np.int32))
    canvas_region = rgba_region[y:y+h, x:x+w, :3]
    mask_region = rgba_region[y:y+h, x:x+w, 3:4].astype(np.float32) / 255.0
    img[y:y+h, x:x+w] = np.clip((img[y:y+h, x:x+w].astype(np.float32) * (1 - mask_region) +
                                 canvas_region.astype(np.float32) * mask_region), 0, 255).astype(np.uint8)
    return img

async def dispatch_eng_render(img_canvas: np.ndarray, original_img: np.ndarray, text_regions: List[TextBlock], font_path: str = '', line_spacing: int = 0, disable_font_border: bool = False) -> np.ndarray:
    if len(text_regions) == 0:
        return img_canvas

    if not font_path:
        font_path = os.path.join(BASE_PATH, 'fonts/comic shanns 2.ttf')
    text_render.set_font(font_path)

    return render_textblock_list_eng(img_canvas, text_regions, line_spacing=line_spacing, size_tol=1.2, original_img=original_img, downscale_constraint=0.8,disable_font_border=disable_font_border)
