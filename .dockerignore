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
from ..utils.bubble import is_ignore, check_color  
from ..utils import (
    BASE_PATH,
    TextBlock,
    color_difference,
    get_logger,
    rotate_polygons,
)

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

# def resize_regions_to_font_size(img: np.ndarray, text_regions: List[TextBlock], font_size_fixed: int, font_size_offset: int, font_size_minimum: int):
    # if font_size_minimum == -1:
        # # Automatically determine font_size by image size
        # font_size_minimum = round((img.shape[0] + img.shape[1]) / 200)
    # logger.debug(f'font_size_minimum {font_size_minimum}')

    # dst_points_list = []
    # for region in text_regions:
        # char_count_orig = len(region.text)
        # char_count_trans = len(region.translation.strip())
        # if char_count_trans > char_count_orig:
            # # More characters were added, have to reduce fontsize to fit allotted area
            # # print('count', char_count_trans, region.font_size)
            # rescaled_font_size = region.font_size
            # while True:
                # rows = region.unrotated_size[0] // rescaled_font_size
                # cols = region.unrotated_size[1] // rescaled_font_size
                # if rows * cols >= char_count_trans:
                    # # print(rows, cols, rescaled_font_size, rows * cols, char_count_trans)
                    # # print('rescaled', rescaled_font_size)
                    # region.font_size = rescaled_font_size
                    # break
                # rescaled_font_size -= 1
                # if rescaled_font_size <= 0:
                    # break
        # # Otherwise no need to increase fontsize

        # # Infer the target fontsize
        # target_font_size = region.font_size
        # if font_size_fixed is not None:
            # target_font_size = font_size_fixed
        # elif target_font_size < font_size_minimum:
            # target_font_size = max(region.font_size, font_size_minimum)
        # target_font_size += font_size_offset

        # # Rescale dst_points accordingly
        # if target_font_size != region.font_size:
            # target_scale = target_font_size / region.font_size
            # dst_points = region.unrotated_min_rect[0]
            # poly = Polygon(region.unrotated_min_rect[0])
            # poly = affinity.scale(poly, xfact=target_scale, yfact=target_scale)
            # dst_points = np.array(poly.exterior.coords[:4])
            # dst_points = rotate_polygons(region.center, dst_points.reshape(1, -1), -region.angle).reshape(-1, 4, 2)

            # # Clip to img width and height
            # dst_points[..., 0] = dst_points[..., 0].clip(0, img.shape[1])
            # dst_points[..., 1] = dst_points[..., 1].clip(0, img.shape[0])

            # dst_points = dst_points.reshape((-1, 4, 2))
            # region.font_size = int(target_font_size)
        # else:
            # dst_points = region.min_rect

        # dst_points_list.append(dst_points)
    # return dst_points_list

def resize_regions_to_font_size(img: np.ndarray, text_regions: List['TextBlock'], font_size_fixed: int, font_size_offset: int, font_size_minimum: int):
    # 1. 确定最小字号
    if font_size_minimum == -1:
        # Automatically determine font_size by image size
        font_size_minimum = round((img.shape[0] + img.shape[1]) / 200)
    logger.debug(f'font_size_minimum {font_size_minimum}')
    # 确保最小字号至少是 1 像素。（添加了安全措施）
    font_size_minimum = max(1, font_size_minimum)

    dst_points_list = []
    for region in text_regions:
        # 存储本次循环开始前，该区域的原始字号
        original_region_font_size = region.font_size

        # 2. 如果翻译后字数变多，可能需要缩小字号
        char_count_orig = len(region.text)
        char_count_trans = len(region.translation.strip())

        # 这个代码块仅在翻译后字数变多且原始字号有效时执行
        if char_count_trans > char_count_orig and original_region_font_size > 0:
            rescaled_font_size = original_region_font_size
            font_size_before_reduction = original_region_font_size # 记录缩小前的字号

            # --- 修正开始 ---
            # 获取区域的几何宽度和高度 (注意：基于旋转后的 min_rect 计算)
            width, height = region.unrotated_size
            # --- 修正结束 ---

            while True:
                # 在除法前检查，防止除零错误
                if rescaled_font_size <= 0:
                     logger.warning(f"为文本 '{region.translation[:10]}...' 尝试缩小时字号变为零。恢复缩小前的字号。")
                     # 如果字号变无效，恢复到缩小前的大小并停止尝试
                     region.font_size = font_size_before_reduction
                     # 需要跳出 while 循环，并且让后续逻辑使用 font_size_before_reduction
                     # 重置 rescaled_font_size 以便后续 target_font_size 计算正确
                     rescaled_font_size = font_size_before_reduction # 这行其实可以省略，因为 region.font_size 会被用于后续计算
                     break # 退出 while 循环

                # --- 修正开始 ---
                # 根据文本方向正确估算行数和列数
                if region.horizontal:
                    # 水平文本：行数基于高度，列数基于宽度
                    est_rows = max(1, height // rescaled_font_size) # 至少 1 行
                    est_cols = max(1, width // rescaled_font_size)  # 至少 1 列
                else: # region.vertical
                    # 垂直文本：列数基于宽度，行数基于高度
                    est_cols = max(1, width // rescaled_font_size)  # 至少 1 列
                    est_rows = max(1, height // rescaled_font_size) # 至少 1 行

                logger.debug(f"尝试字号 {rescaled_font_size}: 估算 行={est_rows}, 列={est_cols} (基于宽={width}, 高={height}, 方向={'H' if region.horizontal else 'V'})")

                # 使用修正后的估算进行比较
                if est_rows * est_cols >= char_count_trans:
                    # 近似计算表明，这个字号 *也许* 能放下。
                    # 更新区域的字号，供 *本次迭代* 后续步骤使用。
                    region.font_size = rescaled_font_size
                    logger.debug(f"根据修正估算，将 '{region.translation[:10]}...' 的字号从 {font_size_before_reduction} 缩小到 {rescaled_font_size}。")
                    break # 退出 while 循环
                # --- 修正结束 ---

                # 如果估算认为放不下，缩小字号再试
                rescaled_font_size -= 1
                # while True 会自动继续，并在下一次循环开始时检查 rescaled_font_size <= 0

        # 如果翻译后字数没有变多，region.font_size 保持为 original_region_font_size

        # 3. 确定最终的目标字号（使用可能已被缩小的字号作为基础）
        current_base_font_size = region.font_size # 这是原始字号或缩小后的字号

        # --- 修正开始：优化目标字号计算逻辑，使其更清晰 ---
        if font_size_fixed is not None:
            # 如果指定了固定字号，则使用固定字号
            target_font_size = font_size_fixed
        else:
            # 否则，应用字号偏移量
            target_font_size = current_base_font_size + font_size_offset

        # 应用最小字号限制
        target_font_size = max(target_font_size, font_size_minimum)
        # 再次确保字号至少为 1 像素。
        target_font_size = max(1, target_font_size)
        # --- 修正结束 ---

        logger.debug(f"文本 '{region.translation[:10]}...': 基础字号={current_base_font_size}, 固定值={font_size_fixed}, 偏移量={font_size_offset}, 最小值={font_size_minimum} -> 目标字号={target_font_size}")

        # 4. 如果字号改变，重新计算几何形状 (`dst_points`)
        # 比较 *最终* 的 target_font_size 和这个区域 *原始* 的字号
        size_basis_for_geometry = original_region_font_size

        # 检查是否真的需要缩放
        if target_font_size != size_basis_for_geometry:
            # --- 修正开始：添加除零检查和错误处理 ---
            if size_basis_for_geometry <= 0: # 防止除以零的安全措施
                 logger.warning(f"无法为文本 '{region.translation[:10]}...' 缩放几何形状，因为原始字号是 {size_basis_for_geometry}。使用原始 min_rect。")
                 dst_points = region.min_rect # 回退到使用原始的旋转后坐标点
            else:
                try:
                    # 计算缩放比例
                    target_scale = target_font_size / size_basis_for_geometry
                    logger.debug(f"需要缩放：目标={target_font_size}, 基础={size_basis_for_geometry}, 比例={target_scale}")

                    # 使用未旋转的矩形进行缩放，以防止倾斜
                    poly = Polygon(region.unrotated_min_rect[0])
                    # 从多边形的中心点进行缩放
                    poly = affinity.scale(poly, xfact=target_scale, yfact=target_scale, origin='center')
                    scaled_unrotated_points = np.array(poly.exterior.coords[:4])

                    # 将缩放后的点旋转回原始方向
                    dst_points = rotate_polygons(region.center, scaled_unrotated_points.reshape(1, -1), -region.angle).reshape(-1, 4, 2)

                    # 将坐标裁剪到图像边界内
                    dst_points[..., 0] = dst_points[..., 0].clip(0, img.shape[1])
                    dst_points[..., 1] = dst_points[..., 1].clip(0, img.shape[0])

                    # 确保最终形状正确（通常是多余的，但安全）
                    dst_points = dst_points.reshape((-1, 4, 2))
                    logger.debug(f"已计算缩放后的 dst_points。")

                except Exception as e:
                    # 如果几何缩放过程中出错，记录错误并回退
                    logger.error(f"为文本 '{region.translation[:10]}...' 进行几何缩放时出错: {e}。使用原始 min_rect。")
                    dst_points = region.min_rect # 出错时回退
            # --- 修正结束 ---
        else:
            # 相对于原始几何基础，字号没有变化，使用原始的旋转后坐标点
            logger.debug(f"无需缩放：目标字号={target_font_size} == 原始字号={size_basis_for_geometry}。使用原始 min_rect。")
            dst_points = region.min_rect

        # 5. 将区域的 font_size 属性更新为 *最终* 计算出的目标大小（非常重要！）
        region.font_size = int(target_font_size)

        # 6. 存储最终计算出的目标坐标点
        dst_points_list.append(dst_points)

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
    disable_font_border: bool = False
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
        img = render(img, region, dst_points, hyphenate, line_spacing, disable_font_border)
    return img

def render(
    img,
    region: TextBlock,
    dst_points,
    hyphenate,
    line_spacing,
    disable_font_border
):
    fg, bg = region.get_font_colors()
    fg, bg = fg_bg_compare(fg, bg)

    if disable_font_border :
        bg = None

    middle_pts = (dst_points[:, [1, 2, 3, 0]] + dst_points) / 2
    norm_h = np.linalg.norm(middle_pts[:, 1] - middle_pts[:, 3], axis=1)
    norm_v = np.linalg.norm(middle_pts[:, 2] - middle_pts[:, 0], axis=1)
    r_orig = np.mean(norm_h / norm_v)

    if region.horizontal:
        temp_box = text_render.put_text_horizontal(
            region.font_size,
            region.get_translation_for_rendering(),
            round(norm_h[0]),
            round(norm_v[0]),
            region.alignment,
            region.direction == 'hr',
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
            region.target_lang
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
                # 后果：无法处理一个大或形状复杂的气泡包含多个独立文本块的情况，也无法判断哪个文本块对应气泡的哪一部分

                # 行已排满，文字左侧不留空列，否则当存在多个文本框的左边线处于一条线上时译后文本无法对齐，搭配左对齐排版更美观。常见场景：无框漫画、漫画后记
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
                #box[h_ext:h_ext+h, 0:w] = temp_box
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


    # box = None # Initialize box to None  

    # print("\n" + "="*50)  
    # print(f"Processing text: \"{region.get_translation_for_rendering()}\"")  
    # print(f"Text direction: {'Horizontal' if region.horizontal else 'Vertical'}")  
    # print(f"Font size: {region.font_size}, Alignment: {region.alignment}")  
    # print(f"Target language: {region.target_lang}")      
    # print(f"Region horizontal: {region.horizontal}")  

    # # 检测当前区域是否为气泡  
    # # 当前的气泡检测如果文本框四周为纯色，会检测有误，造成排版有误，颜色检测也有概率因为噪点产生问题
    # # 可以使用之前的is_ignore函数或专门的气泡检测函数  
    # # 使用TextBlock的min_rect或xyxy属性从原图中裁剪区域  
    # x1, y1, x2, y2 = region.xyxy  
    # region_image = img[y1:y2, x1:x2]  # 需要原始图像img  
    # is_bubble = not is_ignore(region_image, ignore_bubble=10)      
    # print(f"Is bubble region: {is_bubble}")  
    # print(f"Starting image adjustment: r_temp={r_temp}, r_orig={r_orig}, h={h}, w={w}")  
    
    # if region.horizontal:  
        # print("Processing HORIZONTAL region")  
        
        # if r_temp > r_orig:  
            # print(f"Case: r_temp({r_temp}) > r_orig({r_orig}) - Need vertical padding")  
            # h_ext = int((w / r_orig - h) // 2) if r_orig > 0 else 0  
            # print(f"Calculated h_ext = {h_ext}")  
            
            # if h_ext >= 0:  
                # print(f"Creating new box with dimensions: {h + h_ext * 2}x{w}")  
                # box = np.zeros((h + h_ext * 2, w, 4), dtype=np.uint8)  
                # print(f"Placing temp_box at position [h_ext:h_ext+h, :w] = [{h_ext}:{h_ext+h}, 0:{w}]")  
                # # 列已排满，行居中  
                # box[h_ext:h_ext+h, 0:w] = temp_box  
            # else:  
                # print("h_ext < 0, using original temp_box")  
                # box = temp_box.copy()  
        # else:  
            # print(f"Case: r_temp({r_temp}) <= r_orig({r_orig}) - Need horizontal padding")  
            # w_ext = int((h * r_orig - w) // 2)  
            # print(f"Calculated w_ext = {w_ext}")  
            
            # if w_ext >= 0:  
                # print(f"Creating new box with dimensions: {h}x{w + w_ext * 2}")  
                # box = np.zeros((h, w + w_ext * 2, 4), dtype=np.uint8)  
                # print(f"Placing temp_box at position - Using {'centered' if is_bubble else 'left-aligned'} positioning")  
                
                # if is_bubble:  
                    # # 气泡内文字居中  
                    # print(f"Bubble detected - centering text: [0:{h}, {w_ext}:{w_ext+w}]")  
                    # box[0:h, w_ext:w_ext+w] = temp_box  
                # else:  
                    # # 无框漫画、漫画后记等场景左对齐  
                    # print(f"No bubble detected - left aligning text: [0:{h}, 0:{w}]")  
                    # box[0:h, 0:w] = temp_box  
            # else:  
                # print("w_ext < 0, using original temp_box")  
                # box = temp_box.copy()  
    # else:  
        # print("Processing VERTICAL region")  
        
        # if r_temp > r_orig:  
            # print(f"Case: r_temp({r_temp}) > r_orig({r_orig}) - Need vertical padding")  
            # h_ext = int(w / (2 * r_orig) - h / 2) if r_orig > 0 else 0  
            # print(f"Calculated h_ext = {h_ext}")  
            
            # if h_ext >= 0:  
                # print(f"Creating new box with dimensions: {h + h_ext * 2}x{w}")  
                # box = np.zeros((h + h_ext * 2, w, 4), dtype=np.uint8)  
                # print(f"Placing temp_box position - Using {'centered' if is_bubble else 'top-aligned'} positioning")  
                
                # if is_bubble:  
                    # # 气泡内文字垂直居中  
                    # print(f"Bubble detected - centering text: [{h_ext}:{h_ext+h}, 0:{w}]")  
                    # box[h_ext:h_ext+h, 0:w] = temp_box  
                # else:  
                    # # 无框漫画、CG等场景顶部对齐  
                    # print(f"No bubble detected - top aligning text: [0:{h}, 0:{w}]")  
                    # box[0:h, 0:w] = temp_box  
            # else:  
                # print("h_ext < 0, using original temp_box")  
                # box = temp_box.copy()  
        # else:  
            # print(f"Case: r_temp({r_temp}) <= r_orig({r_orig}) - Need horizontal padding")  
            # w_ext = int((h * r_orig - w) / 2)  
            # print(f"Calculated w_ext = {w_ext}")  
            
            # if w_ext >= 0:  
                # print(f"Creating new box with dimensions: {h}x{w + w_ext * 2}")  
                # box = np.zeros((h, w + w_ext * 2, 4), dtype=np.uint8)  
                # print(f"Placing temp_box at position [0:h, w_ext:w_ext+w] = [0:{h}, {w_ext}:{w_ext+w}]")  
                # # 行已排满，列居中  
                # box[0:h, w_ext:w_ext+w] = temp_box  
            # else:  
                # print("w_ext < 0, using original temp_box")  
                # box = temp_box.copy()  

    # print(f"Final box dimensions: {box.shape if box is not None else 'None'}")  


    # --- End Modification ---

    src_points = np.array([[0, 0], [box.shape[1], 0], [box.shape[1], box.shape[0]], [0, box.shape[0]]]).astype(np.float32)
    #src_pts[:, 0] = np.clip(np.round(src_pts[:, 0]), 0, enlarged_w * 2)
    #src_pts[:, 1] = np.clip(np.round(src_pts[:, 1]), 0, enlarged_h * 2)

    M, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    rgba_region = cv2.warpPerspective(box, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    x, y, w, h = cv2.boundingRect(dst_points.astype(np.int32))
    canvas_region = rgba_region[y:y+h, x:x+w, :3]
    mask_region = rgba_region[y:y+h, x:x+w, 3:4].astype(np.float32) / 255.0
    img[y:y+h, x:x+w] = np.clip((img[y:y+h, x:x+w].astype(np.float32) * (1 - mask_region) + canvas_region.astype(np.float32) * mask_region), 0, 255).astype(np.uint8)
    return img

async def dispatch_eng_render(img_canvas: np.ndarray, original_img: np.ndarray, text_regions: List[TextBlock], font_path: str = '', line_spacing: int = 0, disable_font_border: bool = False) -> np.ndarray:
    if len(text_regions) == 0:
        return img_canvas

    if not font_path:
        font_path = os.path.join(BASE_PATH, 'fonts/comic shanns 2.ttf')
    text_render.set_font(font_path)

    return render_textblock_list_eng(img_canvas, text_regions, line_spacing=line_spacing, size_tol=1.2, original_img=original_img, downscale_constraint=0.8,disable_font_border=disable_font_border)
