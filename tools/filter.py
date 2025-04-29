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
