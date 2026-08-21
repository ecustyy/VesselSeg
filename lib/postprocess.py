import numpy as np
from scipy.ndimage import distance_transform_edt
import cv2


def connect_vessel_by_direction(segmentation_mask, distance_threshold=5, angle_threshold=22.5):
    """
    基于方向一致性连接断裂的血管

    参数:
    ----------
    segmentation_mask : np.ndarray
        血管分割结果，二维数组，值在0-1之间
    distance_threshold : int
        连接的最大距离阈值（像素）
    angle_threshold : float
        方向一致性的角度阈值（度）

    返回:
    ----------
    np.ndarray : 优化后的分割结果
    """
    # 二值化分割结果（阈值可根据实际情况调整）
    binary_mask = (segmentation_mask > 0.5).astype(np.uint8)
    clean_mask = binary_mask.copy()

    # 计算距离变换，找到潜在的连接点
    dist_map = distance_transform_edt(1 - clean_mask)
    # 找到距离血管在一定阈值内的非血管点
    connect_points = (dist_map < distance_threshold) & (clean_mask == 0)

    if not connect_points.any():
        return segmentation_mask.astype(np.float32)

    # 获取连接点坐标
    yy, xx = np.where(connect_points)

    # 计算血管骨架，用于更准确的方向估计
    # skeleton = cv2.ximgproc.thinning(clean_mask)

    # 计算方向场
    direction_field = compute_direction_field(clean_mask)

    to_connect = []

    for y, x in zip(yy, xx):
        # 获取更大邻域内的血管点
        search_radius = min(10, max(clean_mask.shape) // 20)
        vessel_points = []

        # 在搜索半径内寻找血管点
        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < clean_mask.shape[0] and 0 <= nx < clean_mask.shape[1]:
                    if clean_mask[ny, nx] and (dy != 0 or dx != 0):
                        distance = np.sqrt(dy ** 2 + dx ** 2)
                        if distance <= search_radius:
                            vessel_points.append((ny, nx, dy, dx))

        if len(vessel_points) < 2:
            continue

        # 计算连接点处可能的方向
        if check_direction_consistency(vessel_points, direction_field,
                                       (y, x), angle_threshold):
            to_connect.append((y, x))

    # 执行连接
    if to_connect:
        for y, x in to_connect:
            clean_mask[y, x] = 1

        # 可选：进行形态学操作平滑连接点
        kernel = np.ones((3, 3), np.uint8)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)

    return clean_mask.astype(np.float32)


def compute_direction_field(binary_mask):
    """
    计算血管方向场

    参数:
    ----------
    binary_mask : np.ndarray
        二值化的血管分割结果

    返回:
    ----------
    np.ndarray : 方向场，每个像素包含方向信息
    """
    height, width = binary_mask.shape
    direction_field = np.zeros((height, width), dtype=np.float32)

    # 计算梯度
    sobelx = cv2.Sobel(binary_mask.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(binary_mask.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)

    # 计算方向（角度）
    direction_field = np.arctan2(sobely, sobelx) * 180 / np.pi

    # 将角度转换到0-180度范围（血管方向无正反）
    direction_field = direction_field % 180

    return direction_field


def check_direction_consistency(vessel_points, direction_field,
                                connect_point, angle_threshold):
    """
    检查方向一致性

    参数:
    ----------
    vessel_points : list
        周围血管点列表，每个元素为(y, x, dy, dx)
    direction_field : np.ndarray
        方向场
    connect_point : tuple
        连接点坐标(y, x)
    angle_threshold : float
        角度阈值

    返回:
    ----------
    bool : 是否满足方向一致性
    """
    if len(vessel_points) < 2:
        return False

    y_conn, x_conn = connect_point

    # 收集所有血管点的方向
    directions = []
    weights = []

    for y_v, x_v, dy, dx in vessel_points:
        # 计算相对于连接点的方向
        direct_angle = np.arctan2(dy, dx) * 180 / np.pi
        direct_angle = direct_angle % 180  # 转换到0-180度

        # 使用方向场的方向（如果可用）
        if direction_field[y_v, x_v] > 0:
            field_angle = direction_field[y_v, x_v]
            # 结合两种方向信息
            final_angle = 0.7 * field_angle + 0.3 * direct_angle
        else:
            final_angle = direct_angle

        # 计算权重（距离越近权重越大）
        distance = np.sqrt(dy ** 2 + dx ** 2)
        weight = 1.0 / (distance + 1e-6)

        directions.append(final_angle)
        weights.append(weight)

    # 对方向进行聚类分析
    directions = np.array(directions)
    weights = np.array(weights)

    # 主方向分析
    if len(directions) >= 2:
        # 计算加权平均方向
        mean_angle = np.arctan2(
            np.sum(weights * np.sin(directions * np.pi / 180)),
            np.sum(weights * np.cos(directions * np.pi / 180))
        ) * 180 / np.pi
        mean_angle = mean_angle % 180

        # 计算方向一致性
        angle_diffs = np.abs(directions - mean_angle)
        # 处理角度循环性
        angle_diffs = np.minimum(angle_diffs, 180 - angle_diffs)

        # 计算加权平均差异
        weighted_diff = np.sum(weights * angle_diffs) / np.sum(weights)

        # 如果平均差异小于阈值，认为方向一致
        if weighted_diff < angle_threshold:
            return True

    return False


def enhanced_vessel_connection(segmentation_mask, distance_threshold=5,
                               angle_threshold=22.5, connectivity='8-neighbor'):
    """
    增强的血管连接函数，支持多种连接策略

    参数:
    ----------
    segmentation_mask : np.ndarray
        血管分割结果
    distance_threshold : int
        距离阈值
    angle_threshold : float
        角度阈值
    connectivity : str
        连接策略，'8-neighbor' 或 'linear'

    返回:
    ----------
    np.ndarray : 优化后的分割结果
    """
    binary_mask = (segmentation_mask > 0.5).astype(np.uint8)
    clean_mask = binary_mask.copy()

    # 计算距离变换
    dist_map = distance_transform_edt(1 - clean_mask)
    connect_points = (dist_map < distance_threshold) & (clean_mask == 0)

    if not connect_points.any():
        return segmentation_mask.astype(np.float32)

    yy, xx = np.where(connect_points)

    if connectivity == '8-neighbor':
        # 使用8邻域连接策略
        for y, x in zip(yy, xx):
            neighbors = []
            # 检查8邻域
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < clean_mask.shape[0] and 0 <= nx < clean_mask.shape[1]:
                        if clean_mask[ny, nx]:
                            neighbors.append((ny, nx, dy, dx))

            if len(neighbors) >= 2:
                # 简单方向检查
                angles = []
                for _, _, dy, dx in neighbors:
                    angle = np.arctan2(dy, dx) * 180 / np.pi
                    angle = angle % 180
                    angles.append(angle)

                # 检查方向一致性
                angles = np.array(angles)
                mean_angle = np.mean(angles)
                diffs = np.abs(angles - mean_angle)
                diffs = np.minimum(diffs, 180 - diffs)

                if np.mean(diffs) < angle_threshold:
                    clean_mask[y, x] = 1

    elif connectivity == 'linear':
        # 线性连接策略
        for y, x in zip(yy, xx):
            # 寻找两个最近的血管点
            vessel_points = []
            search_range = distance_threshold * 2

            for dy in range(-search_range, search_range + 1):
                for dx in range(-search_range, search_range + 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < clean_mask.shape[0] and 0 <= nx < clean_mask.shape[1]:
                        if clean_mask[ny, nx]:
                            distance = np.sqrt(dy ** 2 + dx ** 2)
                            angle = np.arctan2(dy, dx) * 180 / np.pi
                            vessel_points.append((distance, angle, ny, nx))

            if len(vessel_points) >= 2:
                # 按距离排序
                vessel_points.sort(key=lambda x: x[0])
                closest_points = vessel_points[:2]

                # 检查两个点是否在近似直线上
                _, angle1, _, _ = closest_points[0]
                _, angle2, _, _ = closest_points[1]

                angle_diff = np.abs(angle1 - angle2)
                angle_diff = min(angle_diff, 360 - angle_diff)

                # 如果两个点方向相反或近似相反，可能是断裂点
                if 150 < angle_diff < 210 or angle_diff < 30:
                    clean_mask[y, x] = 1

    # 后处理：细化和平滑
    kernel = np.ones((2, 2), np.uint8)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)

    return clean_mask.astype(np.float32)


# 使用示例
def process_vessel_segmentation(segmentation_result):
    """
    完整的血管分割后处理流程

    参数:
    ----------
    segmentation_result : np.ndarray
        模型预测的原始分割结果

    返回:
    ----------
    np.ndarray : 优化后的分割结果
    """
    # 步骤1: 二值化和初步清理
    binary_result = (segmentation_result > 0.5).astype(np.uint8)

    # 步骤2: 形态学操作去除小噪声
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary_result, cv2.MORPH_OPEN, kernel)

    # 步骤3: 连接断裂血管
    connected = connect_vessel_by_direction(cleaned, distance_threshold=6, angle_threshold=25)

    # 步骤4: 细化血管
    skeleton = cv2.ximgproc.thinning(connected)

    # 步骤5: 恢复血管宽度（可选）
    dilated = cv2.dilate(skeleton, kernel, iterations=1)

    return dilated.astype(np.float32)


# 快速连接函数（简化版本）
def quick_connect_vessels(segmentation_mask, max_gap=5):
    """
    快速血管连接函数

    参数:
    ----------
    segmentation_mask : np.ndarray
        分割结果
    max_gap : int
        最大断裂距离

    返回:
    ----------
    np.ndarray : 连接后的结果
    """
    binary_mask = (segmentation_mask > 0.5).astype(np.uint8)

    # 使用距离变换找到断裂点
    dist_map = distance_transform_edt(1 - binary_mask)

    # 找到可能的连接点
    gap_points = (dist_map <= max_gap) & (binary_mask == 0)

    # 标记需要连接的像素
    markers = np.zeros_like(binary_mask, dtype=np.uint8)
    markers[binary_mask == 1] = 1
    markers[gap_points] = 2

    # 使用分水岭算法连接
    if np.any(gap_points):
        # 计算距离作为前景标记
        dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)

        # 未知区域
        unknown = cv2.subtract(markers, sure_fg)

        # 标记标记
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        # 应用分水岭
        markers = cv2.watershed(cv2.cvtColor(binary_mask * 255, cv2.COLOR_GRAY2BGR), markers)
        binary_mask[markers > 1] = 1

    return binary_mask.astype(np.float32)