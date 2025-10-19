import os
import json
import numpy as np
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import re
import argparse

def load_calibration(calib_file):
    """캘리브레이션 파일에서 변환 행렬과 카메라 정보를 로드"""
    with open(calib_file, 'r') as f:
        calib_data = json.load(f)
    
    # 카메라→라이다 변환 (calib.json에 저장된 형식)
    t_camera_lidar = calib_data['results']['T_lidar_camera']
    translation = np.array(t_camera_lidar[:3])
    quaternion = np.array(t_camera_lidar[3:])
    
    # 카메라→라이다 변환 행렬 생성
    rotation = Rotation.from_quat(quaternion).as_matrix()
    camera_to_lidar = np.eye(4)
    camera_to_lidar[:3, :3] = rotation
    camera_to_lidar[:3, 3] = translation
    
    # 라이다→카메라 변환은 역행렬
    lidar_to_camera = np.linalg.inv(camera_to_lidar)
    
    # 카메라 정보 추출
    camera_model = calib_data['camera']['camera_model']
    intrinsics = calib_data['camera']['intrinsics']
    
    return lidar_to_camera, camera_model, intrinsics

def load_pointcloud(pointcloud_file, max_distance=130.0):
    """포인트 클라우드 파일을 로드하고 필터링"""
    if pointcloud_file.endswith('.bin'):
        with open(pointcloud_file, 'rb') as f:
            data = f.read()
        
        xyz = np.frombuffer(data, dtype=np.float32).reshape(-1, 3)
    else:
        # Open3D 지원 형식
        pcd = o3d.io.read_point_cloud(pointcloud_file)
        if not pcd.has_points():
            return None, None, None
        xyz = np.asarray(pcd.points)
    
    # 유효한 포인트만 필터링
    mask = np.all(np.isfinite(xyz), axis=1) & np.all(np.abs(xyz) < 100.0, axis=1)
    xyz = xyz[mask]

    if np.isnan(xyz).any() or np.isinf(xyz).any():
        print(f"[NaN/Inf Error] {pointcloud_file}")
    
    # 거리 기반 필터링
    distances = np.linalg.norm(xyz, axis=1)
    distance_mask = distances < max_distance
    xyz = xyz[distance_mask]
    
    return xyz, distances[distance_mask]

def project_point_standard(point, width, height):
    """표준 equirectangular 투영 공식 사용 (개선된 수치 안정성)"""
    depth = np.linalg.norm(point)
    if depth < 1e-5:
        return None
    
    # z가 너무 0 근처면, phi = atan2(x, z) 가 급격히 요동할 수 있음
    if abs(point[2]) < 1e-6 and abs(point[0]) > 1e-6:
        return None  # 혹은 작은 바이어스 적용 가능
    
    # 표준 공식: 
    # longitude (φ) = arctan2(x, z)
    # latitude (θ) = arcsin(y/depth)
    phi = np.arctan2(point[0], point[2])  # x,z 사용 (전방 방향이 z축)
    theta = np.arcsin(np.clip(point[1] / depth, -1.0, 1.0))  # y는 상하 방향
    
    if np.isnan(phi) or np.isnan(theta):
        return None
    
    # 표준 변환:
    # u = (φ/(2π) + 0.5) * width
    # v = (0.5 - θ/π) * height
    u = ((phi / (2 * np.pi)) + 0.5) * width
    v = (0.5 - (theta / np.pi)) * height

    u_i = int(round(u))
    v_i = int(round(v))
    
    return u_i, v_i

def project_points_vectorized(points, lidar_to_camera, width, height):
    """벡터화된 포인트 투영 with 퍼센타일 게이팅"""
    if len(points) == 0:
        return [], []
    
    # 1. 변환 적용
    # 동차 좌표로 변환
    P = np.hstack([points, np.ones((len(points), 1), dtype=np.float32)])   # (N,4)
    Pc = (lidar_to_camera @ P.T).T[:, :3]                                 # (N,3)
    # Y축 반전 적용
    Pc = np.column_stack([Pc[:, 0], -Pc[:, 1], Pc[:, 2]])

    
    # 2. 거리 계산
    r = np.linalg.norm(Pc, axis=1)
    
    # 3. 프레임별 퍼센타일 게이팅 (이상치 제거)
    valid_r = r[np.isfinite(r) & (r > 1e-5)]
    if len(valid_r) > 0:
        r_lo = np.percentile(valid_r, 0.1)   # 0.1% 하위 제거
        r_hi = np.percentile(valid_r, 99.9)  # 99.9% 상위 제거
    
    # 4. 유효한 포인트 마스크
    mask = (np.isfinite(Pc).all(axis=1) & 
            (r > r_lo) & 
            (r < r_hi) & 
            (np.abs(Pc[:, 2]) >= 1e-6))  # z=0 근처 불안정성 회피
    
    # 5. 투영 좌표 계산
    valid_points = Pc[mask]
    valid_indices = np.where(mask)[0]
    projected_pixels = []
    
    for i, point in enumerate(valid_points):
        pixel = project_point_standard(point, width, height)
        if pixel is not None:
            u, v = pixel
            if 0 <= u < width and 0 <= v < height:
                projected_pixels.append((valid_indices[i], u, v))
    
    return projected_pixels, mask

def apply_y_flip(point):
    """Y축 반전 적용"""
    return np.array([point[0], -point[1], point[2]])

def extract_timestamp_from_filename(filename):
    """파일명에서 정확한 타임스탬프 추출"""
    # 확장자 제거
    name = os.path.splitext(os.path.basename(filename))[0]
    
    # 파일명이 숫자로만 구성된 경우 (ROS 타임스탬프)
    if name.isdigit():
        return int(name)
    
    # 숫자가 포함된 경우, 가장 긴 연속된 숫자를 타임스탬프로 간주
    numbers = re.findall(r'\d+', name)
    if numbers:
        # 가장 긴 숫자 문자열을 선택 (보통 타임스탬프가 가장 김)
        longest_number = max(numbers, key=len)
        return int(longest_number)
    
    return 0

def convert_ns_to_index(timestamp_ns, min_timestamp, max_timestamp, total_count):
    """나노초 타임스탬프를 인덱스로 변환"""
    if max_timestamp == min_timestamp:
        return 0
    normalized = (timestamp_ns - min_timestamp) / (max_timestamp - min_timestamp)
    return int(normalized * (total_count - 1))

def find_nearest_by_sequence(image_files, pointcloud_files):
    """시퀀스 기반 매칭 (타임스탬프가 다른 범위일 때)"""
    print("=== 시퀀스 기반 매칭 사용 ===")
    
    # 이미지와 포인트클라우드를 시간순으로 정렬
    img_with_ts = [(f, extract_timestamp_from_filename(f)) for f in image_files]
    pc_with_ts = [(f, extract_timestamp_from_filename(f)) for f in pointcloud_files]
    
    img_with_ts.sort(key=lambda x: x[1])
    pc_with_ts.sort(key=lambda x: x[1])
    
    # 이미지 개수에 맞춰 포인트클라우드 인덱스 계산
    matches = []
    for i, (img_file, img_ts) in enumerate(img_with_ts):
        # 이미지 인덱스를 포인트클라우드 인덱스로 매핑
        pc_idx = int((i / len(img_with_ts)) * len(pc_with_ts))
        if pc_idx >= len(pc_with_ts):
            pc_idx = len(pc_with_ts) - 1
        
        pc_file, pc_ts = pc_with_ts[pc_idx]
        matches.append((img_file, pc_file, abs(i - pc_idx)))
    
    return matches

def analyze_timestamps(image_files, pointcloud_files):
    """타임스탬프 분석 및 매칭 전략 결정"""
    print("=== 타임스탬프 분석 ===")
    
    # 이미지 타임스탬프 샘플
    img_timestamps = []
    for i, img_file in enumerate(image_files[:5]):  # 처음 5개만 분석
        ts = extract_timestamp_from_filename(img_file)
        img_timestamps.append(ts)
        print(f"Image {i+1}: {img_file} -> {ts}")
    
    print()
    
    # 포인트클라우드 타임스탬프 샘플
    pc_timestamps = []
    for i, pc_file in enumerate(pointcloud_files[:5]):  # 처음 5개만 분석
        ts = extract_timestamp_from_filename(pc_file)
        pc_timestamps.append(ts)
        print(f"PointCloud {i+1}: {os.path.basename(pc_file)} -> {ts}")
    
    print()
    
    # 타임스탬프 범위 분석
    all_img_ts = [extract_timestamp_from_filename(f) for f in image_files]
    all_pc_ts = [extract_timestamp_from_filename(f) for f in pointcloud_files]
    
    img_min, img_max = min(all_img_ts), max(all_img_ts)
    pc_min, pc_max = min(all_pc_ts), max(all_pc_ts)
    
    print(f"이미지 타임스탬프 범위: {img_min} ~ {img_max}")
    print(f"포인트클라우드 타임스탬프 범위: {pc_min} ~ {pc_max}")
    print(f"이미지 개수: {len(image_files)}, 포인트클라우드 개수: {len(pointcloud_files)}")
    
    # 오버랩 확인
    overlap_exists = not (img_max < pc_min or pc_max < img_min)
    print(f"타임스탬프 오버랩 존재: {overlap_exists}")
    
    # 타임스탬프 차이 분석
    if len(all_img_ts) > 1:
        img_intervals = [all_img_ts[i+1] - all_img_ts[i] for i in range(len(all_img_ts)-1)]
        avg_img_interval = np.mean(img_intervals)
        print(f"평균 이미지 간격: {avg_img_interval:.0f} ns")
    
    if len(all_pc_ts) > 1:
        pc_intervals = [all_pc_ts[i+1] - all_pc_ts[i] for i in range(len(all_pc_ts)-1)]
        avg_pc_interval = np.mean(pc_intervals)
        print(f"평균 포인트클라우드 간격: {avg_pc_interval:.0f} ns")
    
    # 매칭 전략 결정
    if overlap_exists:
        strategy = "timestamp"
        print("전략: 타임스탬프 기반 매칭")
    else:
        strategy = "sequence"
        print("전략: 시퀀스 기반 매칭 (타임스탬프 범위가 다름)")
    
    print("=" * 40)
    
    return all_img_ts, all_pc_ts, strategy

def find_nearest_pointcloud_optimized(image_timestamp, pointcloud_files, pc_timestamps_cache=None):
    """최적화된 가장 가까운 포인트클라우드 찾기"""
    if pc_timestamps_cache is None:
        pc_timestamps = [extract_timestamp_from_filename(f) for f in pointcloud_files]
    else:
        pc_timestamps = pc_timestamps_cache
    
    # 가장 가까운 타임스탬프 찾기
    diffs = [abs(image_timestamp - pc_ts) for pc_ts in pc_timestamps]
    min_idx = np.argmin(diffs)
    
    return pointcloud_files[min_idx], diffs[min_idx]

def project_lidar_to_image(pc_file, img_file, lidar_to_camera, width, height, output_dir):
    """캘리브레이션을 사용한 라이다 포인트 투영 (개선된 버전)"""
    pc_name = os.path.basename(pc_file).replace('.bin', '')
    img_name = os.path.basename(img_file).replace('.png', '')
    
    points, distances = load_pointcloud(pc_file)
    if points is None or len(distances) == 0:
        print(f"포인트클라우드 로드 실패: {pc_file}")
        return None
    
    image = cv2.imread(img_file)
    if image is None:
        print(f"이미지 로드 실패: {img_file}")
        return None
    
    result_image = image.copy()
    
    # 거리 통계 계산 (안정성 개선)
    min_dist = float(np.min(distances))
    max_dist = float(np.max(distances))
    mean_dist = float(np.mean(distances))
    std_dist = float(np.std(distances))
    
    print(f"포인트클라우드 통계: 총 {len(points)}개, 거리 범위: {min_dist:.2f}~{max_dist:.2f}m")
    
    # 거리 정규화 (안정성 개선)
    if max_dist - min_dist < 1e-6:
        norm_distances = np.zeros_like(distances)
    else:
        norm_distances = (distances - min_dist) / (max_dist - min_dist)
    
    # 벡터화된 투영 (캘리브레이션 사용)
    projected_pixels, valid_mask = project_points_vectorized(points, lidar_to_camera, width, height)
    
    valid_count = len(projected_pixels)
    projection_attempts = np.sum(valid_mask)
    
    # 투영된 포인트 그리기
    for orig_idx, u, v in projected_pixels:
        # 거리를 0-255 범위로 정규화
        depth_value = int(norm_distances[orig_idx] * 255)
        
        # OpenCV COLORMAP_VIRIDIS 사용 (아름다운 보라-파랑-초록-노랑 그라데이션)
        color = cv2.applyColorMap(np.array([[depth_value]], dtype=np.uint8), cv2.COLORMAP_VIRIDIS)[0, 0]
        b, g, r = int(color[0]), int(color[1]), int(color[2])
        
        # 더 큰 점과 부드러운 가장자리를 위한 안티앨리어싱
        # 메인 포인트 (크기 2)
        cv2.circle(result_image, (u, v), 2, (b, g, r), -1)
        
        # 투명한 테두리 효과를 위한 더 큰 원 (크기 3, 더 연한 색상)
        border_b = min(255, b + 30)
        border_g = min(255, g + 30) 
        border_r = min(255, r + 30)
        cv2.circle(result_image, (u, v), 3, (border_b, border_g, border_r), 1)
    
    print(f"투영 결과: {valid_count}/{projection_attempts} 포인트가 이미지 내에 투영됨")
    
    # 타임스탬프 정보
    img_ts = extract_timestamp_from_filename(img_file)
    pc_ts = extract_timestamp_from_filename(pc_file)
    ts_diff = abs(img_ts - pc_ts)
    
    # 정보 표시
    info_text = f"PC: {pc_name[:15]}..., Img: {img_name[:15]}... (Calib)"
    cv2.putText(result_image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    stats_text = f"Points: {valid_count}/{len(points)} ({valid_count/len(points)*100:.1f}%)"
    cv2.putText(result_image, stats_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    ts_text = f"TS diff: {ts_diff}"
    cv2.putText(result_image, ts_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # 거리 정보 표시 + 색상 범례 (업데이트된 설명)
    dist_text = f"Dist: {min_dist:.1f}-{max_dist:.1f}m (Purple=Close, Yellow=Far)"
    cv2.putText(result_image, dist_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    output_file = os.path.join(output_dir, f"projection_{img_name}_pc_{pc_name}.png")
    cv2.imwrite(output_file, result_image)
    
    # 거리 통계 정보 반환
    distance_stats = {
        "min_distance": min_dist,
        "max_distance": max_dist,
        "mean_distance": mean_dist,
        "std_distance": std_dist
    }
    
    return output_file, valid_count, len(points), ts_diff, distance_stats

def create_colored_pointcloud(pc_file, img_file, lidar_to_camera, width, height, output_file):
    """컬러 포인트클라우드 생성 (사용하지 않음)"""
    pass

def convert_to_serializable(obj):
    """numpy 타입을 JSON 직렬화 가능한 타입으로 변환"""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='Project LiDAR points to nearest timestamp images')
    parser.add_argument('--base_dir', type=str, 
                        default='/data/gpfs/projects/punim2482/workspace/PaRL/datasets/rosbag/input/IxT1',
                        help='Base directory containing the data')
    parser.add_argument('--rosbag_name', type=str, 
                        default='rosbag2_2025_09_24-23_00_10',
                        help='Name of the rosbag folder')
    parser.add_argument('--output_suffix', type=str, 
                        default='_projections',
                        help='Suffix for output directory name')
    
    args = parser.parse_args()
    
    base_dir = args.base_dir
    rosbag_name = args.rosbag_name
    
    print(f"처리할 데이터: {base_dir}/{rosbag_name}")
    
    # 경로 설정
    print(f"처리할 데이터: {base_dir}/{rosbag_name}")
    
    # 경로 설정: calib_result 안에 하나의 rosbag 폴더(또는 직접 calib.json)가 있으면 자동 선택
    calib_root = os.path.join(base_dir, "calib_result")
    calib_file = None
    if os.path.isdir(calib_root):
        # calib_result/calib.json 직접 있는 경우 우선 사용
        direct_calib = os.path.join(calib_root, "calib.json")
        if os.path.exists(direct_calib):
            calib_file = direct_calib
            print(f"calib_result/calib.json 사용: {calib_file}")
        else:
            # 폴더 목록 수집
            subdirs = [d for d in os.listdir(calib_root) if os.path.isdir(os.path.join(calib_root, d))]
            if len(subdirs) == 1:
                calib_file = os.path.join(calib_root, subdirs[0], "calib.json")
                print(f"calib_result에서 단일 폴더 발견, 해당 폴더 사용: {subdirs[0]}")
            elif rosbag_name and os.path.isdir(os.path.join(calib_root, rosbag_name)):
                # 기존 동작(rosbag_name 폴더가 있으면 사용)
                calib_file = os.path.join(calib_root, rosbag_name, "calib.json")
                print(f"rosbag_name 폴더에서 캘리브레이션 사용: {rosbag_name}")
            elif subdirs:
                # 여러 폴더가 있으면 첫 번째를 사용하되 경고 출력
                calib_file = os.path.join(calib_root, subdirs[0], "calib.json")
                print(f"calib_result에 여러 폴더가 있습니다. 첫 번째 폴더 사용: {subdirs[0]}")
    else:
        print("calib_result 폴더가 존재하지 않음. 캘리브레이션 없이 실행합니다.")
    extracted_dir = os.path.join(base_dir, "extracted_frames", rosbag_name)
    image_dir = os.path.join(extracted_dir, "images")
    pointcloud_dir = os.path.join(extracted_dir, "pointclouds")
    output_dir = os.path.join(base_dir, f"{rosbag_name}{args.output_suffix}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 파일 목록 가져오기
    if not os.path.exists(image_dir) or not os.path.exists(pointcloud_dir):
        print(f"디렉토리를 찾을 수 없습니다.")
        print(f"이미지: {image_dir}")
        print(f"포인트클라우드: {pointcloud_dir}")
        return
    
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    pointcloud_files = sorted([os.path.join(pointcloud_dir, f) for f in os.listdir(pointcloud_dir) if f.endswith('.bin')])
    
    if not image_files or not pointcloud_files:
        print("파일을 찾을 수 없습니다.")
        return
    
    print(f"이미지 파일: {len(image_files)}개")
    print(f"포인트클라우드 파일: {len(pointcloud_files)}개")
    
    # 타임스탬프 분석
    img_timestamps, pc_timestamps, strategy = analyze_timestamps(image_files, pointcloud_files)
    
    # 캘리브레이션 로드
    lidar_to_camera = None
    width, height = 2880, 1440
    
    if os.path.exists(calib_file):
        try:
            lidar_to_camera, camera_model, intrinsics = load_calibration(calib_file)
            width, height = int(intrinsics[0]), int(intrinsics[1])
            print(f"캘리브레이션 로드 완료: {camera_model}, {width}x{height}")
        except Exception as e:
            print(f"캘리브레이션 로드 실패: {e}")
            lidar_to_camera = None
    else:
        # 이미지에서 크기 추정
        first_image = cv2.imread(os.path.join(image_dir, image_files[0]))
        if first_image is not None:
            height, width = first_image.shape[:2]
            print(f"이미지에서 크기 추정: {width}x{height}")
    
    # 매칭 결과 저장
    matched_pairs = []
    total_processed = 0
    total_valid_points = 0
    total_points = 0
    timestamp_diffs = []
    all_distance_stats = []
    
    print("\n시간별 매칭 및 투영 시작...")
    
    # 매칭 전략에 따른 처리
    if strategy == "sequence":
        # 시퀀스 기반 매칭
        matches = find_nearest_by_sequence(image_files, pointcloud_files)
        
        for img_file, pc_file, sequence_diff in tqdm(matches, desc="Processing"):
            img_path = os.path.join(image_dir, img_file)
            img_timestamp = extract_timestamp_from_filename(img_file)
            pc_timestamp = extract_timestamp_from_filename(pc_file)
            
            # 페어 정보 생성
            pair_info = {
                "image_file": img_file,
                "image_timestamp": int(img_timestamp),  # 명시적 int 변환
                "pointcloud_file": os.path.basename(pc_file),
                "pointcloud_timestamp": int(pc_timestamp),  # 명시적 int 변환
                "sequence_diff": int(sequence_diff),  # 명시적 int 변환
                "matching_strategy": "sequence",
                "processed": False
            }
            
            # 투영 수행
            result = project_lidar_to_image(pc_file, img_path, lidar_to_camera, width, height, output_dir)
            
            if result is not None:
                output_file, valid_count, total_point_count, ts_diff, distance_stats = result
                
                pair_info.update({
                    "processed": True,
                    "output_file": os.path.relpath(output_file, base_dir),
                    "valid_points": int(valid_count),  # 명시적 int 변환
                    "total_points": int(total_point_count),  # 명시적 int 변환
                    "projection_ratio": float(valid_count / total_point_count) if total_point_count > 0 else 0.0
                })
                
                # 거리 통계 추가
                pair_info.update(distance_stats)
                all_distance_stats.append(distance_stats)
                
                total_processed += 1
                total_valid_points += valid_count
                total_points += total_point_count
                timestamp_diffs.append(sequence_diff)
            
            matched_pairs.append(pair_info)
    
    else:
        # 타임스탬프 기반 매칭 (기존 방식)
        pc_timestamps_cache = [extract_timestamp_from_filename(f) for f in pointcloud_files]
        
        for img_file in tqdm(image_files, desc="Processing"):
            img_path = os.path.join(image_dir, img_file)
            img_timestamp = extract_timestamp_from_filename(img_file)
            
            nearest_pc, time_diff = find_nearest_pointcloud_optimized(
                img_timestamp, pointcloud_files, pc_timestamps_cache
            )
            
            if nearest_pc is None:
                continue
            
            pair_info = {
                "image_file": img_file,
                "image_timestamp": int(img_timestamp),  # 명시적 int 변환
                "pointcloud_file": os.path.basename(nearest_pc),
                "pointcloud_timestamp": int(extract_timestamp_from_filename(nearest_pc)),  # 명시적 int 변환
                "timestamp_diff_ns": int(time_diff),  # 명시적 int 변환
                "matching_strategy": "timestamp",
                "processed": False
            }
            
            result = project_lidar_to_image(nearest_pc, img_path, lidar_to_camera, width, height, output_dir)
            
            if result is not None:
                output_file, valid_count, total_point_count, ts_diff, distance_stats = result
                
                pair_info.update({
                    "processed": True,
                    "output_file": os.path.relpath(output_file, base_dir),
                    "valid_points": int(valid_count),  # 명시적 int 변환
                    "total_points": int(total_point_count),  # 명시적 int 변환
                    "projection_ratio": float(valid_count / total_point_count) if total_point_count > 0 else 0.0
                })
                
                # 거리 통계 추가
                pair_info.update(distance_stats)
                all_distance_stats.append(distance_stats)
                
                total_processed += 1
                total_valid_points += valid_count
                total_points += total_point_count
                timestamp_diffs.append(time_diff)
            
            matched_pairs.append(pair_info)
    
    # 전체 거리 통계 계산
    if all_distance_stats:
        all_min_distances = [stat["min_distance"] for stat in all_distance_stats]
        all_max_distances = [stat["max_distance"] for stat in all_distance_stats]
        all_mean_distances = [stat["mean_distance"] for stat in all_distance_stats]
        
        global_distance_stats = {
            "global_min_distance": float(np.min(all_min_distances)),
            "global_max_distance": float(np.max(all_max_distances)),
            "avg_min_distance": float(np.mean(all_min_distances)),
            "avg_max_distance": float(np.mean(all_max_distances)),
            "avg_mean_distance": float(np.mean(all_mean_distances))
        }
    else:
        global_distance_stats = {}
    
    # 결과 JSON 저장
    pairs_json_path = os.path.join(output_dir, "matched_pairs.json")
    pairs_data = {
        "metadata": {
            "base_dir": base_dir,
            "rosbag_name": rosbag_name,
            "total_images": len(image_files),
            "total_pointclouds": len(pointcloud_files),
            "processed_pairs": int(total_processed),  # 명시적 int 변환
            "calibration_used": lidar_to_camera is not None,
            "image_size": [int(width), int(height)],  # 명시적 int 변환
            "matching_strategy": strategy
        },
        "statistics": {
            "total_valid_points": int(total_valid_points),  # 명시적 int 변환
            "total_points": int(total_points),  # 명시적 int 변환
            "overall_projection_ratio": float(total_valid_points / total_points) if total_points > 0 else 0.0,
            "avg_diff": float(np.mean(timestamp_diffs)) if timestamp_diffs else 0.0,
            "max_diff": float(np.max(timestamp_diffs)) if timestamp_diffs else 0.0,
            "min_diff": float(np.min(timestamp_diffs)) if timestamp_diffs else 0.0
        },
        "distance_statistics": global_distance_stats,
        "pairs": matched_pairs
    }
    
    # JSON 직렬화 가능하도록 변환
    pairs_data = convert_to_serializable(pairs_data)
    
    with open(pairs_json_path, 'w', encoding='utf-8') as f:
        json.dump(pairs_data, f, indent=2, ensure_ascii=False)
    
    # 최종 결과
    print(f"\n=== 처리 완료 ===")
    print(f"처리된 이미지: {total_processed}/{len(image_files)}")
    print(f"총 투영 포인트: {total_valid_points}/{total_points} ({total_valid_points/total_points*100:.1f}%)" if total_points > 0 else "총 투영 포인트: 0")
    if timestamp_diffs:
        print(f"평균 차이: {np.mean(timestamp_diffs):.1f}")
        print(f"최대 차이: {np.max(timestamp_diffs):.1f}")
    if global_distance_stats:
        print(f"전체 거리 범위: {global_distance_stats['global_min_distance']:.2f}~{global_distance_stats['global_max_distance']:.2f}m")
    print(f"결과 폴더: {output_dir}")
    print(f"매칭 정보: {pairs_json_path}")

if __name__ == "__main__":
    main()