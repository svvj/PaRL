import os
import json
import numpy as np
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation
from tqdm import tqdm

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

def load_pointcloud(pointcloud_file, max_distance=50.0):
    """포인트 클라우드 파일을 로드하고 필터링"""
    if pointcloud_file.endswith('.bin'):
        with open(pointcloud_file, 'rb') as f:
            data = f.read()
        
        try:
            # XYZI 형식 (4채널) 시도
            points = np.frombuffer(data, dtype=np.float32).reshape(-1, 4)[:, :3]
        except ValueError:
            try:
                # XYZ 형식 (3채널) 시도
                points = np.frombuffer(data, dtype=np.float32).reshape(-1, 3)
            except ValueError:
                return None, None
    else:
        # Open3D 지원 형식
        pcd = o3d.io.read_point_cloud(pointcloud_file)
        if not pcd.has_points():
            return None, None
        points = np.asarray(pcd.points)
    
    # 유효한 포인트만 필터링
    mask = np.all(np.isfinite(points), axis=1) & np.all(np.abs(points) < 100.0, axis=1)
    points = points[mask]
    
    # 거리 기반 필터링
    distances = np.linalg.norm(points, axis=1)
    distance_mask = distances < max_distance
    points = points[distance_mask]
    
    return points, distances[distance_mask]

def project_point_standard(point, width, height):
    """표준 equirectangular 투영 공식 사용"""
    depth = np.sqrt(np.sum(point**2))
    if depth < 1e-5:
        return None
    
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
    
    return int(u) % width, int(v) % height

def apply_y_flip(point):
    """Y축 반전 적용"""
    return np.array([point[0], -point[1], point[2]])

def project_lidar_to_image(pc_file, img_file, lidar_to_camera, width, height, output_dir):
    """라이다 포인트를 이미지에 투영"""
    # 파일명 추출
    pc_name = os.path.basename(pc_file).replace('.bin', '')
    img_name = os.path.basename(img_file).replace('.png', '')
    
    # 포인트 클라우드와 이미지 로드
    points, distances = load_pointcloud(pc_file)
    if points is None or len(points) == 0:
        print(f"포인트클라우드 로드 실패: {pc_file}")
        return None
    
    image = cv2.imread(img_file)
    if image is None:
        print(f"이미지 로드 실패: {img_file}")
        return None
    
    # 결과 이미지 복사
    result_image = image.copy()
    valid_count = 0
    
    # 거리 정규화 (색상 매핑용)
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    norm_distances = (distances - min_dist) / (max_dist - min_dist) if max_dist > min_dist else np.zeros_like(distances)
    
    # 각 포인트 투영
    for i, point in enumerate(points):
        # 라이다→카메라 변환 적용
        camera_point = lidar_to_camera @ np.append(point, 1)
        
        # Y축 반전 적용
        transformed_point = apply_y_flip(camera_point[:3])
        
        # 표준 공식으로 투영
        pixel = project_point_standard(transformed_point, width, height)
        
        if pixel is not None:
            u, v = pixel
            if 0 <= u < width and 0 <= v < height:
                # 거리에 따른 색상: 가까울수록 빨강(255,0,0), 멀수록 파랑(0,0,255)
                blue = int(255 * norm_distances[i])
                red = int(255 * (1 - norm_distances[i]))
                green = 0  # 중간 거리는 보라색 계열로 표현
                
                # 포인트 크기는 거리에 반비례 (가까울수록 크게)
                point_size = max(1, int(5 * (1 - norm_distances[i])))
                
                # 이미지에 포인트 그리기
                cv2.circle(result_image, (u, v), point_size, (blue, green, red), -1)
                valid_count += 1
    
    # 정보 추가
    info_text = f"PC: {pc_name}, Img: {img_name}"
    cv2.putText(result_image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    stats_text = f"포인트: {valid_count}/{len(points)} ({valid_count/len(points)*100:.1f}%)"
    cv2.putText(result_image, stats_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # 결과 저장
    output_file = os.path.join(output_dir, f"projection_pc_{pc_name}.png")
    cv2.imwrite(output_file, result_image)
    
    print(f"PC {pc_name}: {valid_count}/{len(points)} 포인트 ({valid_count/len(points)*100:.1f}%)")
    
    return output_file, valid_count, len(points)

def main():
    """메인 실행 함수"""
    # 데이터 경로 설정
    base_dir = "/home/cgdesktop01/workspace/direct_visual_lidar_calibration/data"
    rosbag_name = "rosbag2_2024_11_11-14_55_01"
    data_dir = os.path.join(base_dir, "extracted_data", rosbag_name)
    calib_file = os.path.join(base_dir, "calibration output", rosbag_name, "calib.json")
    
    # 결과 저장 폴더 생성
    output_dir = os.path.join(base_dir, "first_image_projections")
    os.makedirs(output_dir, exist_ok=True)
    
    # 이미지 및 포인트 클라우드 디렉토리
    image_dir = os.path.join(data_dir, "images", "image")
    pointcloud_dir = os.path.join(data_dir, "pointclouds", "ouster_points")
    
    # 파일 목록 가져오기
    image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')])
    pointcloud_files = sorted([os.path.join(pointcloud_dir, f) for f in os.listdir(pointcloud_dir) if f.endswith('.bin')])
    
    if not image_files or not pointcloud_files:
        print(f"이미지 또는 포인트 클라우드 파일을 찾을 수 없습니다.")
        return
    
    # 캘리브레이션 로드
    lidar_to_camera, camera_model, intrinsics = load_calibration(calib_file)
    width, height = int(intrinsics[0]), int(intrinsics[1])
    
    # 첫 번째 이미지만 사용
    first_image = image_files[0]
    img_name = os.path.basename(first_image).replace('.png', '')
    
    print(f"첫 번째 이미지({img_name})를 사용하여 {len(pointcloud_files)}개의 라이다 프레임 테스트")
    
    # 모든 라이다 프레임에 대해 처리
    results = []
    
    with tqdm(total=len(pointcloud_files), desc="라이다 프레임 투영 중") as pbar:
        for pc_file in pointcloud_files:
            result = project_lidar_to_image(pc_file, first_image, lidar_to_camera, width, height, output_dir)
            if result:
                results.append((pc_file, first_image, *result))
            pbar.update(1)
    
    # 결과 정렬 (가장 많은 포인트가 투영된 순)
    valid_results = [(pc, img, out, valid, total) for pc, img, out, valid, total in results if out is not None]
    sorted_results = sorted(valid_results, key=lambda x: x[3]/x[4], reverse=True)
    
    # 상위 10개 결과 출력
    print("\n최적의 라이다 프레임 (상위 10개):")
    for i, (pc, img, out, valid, total) in enumerate(sorted_results[:10]):
        percent = valid / total * 100
        print(f"{i+1}. PC: {os.path.basename(pc)}")
        print(f"   유효 포인트: {valid}/{total} ({percent:.1f}%)")
        print(f"   결과 파일: {os.path.basename(out)}")
    
    # 결과 요약 파일 저장
    with open(os.path.join(output_dir, "results_summary.txt"), "w") as f:
        f.write("순위, 포인트클라우드, 유효 포인트, 전체 포인트, 비율(%)\n")
        for i, (pc, img, out, valid, total) in enumerate(sorted_results):
            percent = valid / total * 100
            f.write(f"{i+1}, {os.path.basename(pc)}, {valid}, {total}, {percent:.1f}\n")
    
    print(f"\n결과 요약이 저장되었습니다: {os.path.join(output_dir, 'results_summary.txt')}")
    
    # 최적의 프레임 쌍 정보 저장
    if sorted_results:
        best_pc, best_img, best_out, best_valid, best_total = sorted_results[0]
        best_percent = best_valid / best_total * 100
        
        best_info = {
            "image_file": img_name,
            "pointcloud_file": os.path.basename(best_pc),
            "valid_points": best_valid,
            "total_points": best_total,
            "percentage": best_percent
        }
        
        with open(os.path.join(output_dir, "best_lidar_frame.json"), "w") as f:
            json.dump(best_info, f, indent=4)
        
        print(f"최적의 라이다 프레임 정보가 저장되었습니다: {os.path.join(output_dir, 'best_lidar_frame.json')}")
        
        # 최적의 프레임 쌍을 대표 이미지로 복사
        best_image_path = os.path.join(output_dir, "best_projection.png")
        try:
            import shutil
            shutil.copy(best_out, best_image_path)
            print(f"최적의 투영 결과가 복사되었습니다: {best_image_path}")
        except Exception as e:
            print(f"최적의 투영 결과 복사 실패: {e}")

if __name__ == "__main__":
    main()