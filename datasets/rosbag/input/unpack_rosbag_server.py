import os
import argparse
import cv2
import numpy as np
from pathlib import Path
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

def save_camera_images(reader, topic, output_dir):
    """카메라 이미지를 PNG 파일로 저장"""
    img_dir = os.path.join(output_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)
    
    typestore = get_typestore(Stores.LATEST)
    connections = [x for x in reader.connections if x.topic == topic]
    
    i = 0
    for connection, timestamp, rawdata in reader.messages(connections=connections):
        try:
            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
            
            # Image 메시지를 OpenCV 이미지로 변환
            height = msg.height
            width = msg.width
            encoding = msg.encoding
            
            # NumPy 배열로 변환
            dtype = np.uint8
            if encoding.endswith('16'):
                dtype = np.uint16
            
            step = msg.step
            img_data = np.frombuffer(msg.data, dtype=dtype)
            
            # 채널 수 계산
            if step > 0 and width > 0:
                channels = step // width
                if channels > 0:
                    img_data = img_data.reshape(height, width, channels)
                else:
                    img_data = img_data.reshape(height, width)
            else:
                img_data = img_data.reshape(height, width)
            
            # RGB->BGR 변환 (필요한 경우)
            if encoding == 'rgb8' and len(img_data.shape) == 3 and img_data.shape[2] == 3:
                img_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
            
            # 타임스탬프 생성
            msg_timestamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
            filename = os.path.join(img_dir, f"{msg_timestamp:020d}.png")
            cv2.imwrite(filename, img_data)
            
            if i % 100 == 0:
                print(f"[camera] saved {i} images to {img_dir}")
            i += 1
            
        except Exception as e:
            print(f"이미지 처리 오류: {e}")
            continue
    
    print(f"[camera] Total saved: {i} images")

def save_pointclouds(reader, topic, output_dir):
    """포인트클라우드를 바이너리 파일로 저장"""
    pc_dir = os.path.join(output_dir, 'pointclouds')
    os.makedirs(pc_dir, exist_ok=True)
    
    typestore = get_typestore(Stores.LATEST)
    connections = [x for x in reader.connections if x.topic == topic]
    
    i = 0
    for connection, timestamp, rawdata in reader.messages(connections=connections):
        try:
            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
            
            # PointCloud2 메시지 파싱
            n_points = msg.width * msg.height
            point_step = msg.point_step
            
            # 필드 정보 분석
            field_names = [field.name for field in msg.fields]
            
            # X, Y, Z 필드의 오프셋 찾기
            x_offset, y_offset, z_offset = None, None, None
            for field in msg.fields:
                if field.name == 'x':
                    x_offset = field.offset
                elif field.name == 'y':
                    y_offset = field.offset
                elif field.name == 'z':
                    z_offset = field.offset
            
            if None in (x_offset, y_offset, z_offset):
                print(f"경고: 포인트 클라우드에서 x, y, z 필드를 찾을 수 없습니다.")
                print(f"사용 가능한 필드: {field_names}")
                continue
            
            # 각 필드에 대한 데이터 추출
            cloud_array = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, point_step)
            
            # x, y, z 데이터 추출
            x_data = np.frombuffer(cloud_array[:, x_offset:x_offset+4].tobytes(), dtype=np.float32)
            y_data = np.frombuffer(cloud_array[:, y_offset:y_offset+4].tobytes(), dtype=np.float32)
            z_data = np.frombuffer(cloud_array[:, z_offset:z_offset+4].tobytes(), dtype=np.float32)
            
            # 3차원 포인트 배열 생성
            points = np.column_stack((x_data, y_data, z_data))
            
            # 유효하지 않은 포인트 필터링 (NaN, 무한대 등)
            valid_mask = np.all(np.isfinite(points), axis=1)
            valid_points = points[valid_mask]
            
            if len(valid_points) < len(points):
                filtered_count = len(points) - len(valid_points)
                if i % 10 == 0:  # 로그 스팸 방지
                    print(f"{filtered_count}개의 유효하지 않은 포인트 필터링됨")
            
            # 파일로 저장
            msg_timestamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
            filename = os.path.join(pc_dir, f"{msg_timestamp:020d}.bin")
            valid_points.astype(np.float32).tofile(filename)
            
            if i % 10 == 0:
                print(f"[lidar] {pc_dir}에 {len(valid_points)}개 포인트의 포인트클라우드 {i}번 저장함")
            i += 1
            
        except Exception as e:
            print(f"포인트클라우드 처리 오류: {e}")
            continue
    
    print(f"[lidar] 총 저장된 포인트클라우드: {i}개")

def extract_rosbag_data(bag_path, output_dir, camera_topic='/image', lidar_topic='/ouster/points'):
    """
    rosbag에서 데이터를 추출하여 저장
    
    Args:
        bag_path: rosbag 디렉토리 경로
        output_dir: 출력 베이스 디렉토리
        camera_topic: 카메라 토픽 이름
        lidar_topic: 라이다 토픽 이름
    """
    bag_path = Path(bag_path)
    
    # output dir: output + bag_name
    final_output_dir = os.path.join(output_dir, os.path.basename(bag_path))
    
    print(f"Reading rosbag from: {bag_path}")
    print(f"Extracting to: {final_output_dir}")
    
    # rosbag reader 생성
    with Reader(str(bag_path)) as reader:
        # 사용 가능한 토픽 정보 출력
        topic_types = {}
        for connection in reader.connections:
            topic_types[connection.topic] = connection.msgtype
            print(f"Topic: {connection.topic} -> {connection.msgtype}")
        
        print(f"사용 가능한 토픽: {list(topic_types.keys())}")
        
        # 카메라 이미지 저장
        print(f"카메라 이미지 추출 시작: {camera_topic}...")
        if camera_topic in topic_types:
            try:
                save_camera_images(reader, camera_topic, final_output_dir)
            except Exception as e:
                print(f"카메라 이미지 추출 오류: {e}")
        else:
            print(f"카메라 토픽 {camera_topic}을 찾을 수 없습니다. 사용 가능 토픽: {list(topic_types.keys())}")
        
        # 포인트클라우드 저장
        print(f"라이다 포인트클라우드 추출 시작: {lidar_topic}...")
        if lidar_topic in topic_types:
            try:
                save_pointclouds(reader, lidar_topic, final_output_dir)
            except Exception as e:
                print(f"포인트클라우드 추출 오류: {e}")
        else:
            print(f"라이다 토픽 {lidar_topic}을 찾을 수 없습니다. 사용 가능 토픽: {list(topic_types.keys())}")
    
    print("완료.")

def main():
    parser = argparse.ArgumentParser(description="Unpack ROS 2 bag to image and pointcloud files using rosbags library")
    parser.add_argument('--bag', '-b', required=True, help="input ROS 2 bag directory")
    parser.add_argument('--output', '-o', required=True, help="output base directory")
    parser.add_argument('--camera_topic', default='/image', help="ROS camera topic")
    parser.add_argument('--lidar_topic', default='/ouster/points', help="ROS pointcloud topic")
    args = parser.parse_args()
    
    if not os.path.exists(args.bag):
        print(f"Error: Rosbag path '{args.bag}' does not exist")
        return
    
    extract_rosbag_data(args.bag, args.output, args.camera_topic, args.lidar_topic)

if __name__ == "__main__":
    main()
