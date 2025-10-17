import os
import argparse
import cv2
import numpy as np
import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py
from rosbag2_py import StorageFilter

def save_camera_images_direct(reader, topic, topic_type, output_dir, serialization_format):
    img_dir = os.path.join(output_dir, 'images', topic.lstrip('/').replace('/', '_'))
    os.makedirs(img_dir, exist_ok=True)
    
    msg_type = get_message(topic_type)
    
    i = 0
    while reader.has_next():
        topic_name, data, t = reader.read_next()
        if topic_name != topic:
            continue
            
        msg = deserialize_message(data, msg_type)
        
        # 수동으로 Image 메시지를 OpenCV 이미지로 변환
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
        channels = step // width
        img_data = img_data.reshape(height, width, channels)
        
        # RGB->BGR 변환 (필요한 경우)
        if encoding == 'rgb8' and channels == 3:
            img_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
        
        timestamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        filename = os.path.join(img_dir, f"{timestamp:020d}.png")
        cv2.imwrite(filename, img_data)
        if i % 100 == 0:
            print(f"[camera] saved {i} images to {img_dir}")
        i += 1
    print(f"[camera] Total saved: {i} images")

def save_pointclouds(reader, topic, topic_type, output_dir, serialization_format):
    pc_dir = os.path.join(output_dir, 'pointclouds', topic.lstrip('/').replace('/', '_'))
    os.makedirs(pc_dir, exist_ok=True)
    
    msg_type = get_message(topic_type)
    
    i = 0
    while reader.has_next():
        topic_name, data, t = reader.read_next()
        if topic_name != topic:
            continue
            
        msg = deserialize_message(data, msg_type)
        
        try:
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
            
            # 각 필드에 대한 데이터 추출 (더 효율적인 방법)
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
                print(f"{len(points) - len(valid_points)}개의 유효하지 않은 포인트 필터링됨")
            
            # 파일로 저장
            timestamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
            filename = os.path.join(pc_dir, f"{timestamp:020d}.bin")
            valid_points.astype(np.float32).tofile(filename)
            
            if i % 10 == 0:
                print(f"[lidar] {pc_dir}에 {len(valid_points)}개 포인트의 포인트클라우드 {i}번 저장함")
            i += 1
            
        except Exception as e:
            print(f"포인트클라우드 처리 오류: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"[lidar] 총 저장된 포인트클라우드: {i}개")

def main():
    parser = argparse.ArgumentParser(description="Unpack ROS 2 bag to image and pointcloud files")
    parser.add_argument('--bag', '-b', required=True, help="input ROS 2 bag directory")
    parser.add_argument('--output', '-o', required=True, help="output base directory")
    parser.add_argument('--camera_topic', default='/image', help="ROS camera topic")
    parser.add_argument('--lidar_topic', default='/ouster/points', help="ROS pointcloud topic")
    args = parser.parse_args()
    
    # ROS 2 컨텍스트 초기화
    rclpy.init()
    
    # Reader 인스턴스 생성
    storage_options = rosbag2_py.StorageOptions(
        uri=args.bag,
        storage_id='sqlite3'
    )
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    
    # 모든 토픽과 타입 정보 가져오기
    topics = reader.get_all_topics_and_types()
    topic_types = {topic.name: topic.type for topic in topics}
    
    print(f"사용 가능한 토픽: {list(topic_types.keys())}")

    # output dir: output + args.bag
    output_dir = os.path.join(args.output, os.path.basename(args.bag))
    
    # 카메라 이미지 저장
    print(f"카메라 이미지 추출 시작: {args.camera_topic}...")
    if args.camera_topic in topic_types:
        try:
            reader.reset_filter()
            reader.seek(0)
            storage_filter = StorageFilter(topics=[args.camera_topic])
            reader.set_filter(storage_filter)
            save_camera_images_direct(reader, args.camera_topic, topic_types[args.camera_topic], 
                              output_dir, converter_options.output_serialization_format)
        except Exception as e:
            print(f"카메라 이미지 추출 오류: {e}")
    else:
        print(f"카메라 토픽 {args.camera_topic}을 찾을 수 없습니다. 사용 가능 토픽: {list(topic_types.keys())}")
    
    # 포인트클라우드 저장
    print(f"라이다 포인트클라우드 추출 시작: {args.lidar_topic}...")
    if args.lidar_topic in topic_types:
        try:
            reader.reset_filter()
            reader.seek(0)
            storage_filter = StorageFilter(topics=[args.lidar_topic])
            reader.set_filter(storage_filter)
            save_pointclouds(reader, args.lidar_topic, topic_types[args.lidar_topic],
                            output_dir, converter_options.output_serialization_format)
        except Exception as e:
            print(f"포인트클라우드 추출 오류: {e}")
    else:
        print(f"라이다 토픽 {args.lidar_topic}을 찾을 수 없습니다. 사용 가능 토픽: {list(topic_types.keys())}")
    
    print("완료.")
    rclpy.shutdown()

if __name__ == "__main__":
    main()