#调用intel d435摄像头使用pyrealsense2库获取深度图、RGB图。使用mediapipe库识别手掌，使用矩形框框选住手掌，计算手掌宽度，计算手掌中心位置及其到摄像头的距离，在窗口中的深度图画面以及RGB图画面同时进行渲染。
import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Start streaming
pipe_profile = pipeline.start(config)  # streaming流开始
# align_to = rs.stream.color  # align_to 是计划对齐深度帧的流类型
# align = rs.align(align_to)  # rs.align 执行深度帧与其他帧的对齐
align = rs.align(rs.stream.color)  #将上两句合成一句，将深度与color对齐

# Initialize mediapipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Create opencv window
cv2.namedWindow("Hand Tracking", cv2.WINDOW_AUTOSIZE)


def get_aligned_images():
    ''' 
    获取对齐图像帧与相机参数
    '''
    frames = pipeline.wait_for_frames()  # 等待获取图像帧，获取颜色和深度的框架集
    aligned_frames = align.process(frames)  # 获取对齐帧，将深度框与颜色框对齐

    aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的的depth帧
    aligned_color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的的color帧

    #### 获取相机参数 ####
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）
    color_intrin = aligned_color_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参

    #### 将images转为numpy arrays ####
    color_image = np.asanyarray(aligned_color_frame.get_data())  # RGB图
    depth_image = np.asanyarray(aligned_depth_frame.get_data())  # 深度图（默认16位）

    return color_intrin, depth_intrin, color_image, depth_image, aligned_depth_frame, aligned_color_frame



def get_bounding_box(hand_landmarks, shape):
    x = [landmark.x for landmark in hand_landmarks.landmark]
    y = [landmark.y for landmark in hand_landmarks.landmark]
    z = [landmark.z for landmark in hand_landmarks.landmark]
    x_min, x_max = int(min(x) * shape[1]), int(max(x) * shape[1])
    y_min, y_max = int(min(y) * shape[0]), int(max(y) * shape[0])
    z_min, z_max = int(min(z) * 1000), int(max(z) * 1000)
    return x_min, y_min, x_max, y_max

def get_3d_coordinates(pixel_coordinates, aligned_depth_frame, depth_intrin):

    dis =  aligned_depth_frame.get_distance(pixel_coordinates[0], pixel_coordinates[1])
    return rs.rs2_deproject_pixel_to_point(depth_intrin, pixel_coordinates, dis)



while True:
    # Wait for a coherent pair of frames: depth and color
    color_intrin, depth_intrin, color_image, depth_image, aligned_depth_frame, aligned_color_frame = get_aligned_images()  # 获取对齐图像与相机参数
    if not aligned_depth_frame or not aligned_color_frame:
        continue


    # Process hand tracking
    results = hands.process(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on image
            mp.solutions.drawing_utils.draw_landmarks(color_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get bounding box of hand
            x_min, y_min, x_max, y_max = get_bounding_box(hand_landmarks, color_image.shape)

            # Draw bounding box on image
            # cv2.rectangle(color_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Draw line between thumb and index finger
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            if thumb_tip.x > 1 or thumb_tip.x < 0 or thumb_tip.y > 1 or thumb_tip.y < 0: 
                continue
            if index_finger_tip.x > 1 or index_finger_tip.x < 0 or index_finger_tip.y > 1 or index_finger_tip.y < 0: 
                continue
            cv2.line(color_image, (int(thumb_tip.x * color_image.shape[1]), int(thumb_tip.y * color_image.shape[0])), (int(index_finger_tip.x * color_image.shape[1]), int(index_finger_tip.y * color_image.shape[0])), (0, 0, 255), 2)

            # Calculate hand center position
            hand_center_x = int((x_min + x_max) / 2)
            hand_center_y = int((y_min + y_max) / 2)

            # Calculate hand distance from camera
            hand_distance = aligned_depth_frame.get_distance(hand_center_x, hand_center_y)

            # Get the 3D coordinates of the thumb tip and index finger tip
            thumb_tip_3d = get_3d_coordinates(pixel_coordinates=[int(thumb_tip.x * color_image.shape[1]), int(thumb_tip.y * color_image.shape[0])], 
                                              aligned_depth_frame=aligned_depth_frame, 
                                              depth_intrin=depth_intrin)
            index_finger_tip_3d = get_3d_coordinates(pixel_coordinates=[int(index_finger_tip.x * color_image.shape[1]), int(index_finger_tip.y * color_image.shape[0])], 
                                                     aligned_depth_frame=aligned_depth_frame, 
                                                     depth_intrin=depth_intrin)

            # Calculate the distance between the thumb tip and index finger tip
            distance = np.sqrt(np.sum(np.square(np.array(thumb_tip_3d) - np.array(index_finger_tip_3d))))

            # Print the distance between the thumb tip and index finger tip
            # print("Distance between thumb tip and index finger tip: {:.2f} cm".format(distance * 100))
            
            # Draw hand width and distance on image
            cv2.putText(color_image, "Distance between thumb tip and index finger tip: {:.2f} cm".format(distance * 100), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(color_image, "Hand Distance: {:.2f} cm".format(hand_distance * 100), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            
    # Render depth and color images
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    images = np.hstack((depth_colormap, color_image))
    cv2.imshow("Hand Tracking", images)

    # Exit program when "q" key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
hands.close()
pipeline.stop()
cv2.destroyAllWindows()
