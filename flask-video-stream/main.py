import os
import cv2
import csv
import math
import numpy as np
import mediapipe as mp

VISIBILITY_THRESHOLD = 0.5
RGB_CHANNELS = 3


def parse_args():
    parser = argparse.ArgumentParser(description="Face Mask Overlay")
    parser.add_argument("--mask_image", help="path to the .png file with a mask", required=True, type=str)
    args = parser.parse_args()
    return args


def normalized_to_pixel_coordinates(normalized_x, normalized_y, image_width, image_height):
    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value):
        return (value > 0 or math.isclose(0, value)) and (value < 1 or math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and is_valid_normalized_value(normalized_y)):
        return None

    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)

    return x_px, y_px


def absolute_points(landmark_list, image):
    if not landmark_list:
        return
    if image.shape[2] != RGB_CHANNELS:
        raise ValueError('Input image must contain three channel rgb data.')
    image_rows, image_cols, _ = image.shape
    idx_to_coordinates = {}
    for idx, landmark in enumerate(landmark_list.landmark):
        if ((landmark.HasField('visibility') and
                landmark.visibility < VISIBILITY_THRESHOLD) or
                (landmark.HasField('presence') and landmark.presence < PRESENCE_THRESHOLD)):
            continue
        landmark_px = normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                      image_cols, image_rows)
        if landmark_px:
            idx_to_coordinates[idx] = landmark_px
    return idx_to_coordinates


def mask_attach_points(mask_image_path):
    # Load mask annotations from csv file to source points
    mask_annotation = os.path.splitext(os.path.basename(mask_image_path))[0]
    mask_annotation = os.path.join(os.path.dirname(mask_image_path), mask_annotation + ".csv")

    with open(mask_annotation) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        attach_pts = []
        src_pts = []
        for i, row in enumerate(csv_reader):
            # Skip head or empty line if it's there
            try:
                attach_pts.append(int(row[0]))
                src_pts.append(np.array([float(row[1]), float(row[2])]))
            except ValueError:
                continue
    src_pts = np.array(src_pts, dtype="float32")
    return attach_pts, src_pts


def crucial_coords(attach_pts, landmarks, mask_pts):
    mesh_pts = []
    to_del_id_list = []
    for index, val in enumerate(attach_pts):
        try:
            mesh_pts.append(np.array([float(landmarks[val][0]), float(landmarks[val][1])]))
        except KeyError:
            to_del_id_list.append(index)
            #mask_pts = np.delete(mask_pts, index, axis=0)
            continue
    mesh_pts = np.array(mesh_pts, dtype="float32")
    # print(to_del_id_list)
    mask_pts = np.delete(mask_pts, to_del_id_list, axis=0)
    return mesh_pts, mask_pts


def mask_transform(mask_image_path, src_pts, dst_pts, h, w):
    mask_img = cv2.imread(mask_image_path, cv2.IMREAD_UNCHANGED)
    mask_img = mask_img.astype(np.float32)
    mask_img /= 255.0

    # get the perspective transformation matrix
    M, _ = cv2.findHomography(src_pts, dst_pts)

    #print(M)
    # transformed masked image
    transformed_mask = cv2.warpPerspective(mask_img,
                                           M,
                                           (w, h),
                                           None,
                                           cv2.INTER_LINEAR,
                                           cv2.BORDER_CONSTANT)
    # mask overlay
    alpha_mask = transformed_mask[:, :, 3]
    alpha_image = 1.0 - alpha_mask
    return transformed_mask, alpha_mask, alpha_image


def main(image, mask_image):
    attach_pts, mask_pts_const = mask_attach_points(mask_image)
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh
    # drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    with mp_face_mesh.FaceMesh(max_num_faces=3, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = face_mesh.process(image)

            # Draw the face mesh annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            result = image.astype(np.float32)/255.0
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    absolute_face_landmarks = absolute_points(face_landmarks, image)
                    # print(absolute_face_landmarks)
                    # print(len(absolute_face_landmarks))
                    mesh_pts, mask_pts = crucial_coords(attach_pts, absolute_face_landmarks, mask_pts_const)
                    # print(mask_pts)
                    # print(mesh_pts)
                    h, w, _ = image.shape
                    transformed_mask, alpha_mask, alpha_image = mask_transform(mask_image,
                                                                               mask_pts,
                                                                               mesh_pts,
                                                                               h,
                                                                               w)

                    for c in range(0, 3):
                        result[:, :, c] = (alpha_mask*transformed_mask[:, :, c] + alpha_image*result[:, :, c])*255.0
            # cv2.imshow('Masked Face', result)
            return result