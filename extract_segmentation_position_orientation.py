# TODO : remettre code au propre

from ultralytics import YOLOE
import cv2
import numpy as np
import matplotlib.pyplot as plt


def model_prediction(model_choice, image_path, detection_element):

    model = YOLOE(model_choice)

    model.set_classes(detection_element)

    results = model.predict(image_path)

    return results


def find_angle(mask_object):

    vx, vy, x0, y0 = cv2.fitLine(mask_object, cv2.DIST_L2, 0, 0.01, 0.01)

    vx = vx.item()
    vy = vy.item()

    base_angle = np.degrees(np.arctan2(abs(vy), abs(vx)))

    if vx >= 0 and vy <= 0:
        angle_deg = base_angle

    elif vx >= 0 and vy > 0:
        angle_deg = 180 - base_angle

    elif vx < 0 and vy >= 0:
        angle_deg = base_angle

    else:  # vx < 0 et vy < 0
        angle_deg = 180 - base_angle

    return angle_deg


def find_center(bbox):

    x1, y1, x2, y2 = bbox

    cx = float((x1 + x2) / 2)
    cy = float((y1 + y2) / 2)

    return (cx, cy)


def drawing_openCv(image_path, centers, angles):
    image = cv2.imread(image_path)

    for center, angle in zip(centers, angles):
        cv2.circle(
            image,
            center=(int(center[0]), int(center[1])),
            radius=3,
            color=(0, 255, 0),
            thickness=5,
        )

        cv2.putText(
            image,
            f"{angle:.1f} deg",
            (int(center[0]), int(center[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("image", 1000, 700)

    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":

    image_path = "images/test_final.jpg"
    # define the model :
    results = model_prediction(
        model_choice="models/yoloe-26l-seg.pt",
        image_path=image_path,
        detection_element=["fork", "plate", "knife"],
    )
    first_image_result = results[0]
    centers = []
    angles = []
    first_image_result.show()

    for mask, box, bbox in zip(
        first_image_result.masks.xy,
        first_image_result.boxes,
        first_image_result.boxes.xyxy,
    ):
        class_id = int(box.cls[0])
        label = first_image_result.names[class_id]

        if label == "plate":
            angle_deg = 0
        else:
            angle_deg = find_angle(mask)

        center = find_center(bbox)

        angles.append(angle_deg)
        centers.append(center)

    drawing_openCv(image_path=image_path, centers=centers, angles=angles)
