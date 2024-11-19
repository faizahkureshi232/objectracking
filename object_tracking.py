import sys
from mobile_net import ObjectRecognition
from deep_sort.person_id_model.generate_person_features import generate_detections, init_encoder
from deep_sort.deep_sort_app import run_deep_sort, DeepSORTConfig
from deep_sort.application_util.visualization import cv2

cap = cv2.VideoCapture(cv2.samples.findFile(sys.argv[1]))

model = ObjectRecognition()
encoder = init_encoder()
config = DeepSORTConfig()

while(True):
    ret, frame = cap.read()
    boxes = model.get_boxes(frame)

    if len(boxes) > 0:
        encoding = generate_detections(encoder, boxes, frame)
        run_deep_sort(frame, encoding, config)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()