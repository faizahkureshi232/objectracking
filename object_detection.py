import sys
import cv2
from mobile_net import *

cap = cv2.VideoCapture(cv2.samples.findFile(sys.argv[1]))

model = ObjectRecognition()

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame_with_boxes, box_count = model.run_object_recognition(frame)

    cv2.putText(frame_with_boxes, 'People: '+ str(box_count), (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    cv2.imshow('frame', frame_with_boxes)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()