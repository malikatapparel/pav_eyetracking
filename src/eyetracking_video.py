# %%
import cv2
import numpy as np
import pandas as pd

cap = cv2.VideoCapture("../data/raw/eye_movie.mov")

# Get video properties for the output
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Setup video writer (same format/size as input)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' for .avi
out = cv2.VideoWriter("../data/processed/eye_movie_tracked.mov", fourcc, fps, (width, height))

# Set up middle line for right/left classification in Pavlovian task (in ROI coordinates: cols=450px wide, middle = 225)
MIDDLE_LINE = 244  # position of the pupil looking at the center
CENTER = 20     # pixels around middle line = CENTER

# Storage for gaze data
results = []
frame_idx = 0

print("Processing video...")

# Add eye-tracking to the video
while True:
    ret, frame = cap.read()
    if ret is False:
        break
    roi = frame[400:700, 100:550]
    rows, cols, _ = roi.shape
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Add some blur to reduce noise and make the pupil more visible
    gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)
    # Set threshold for black (pupil)
    _, threshold = cv2.threshold(gray_roi, 40, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy= cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Pupil detection and gaze classification
    gaze_side = 0  # Default: CENTER (0)
    pupil_center_x = np.nan
    pupil_center_y = np.nan

     # Take the biggest area of the image (pupil) by sorting the contours
    contours = sorted(contours, key = lambda x: cv2.contourArea(x), reverse = True)
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        pupil_center_x = x + int(w/2)  # Horizontal center
        pupil_center_y = y + int(h/2)  # Vertical center
        cv2.rectangle(roi, (x,y), (x+w, y+h), (255, 0, 0), 2)
        cv2.line(roi, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 1)
        cv2.line(roi, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 1)

        # Gaze classification based on horizontal position
        if pupil_center_x < MIDDLE_LINE - CENTER:
            gaze_side = -1  # LEFT
            cv2.putText(roi, "LEFT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        elif pupil_center_x > MIDDLE_LINE + CENTER:
            gaze_side = 1   # RIGHT
            cv2.putText(roi, "RIGHT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(roi, "CENTER", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        break
    results.append({
        "frame": frame_idx,
        "timestamp": frame_idx / fps,
        "pupil_center_x": pupil_center_x,
        "pupil_center_y": pupil_center_y,
        "gaze_side": gaze_side
    })
    # Write frame to output video
    out.write(frame)

    cv2.imshow("Roi", roi)

    frame_idx += 1

    key = cv2.waitKey(30)
    if key == 27:
        break
# Cleanup
cv2.destroyAllWindows()
print("Saved to ../data/processed/eye_movie_tracked.mov")
cap.release()
out.release()

# Export gaze data to CSV
df = pd.DataFrame(results)
output_csv = "../data/processed/gaze_data.csv"
df.to_csv(output_csv, index=False)

# %%
