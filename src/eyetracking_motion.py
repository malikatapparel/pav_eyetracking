# %%
import cv2

cap = cv2.VideoCapture("../data/raw/eye_movie.mov")

# Get video properties for the output
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Setup video writer (same format/size as input)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' for .avi
out = cv2.VideoWriter("../data/processed/eye_movie_tracked.mov", fourcc, fps, (width, height))

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
     # Take the biggest area of the image (pupil) by sorting the contours
    contours = sorted(contours, key = lambda x: cv2.contourArea(x), reverse = True)
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        cv2.rectangle(roi, (x,y), (x+w, y+h), (255, 0, 0), 2)
        cv2.line(roi, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 1)
        cv2.line(roi, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 1)
        #cv2.drawContours(roi, [cnt], -1, (0,0,255), 3)
        break
    
    # Write frame to output video
    out.write(frame)

    cv2.imshow("Roi", roi)
    key = cv2.waitKey(30)
    if key == 27:
        break
# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
print("Saved to ../data/processed/eye_movie_tracked.mov")

# %%
