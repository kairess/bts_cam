import imutils, cv2, dlib
import numpy as np

lib = 'dlib'

# open video file
video_path = 'redvelvet_red.mp4'
cap = cv2.VideoCapture(video_path)

output_size = (375, 667) # (width, height)
fit_to = 'height'

# initialize writing video
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('%s_output.mp4' % (video_path.split('.')[0]), fourcc, cap.get(cv2.CAP_PROP_FPS), output_size)

# check file is opened
if not cap.isOpened():
  exit()

# initialize tracker
if lib == 'dlib':
  tracker = dlib.correlation_tracker()
else:
  OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create
  }

  tracker = OPENCV_OBJECT_TRACKERS['csrt']()

# global variables
img = None
top_left, bottom_right = tuple(), tuple()
top_bottom_list, left_right_list = [], []
mouse_down = False

# tracking function
def track(roi_img, roi):
  if lib == 'dlib':
    tracker.start_track(roi_img, roi)
  else:
    tracker.init(roi_img, roi)

  count = 0
  while True:
    count += 1
    # read frame from video
    ret, img = cap.read()

    if not ret:
      exit()

    # update tracker and get position from new frame
    if lib == 'dlib':
      tracker.update(img)
      rect = tracker.get_position()
    else:
      success, box = tracker.update(img)
      # if success:
      x, y, w, h = [int(v) for v in box]
      rect = dlib.rectangle(x, y, x+w, y+h)

    # if count % 10:
    #   tracker.start_track(img, rect)

    # save sizes of image
    top_bottom_list.append(np.array([rect.top(), rect.bottom()]))
    left_right_list.append(np.array([rect.left(), rect.right()]))

    # use recent 10 elements for crop (window_size=10)
    if len(top_bottom_list) > 10:
      del top_bottom_list[0]
      del left_right_list[0]

    # compute moving average
    avg_height_range = np.mean(top_bottom_list, axis=0).astype(np.int)
    avg_width_range = np.mean(left_right_list, axis=0).astype(np.int)
    avg_center = np.array([np.mean(avg_width_range), np.mean(avg_height_range)]) # (x, y)

    # compute scaled width and height
    scale = 1.3
    avg_height = (avg_height_range[1] - avg_height_range[0]) * scale
    avg_width = (avg_width_range[1] - avg_width_range[0]) * scale

    # compute new scaled ROI
    avg_height_range = np.array([avg_center[1] - avg_height / 2, avg_center[1] + avg_height / 2])
    avg_width_range = np.array([avg_center[0] - avg_width / 2, avg_center[0] + avg_width / 2])

    # fit to output aspect ratio
    if fit_to == 'width':
      avg_height_range = np.array([
        avg_center[1] - avg_width * output_size[1] / output_size[0] / 2,
        avg_center[1] + avg_width * output_size[1] / output_size[0] / 2
      ]).astype(np.int).clip(0, 9999)

      avg_width_range = avg_width_range.astype(np.int).clip(0, 9999)
    elif fit_to == 'height':
      avg_height_range = avg_height_range.astype(np.int).clip(0, 9999)

      avg_width_range = np.array([
        avg_center[0] - avg_height * output_size[0] / output_size[1] / 2,
        avg_center[0] + avg_height * output_size[0] / output_size[1] / 2
      ]).astype(np.int).clip(0, 9999)

    # crop image
    result_img = img[avg_height_range[0]:avg_height_range[1], avg_width_range[0]:avg_width_range[1]].copy()

    # resize image to output size
    result_img = cv2.resize(result_img, output_size)

    # visualize
    pt1 = (int(rect.left()), int(rect.top()))
    pt2 = (int(rect.right()), int(rect.bottom()))
    cv2.rectangle(img, pt1, pt2, (255, 255, 255), 3)

    cv2.imshow('img', img)
    cv2.imshow('result', result_img)
    # write video
    out.write(result_img)
    if cv2.waitKey(1) == ord('q'):
      exit()

  # release everything
  cap.release()
  out.release()
  cv2.destroyAllWindows()

# mouse callback for define ROI
def callback(event, x, y, flags, param):
  global img, mouse_down, top_left, bottom_right

  # mouse down
  if event == cv2.EVENT_LBUTTONDOWN:
    mouse_down = True
    top_left = (x, y)
  # mouse up
  elif event == cv2.EVENT_LBUTTONUP and mouse_down == True:
    mouse_down = False
    bottom_right = (x, y)

    # define ROI
    rect = dlib.rectangle(top_left[0], top_left[1], bottom_right[0], bottom_right[1])

    # destroy select window
    cv2.destroyWindow('Select Window')

    # start tracking
    track(img, rect)
  # mouse move
  elif event == cv2.EVENT_MOUSEMOVE and mouse_down == True:
    im_draw = img.copy()
    cv2.rectangle(im_draw, top_left, (x, y), (255,255,255), 3)
    cv2.imshow('Select Window', im_draw)

# main
ret, img = cap.read()

cv2.namedWindow('Select Window')
cv2.imshow('Select Window', img)

if lib == 'dlib':
  cv2.setMouseCallback('Select Window', callback)
  cv2.waitKey(0)

if lib != 'dlib':
  rect = cv2.selectROI('Select Window', img, fromCenter=False, showCrosshair=True)
  cv2.destroyWindow('Select Window')
  track(img, rect)



