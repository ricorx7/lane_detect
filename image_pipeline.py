import cv2
import numpy as np
from line import Line

# State if you want to convert the lines into solid lines
solid_lines = True
temporal_smoothing = True

def pipeline(image_path):
    image = cv2.imread(image_path)

    # Set image sizes
    img_h, img_w = image.shape[0], image.shape[1]

    # Crop image to remove background
    height, width, channels = image.shape
    #print(image.shape)
    cropTop = int(round(height * 0.45))
    cropBottom = int(round(height * 1.0))
    cropped = image[cropTop:cropBottom, :, :]

    # convert to grayscale
    img_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    # perform gaussian blur
    #17,17 deafualt kernel
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # perform edge detection
    img_edge = cv2.Canny(img_blur, threshold1=50, threshold2=80)

    # perform hough transform
    detected_lines = cv2.HoughLinesP(img_edge,
                                     2,
                                     np.pi / 180,
                                     1,
                                     np.array([]),
                                     minLineLength=7,      # default 15
                                     maxLineGap=5)          # default 5

    # convert (x1, y1, x2, y2) tuples into Lines
    detected_lines = [Line(l[0][0], l[0][1], l[0][2], l[0][3]) for l in detected_lines]

    # draw lanes found
    # prepare empty mask on which lines are drawn
    #line_img = np.zeros(shape=(img_h, img_w))
    line_img = image
    for lane in detected_lines:
        if 0.1 <= np.abs(lane.slope) <= 4:
            lane.draw(line_img, offset_y=cropTop)       # Add in cropped location

    # Return the image with lines drawn on it
    return line_img

def compute_lane_from_candidates(line_candidates, img_shape):
    """
    Compute lines that approximate the position of both road lanes.

    :param line_candidates: lines from hough transform
    :param img_shape: shape of image to which hough transform was applied
    :return: lines that approximate left and right lane position
    """

    # separate candidate lines according to their slope
    pos_lines = [l for l in line_candidates if l.slope > 0]
    neg_lines = [l for l in line_candidates if l.slope < 0]

    # interpolate biases and slopes to compute equation of line that approximates left lane
    # median is employed to filter outliers
    neg_bias = np.median([l.bias for l in neg_lines]).astype(int)
    neg_slope = np.median([l.slope for l in neg_lines])
    x1, y1 = 0, neg_bias
    x2, y2 = -np.int32(np.round(neg_bias / neg_slope)), 0
    left_lane = Line(x1, y1, x2, y2)

    # interpolate biases and slopes to compute equation of line that approximates right lane
    # median is employed to filter outliers
    lane_right_bias = np.median([l.bias for l in pos_lines]).astype(int)
    lane_right_slope = np.median([l.slope for l in pos_lines])
    x1, y1 = 0, lane_right_bias
    x2, y2 = np.int32(np.round((img_shape[0] - lane_right_bias) / lane_right_slope)), img_shape[0]
    right_lane = Line(x1, y1, x2, y2)

    return left_lane, right_lane

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    Returns resulting blend image computed as follows:

    initial_img * α + img * β + λ
    """
    img = np.uint8(img)
    if len(img.shape) is 2:
        img = np.dstack((img, np.zeros_like(img), np.zeros_like(img)))

    return cv2.addWeighted(initial_img, α, img, β, λ)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """

    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)

    return masked_image, mask

def debug_pipeline():
    # Read Image
    image = cv2.imread('test_image_2.jpg')

    img_h, img_w = image.shape[0], image.shape[1]

    # Crop image to remove background
    height, width, channels = image.shape
    print(image.shape)
    cropTop = int(round(height * 0.45))
    cropBottom = int(round(height * 1.0))
    cropped = image[cropTop:cropBottom, :, :]
    cv2.imwrite('1_cropped.jpg', cropped)
    print('Cropped')
    #cropped = image


    # convert to grayscale
    img_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("2_grayscale.jpg", img_gray)
    print("Grayscale")

    # perform gaussian blur
    #17,17 deafualt kernel
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    cv2.imwrite("3_blur.jpg", img_blur)
    print("Blur")

    # perform edge detection
    img_edge = cv2.Canny(img_blur, threshold1=50, threshold2=80)
    cv2.imwrite("4_canny.jpg", img_edge)
    print("Canny")

    # perform hough transform
    detected_lines = cv2.HoughLinesP(img_edge,
                                     2,
                                     np.pi / 180,
                                     1,
                                     np.array([]),
                                     minLineLength=7,      # default 15
                                     maxLineGap=5)          # default 5

    print("Lines")
    print(detected_lines)

    # convert (x1, y1, x2, y2) tuples into Lines
    detected_lines = [Line(l[0][0], l[0][1], l[0][2], l[0][3]) for l in detected_lines]
    print("Detected Lines: ", len(detected_lines))

    # if 'solid_lines' infer the two lane lines
    #if solid_lines:
    #    candidate_lines = []
    #    for line in detected_lines:
    #        # consider only lines with slope between 30 and 60 degrees
    #        if 0.5 <= np.abs(line.slope) <= 2:
    #            candidate_lines.append(line)
    #            print("candidate line found")
    #    # interpolate lines candidates to find both lanes
    #    lane_lines = compute_lane_from_candidates(candidate_lines, img_gray.shape)
    #else:
    #    # if not solid_lines, just return the hough transform output
    #    lane_lines = detected_lines

    #sorted_lines = []
    #for line in detected_lines:
    #    print("Line Slope: ", line.slope)
    # consider only lines with slope between 30 and 60 degrees
    #    if 0.1 <= np.abs(line.slope) <= 4:      # 0.2 to 2
    #        sorted_lines.append(line)
    #print("Sort out the lines")
    #print("Lane Line Count: ", len(lane_lines))
    #lane_lines = compute_lane_from_candidates(sorted_lines, img_gray.shape)
    lane_lines = detected_lines

    # draw lanes found
    # prepare empty mask on which lines are drawn
    #line_img = np.zeros(shape=(img_h, img_w))
    line_img = image
    for lane in lane_lines:
        print("Slope: ", lane.slope)
        if 0.1 <= np.abs(lane.slope) <= 4:
            #print("Selected LIne: X1", lane.x1)
            #print("Selected LIne: Y1", lane.y1)
            #print("Selected LIne: X2", lane.x2)
            #print("Selected LIne: Y2", lane.y2)
            lane.draw(line_img, offset_y=cropTop)       # Add in cropped location
    cv2.imwrite("5_lines.jpg", line_img)
    print("Lines drawn")

    """
    # keep only region of interest by masking
    vertices = np.array([[(50, img_h),
                          (450, 310),
                          (490, 310),
                          (img_w - 50, img_h)]],
                        dtype=np.int32)
    img_masked, _ = region_of_interest(line_img, vertices)
    print("image mask")
    cv2.imwrite("img_mask.jpg", img_masked)
    
    img_blend = weighted_img(img_masked, image, α=0.8, β=1., λ=0.)
    print("Final image drawn")
    cv2.imwrite("final.jpg", img_blend)
    """

if __name__ == "__main__":
    debug_pipeline()