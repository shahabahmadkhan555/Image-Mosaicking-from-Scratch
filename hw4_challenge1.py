from PIL import Image
import numpy as np
from typing import Union, Tuple, List
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt


def computeHomography(src_pts_nx2: np.ndarray, dest_pts_nx2: np.ndarray) -> np.ndarray:
    '''
    Compute the homography matrix.
    Arguments:
        src_pts_nx2: the coordinates of the source points (nx2 numpy array).
        dest_pts_nx2: the coordinates of the destination points (nx2 numpy array).
    Returns:
        H_3x3: the homography matrix (3x3 numpy array).
    '''
    # Formulate the homogeneous coordinate matrices
    src_h = np.hstack([src_pts_nx2, np.ones((src_pts_nx2.shape[0], 1))])
    dst_h = np.hstack([dest_pts_nx2, np.ones((dest_pts_nx2.shape[0], 1))])
    
    
    # Construct the matrix A for the linear equations
    A = np.zeros((src_h.shape[0] * 2, 9))
    for i in range(src_h.shape[0]):
        A[2*i] = np.array([-src_h[i, 0], -src_h[i, 1], -1, 0, 0, 0,
                            src_h[i, 0]*dst_h[i, 0], src_h[i, 1]*dst_h[i, 0], dst_h[i, 0]])
        A[2*i+1] = np.array([0, 0, 0, -src_h[i, 0], -src_h[i, 1], -1,
                             src_h[i, 0]*dst_h[i, 1], src_h[i, 1]*dst_h[i, 1], dst_h[i, 1]])
    
    # Solve the linear system
    ATA = np.dot(A.T, A)
    eigenvalues, eigenvectors = np.linalg.eig(ATA)
    min_idx = np.argmin(eigenvalues)
    H = eigenvectors[:, min_idx].reshape(3, 3)
    return H
    # raise NotImplementedError


def applyHomography(H_3x3: np.ndarray, src_pts_nx2: np.ndarray) ->  np.ndarray:
    '''
    Apply the homography matrix to the source points.
    Arguments:
        H_3x3: the homography matrix (3x3 numpy array).
        src_pts_nx2: the coordinates of the source points (nx2 numpy array).
    Returns:
        dest_pts_nx2: the coordinates of the destination points (nx2 numpy array).
    '''
    # Formulate the homogeneous coordinate matrix for test points
    tst_h = np.hstack([src_pts_nx2, np.ones((src_pts_nx2.shape[0], 1))])
    
    # Apply the homography transformation
    transformed_points = np.dot(tst_h, H_3x3.T)

    # Normalize homogeneous coordinates
    transformed_points /= transformed_points[:, 2][:, np.newaxis]
    
    # Extract (x, y) coordinates
    transformed_points = transformed_points[:, :2]

    return transformed_points
    # raise NotImplementedError


def showCorrespondence(img1: Image.Image, img2: Image.Image, pts1_nx2: np.ndarray, pts2_nx2: np.ndarray) -> Image.Image:
    '''
    Show the correspondences between the two images.
    Arguments:
        img1: the first image.
        img2: the second image.
        pts1_nx2: the coordinates of the points in the first image (nx2 numpy array).
        pts2_nx2: the coordinates of the points in the second image (nx2 numpy array).
    Returns:
        result: image depicting the correspondences.
    '''
    print(img1.shape, img2.shape)
    result = np.hstack([img1, img2])
    result = Image.fromarray(result.astype(np.uint8)).convert('RGB')
    draw = ImageDraw.Draw(result)
    print(np.array(result).shape)

    for i in range(pts2_nx2.shape[0]):
        yp0 = pts1_nx2[i][1]
        xp0 = pts1_nx2[i][0]
        yp1 = pts2_nx2[i][1]
        xp1 = pts2_nx2[i][0] + img1.shape[1]
        draw.line((xp0, yp0, xp1, yp1), fill='red', width = 4)
        
    
    # result.show()
    return result
    # raise NotImplementedError

# function [mask, result_img] = backwardWarpImg(src_img, resultToSrc_H, dest_canvas_width_height)

def backwardWarpImg(src_img: Image.Image, destToSrc_H: np.ndarray, canvas_shape: Union[Tuple, List]) -> Tuple[Image.Image, Image.Image]:
    '''
    Backward warp the source image to the destination canvas based on the
    homography given by destToSrc_H. 
    Arguments:
        src_img: the source image.
        destToSrc_H: the homography that maps points from the destination
            canvas to the source image.
        canvas_shape: shape of the destination canvas (height, width).
    Returns:
        dest_img: the warped source image.
        dest_mask: a mask indicating sourced pixels. pixels within the
            source image are 1, pixels outside are 0.
    '''
    # Get dimensions of canvas
    canvas_height, canvas_width = canvas_shape
    
    # Create black canvas
    warped_image = np.zeros((canvas_height, canvas_width, src_img.shape[2]))
    
    # Create binary mask for the mapped region
    mask = np.zeros((canvas_height, canvas_width), dtype=bool)
    
    # Iterate over each pixel in the canvas
    for y in range(canvas_height):
        for x in range(canvas_width):
            target_pixel = np.array([[x], [y], [1]])
            
            # Compute corresponding source pixel using inverse homography
            source_pixel = np.dot(destToSrc_H, target_pixel)
            source_pixel = source_pixel / source_pixel[2]
            
            # Check if source pixel is within bounds of the source image
            if 0 <= source_pixel[0] < src_img.shape[1] and 0 <= source_pixel[1] < src_img.shape[0]:
                warped_image[y, x] = src_img[int(source_pixel[1]), int(source_pixel[0])]
                mask[y, x] = 1  # Set mask value to 1 for mapped pixels
    
    return mask, warped_image
    # raise NotImplementedError


def blendImagePair(img1: List[Image.Image], mask1: List[Image.Image], img2: Image.Image, mask2: Image.Image, mode: str) -> Image.Image:
    '''
    Blend the warped images based on the masks.
    Arguments:
        img1: list of source images.
        mask1: list of source masks.
        img2: destination image.
        mask2: destination mask.
        mode: either 'overlay' or 'blend'
    Returns:
        out_img: blended image.
    '''
    # Overlay mode
    if mode == 'overlay':
        # Convert masks to binary
        mask1 = (mask1 > 0)
        mask2 = (mask2 > 0)

        # Convert images to float and normalize
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

        out_img = img1.copy()
        out_img[mask2] = img2[mask2]
        return out_img

    # Blend mode
    elif mode == 'blend':
        # Convert images to float and normalize
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0
        mask1 = mask1.astype(float)
        mask2 = mask2.astype(float)

        # Calculate distance transforms
        dist1 = distance_transform_edt(mask1)
        dist2 = distance_transform_edt(mask2)

        # Normalize distance transforms
        dist1 /= np.max(dist1)
        dist2 /= np.max(dist2)


        # Calculate weights
        weight1 = dist1
        weight2 = dist2

        blended_img = np.zeros(img1.shape)

        # Perform weighted blending
        for y in range(img1.shape[0]):
            for x in range(img1.shape[1]):
                blended_img[y][x] = (img1[y][x] * weight1[y][x] + img2[y][x] * weight2[y][x])/(weight1[y][x] + weight2[y][x])
        return blended_img
    # raise NotImplementedError

def runRANSAC(src_pt: np.ndarray, dest_pt: np.ndarray, ransac_n: int, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Run the RANSAC algorithm to find the inliers between the source and
    destination points.
    Arguments:
        src_pt: the coordinates of the source points (nx2 numpy array).
        dest_pt: the coordinates of the destination points (nx2 numpy array).
        ransac_n: the number of iterations to run RANSAC.
        eps: the threshold for considering a point to be an inlier.
    Returns:
        inliers_id: the indices of the inliers (kx1 numpy array).
        H: the homography matrix (3x3 numpy array).
    '''
    best_inliers = np.array([])  # Initialize the best set of inliers
    best_H = np.zeros((3, 3))    # Initialize the best homography matrix
    max_inliers_count = 0        # Initialize the maximum inliers count

    for _ in range(ransac_n):
        # Randomly select 4 point correspondences
        indices = np.random.choice(len(src_pt), 4, replace=False)
        src_sample = src_pt[indices]
        dest_sample = dest_pt[indices]

        # Compute the homography matrix using the 4-point correspondences
        H = computeHomography(src_sample, dest_sample)

        # Transform all source points using the computed homography
        src_transformed = applyHomography(H, src_pt)

        # Calculate Euclidean distance between transformed source points and destination points
        errors = np.linalg.norm(src_transformed - dest_pt, axis=1)

        # Count inliers based on the specified threshold (eps)
        inliers = np.where(errors < eps)[0]
        inliers_count = len(inliers)

        # Update best model if the current model has more inliers
        if inliers_count > max_inliers_count:
            best_inliers = inliers
            best_H = H
            max_inliers_count = inliers_count

    return best_inliers, best_H
    # raise NotImplementedError

def stitchImg(*args: Image.Image) -> Image.Image:
    from helpers import genSIFTMatches
    '''
    Stitch a list of images.
    Arguments:
        args: a variable number of input images.
    Returns:
        stitched_img: the stitched image.
    '''
    num = len(args);
    
    center_img_idx = 1
    
    for idx in range(1, num):
            center_img_idx = idx
            prev_idx = idx - 1
            xs, xd = genSIFTMatches(args[prev_idx], args[center_img_idx])
            xs[:,[0,1]] = xs[:, [1,0]]
            xd[:,[0,1]] = xd[:, [1,0]]
            inliers_id, H = runRANSAC(xs, xd, ransac_n=50, eps=2)
            
            inv_destToSrc_H = np.linalg.inv(H)
            x = args[prev_idx].shape[1]
            y = args[prev_idx].shape[0]
            corner_points = applyHomography(H, np.array([[0,0], [x,0], [0, y], [x, y]]))
            x_min, y_min = corner_points[0][0], corner_points[0][1]
            x_max, y_min = corner_points[1][0], min(y_min, corner_points[1][1])
            x_min, y_max = min(x_min, corner_points[2][0]), corner_points[2][1]
            x_max, y_max = max(x_max, corner_points[3][0]), max(y_max, corner_points[3][1])
            
            translation_matrix = np.array([[1, 0, x_min], [0, 1, y_min], [0, 0, 1]])
            mask, dest_img = backwardWarpImg(args[prev_idx], np.dot(inv_destToSrc_H, translation_matrix), (int(y_max-y_min), int(x_max-x_min)))
            
            width = int(args[center_img_idx].shape[1] - x_min)
            height = int(max(args[center_img_idx].shape[0], y_max - y_min))
            final_mat_src = np.zeros([height, width, 3])
            final_mask_src = np.zeros([height, width], dtype=bool)
            for i in range(height):
                for j in range(width):
                    try:
                        final_mat_src[i][j] = dest_img[i][j]
                        if not dest_img[i][j].any():
                            final_mask_src[i][j] = False
                        elif dest_img[i][j].any(): 
                            final_mask_src[i][j] = True
                    except:
                        final_mask_src[i][j] = False
            
    
            final_mat_dst = np.zeros([height, width, 3])
            final_mask_dst = np.zeros([height, width], dtype=bool)
            dst_img = args[center_img_idx]
            for i in range(dst_img.shape[0]):
                for j in range(args[center_img_idx].shape[1]-1, -1, -1):
                    new_i = i + (height//2 - args[center_img_idx].shape[0]//2)
                    new_j = j + width - args[center_img_idx].shape[1]
                    try:
                        final_mat_dst[new_i][new_j] = dst_img[i][j]
                        if not dst_img[i][j].any():
                            final_mask_dst[new_i][new_j] = False
                        elif dst_img[i][j].any(): 
                            final_mask_dst[new_i][new_j] = True
                    except:
                        final_mask_dst[new_i][new_j] = False

            
            final_mat_src = final_mat_src * 255.0
            final_mat_dst = final_mat_dst * 255.0
            blended_result = blendImagePair(final_mat_src, final_mask_src, final_mat_dst, final_mask_dst, 'blend')

            return blended_result

    # raise NotImplementedError
