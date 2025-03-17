import os
import cv2
import numpy as np

# This class handles simple image and video tasks
class VideoBuffer():
    def __init__(self, framerate = 10, scale_percent=100):
        self.frames = []
        self.framerate = framerate
        self.scale_percent = scale_percent
      
    # Resizes each length image at a given percent. E.g. 200 will double each dimension and 50 will half it
    @staticmethod
    def resize_frame(grid, scale_percent=100):
        width = int(grid.shape[1] * scale_percent / 100)
        height = int(grid.shape[0] * scale_percent / 100)
        dim = (width, height)
        return cv2.resize(grid, dim, interpolation = cv2.INTER_AREA) 
      
    # Write frames to feed and save video
    def save(self, video_name, isColor = False):
        shape = self.frames[0].shape[:2]
        size = (shape[1], shape[0])
        videowriter = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'),self.framerate , size, isColor)
        [videowriter.write(f) for f in self.frames]
        videowriter.release()

# Corner finder class
class CornerFinder(VideoBuffer):
    def __init__(self) -> None:
        super().__init__()
    
    # Get corner function
    def get_corners(self, frame, buffer=10, k_size = 41, scnd_filter = True):    

        # Apply Gaussian Blur then threshold the image to extract paper location
        blur = cv2.GaussianBlur(frame, (11,11), 0)
        _, thresholded = cv2.threshold(blur, 230, 255, cv2.THRESH_BINARY)
        
        # White out inside of paper
        structure = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size,k_size))
        closed = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, structure)
                    
        # Use the whited-out paper region as a mask for the inner ar tag. Then white out the center again             
        inv_thresh = 255 - thresholded
        inner_tag = cv2.bitwise_and(inv_thresh, inv_thresh, mask=closed)
        inner_tag = cv2.morphologyEx(inner_tag, cv2.MORPH_CLOSE, structure)
        
        # Get corners
        corners = cv2.cornerHarris(inner_tag, 10, 11, .05)
        corner_locs = corners>.1*corners.max()
        
        corner_img = np.zeros_like(closed, np.uint8)
        corner_img[corner_locs] = 255
        
        # Get corner centroids
        num_regions, _, stats, centroids = cv2.connectedComponentsWithStats(corner_img)
        corner_filter_1 = np.argpartition(stats[:, cv2.CC_STAT_AREA], num_regions-1)[:num_regions-1]         
        filtered_1 = centroids[corner_filter_1]

        cv2.imshow("crn",corner_img)
        try:               
            final_corners = [(int(round(corner[0])),int(round(corner[1]))) for corner in filtered_1]
            return final_corners, inner_tag
                        
        except TypeError:
            print("Failed")

        return None
        
        
if __name__ == '__main__':
    os.system("cls")
    # Change name here
    video_feed = cv2.VideoCapture("Tag0.mp4")
    corner_finder = CornerFinder()
    wait = 1
    try:
        while True:
            ret, frame = video_feed.read()

            if not ret:
                break
            
            frame = corner_finder.resize_frame(frame, 50)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img = corner_finder.get_corners(frame)
            
            # img = np.vstack((frame, img))
            img = frame
            
            img = corner_finder.resize_frame(img, 70)
            cv2.imshow("img", img)
            k = cv2.waitKey(wait) & 0xff
            if k == ord('q'):
                break
            elif k == ord('s'):
                if wait == 0:
                    wait = 1
                else:
                    wait = 0
    except KeyboardInterrupt:
        pass
    