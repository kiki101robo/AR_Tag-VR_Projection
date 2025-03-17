import cv2
import numpy as np
from overlay_2D import Overlay2D
from sklearn.cluster import KMeans

# 3d Overlay
class Overlay3D(Overlay2D):
    def __init__(self, K_mat) -> None:
        super().__init__()
        self.K = K_mat       
        
    # Overlay cube    
    def overlay_cube(self, frame_color, ar_tag):
        # Preprocessing
        frame = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)        
        corners, inner_tag_mask = self.get_corners(frame, scnd_filter=False)
        
        # Apply kmeans
        corners = np.array(corners)
        if len(corners)%4 == 0:
            num_tags = len(corners)//4
        else:
            num_tags = len(corners)//4 + 1
         
        try:    
            kmeans = KMeans(num_tags).fit(corners)
        except ValueError:
            return
        
        # Get camera to world transformation            
        h_tag, w_tag = ar_tag.shape[:2]
        tag_corners = np.array([[0,0], [w_tag,0], [0, h_tag], [w_tag, h_tag]])
        Hs = []
        for i in range(num_tags):
            try:
                Hs.append(self.get_homography(tag_corners, corners[kmeans.labels_==i]))
            except IndexError:
                continue
            
        # Give me Projection matrix
        P = []
        for H in Hs:
            mat = np.linalg.inv(self.K) @ np.linalg.inv(H)
            b1 = mat[:, 0]
            b2 = mat[:, 1]
            lam = np.mean([np.linalg.norm(b1), np.linalg.norm(b2)])
            r1 = b1*lam
            r2 = b2*lam
            r3 = np.cross(r1, r2)
            
            P.append(self.K @ np.array([r1, r2, r3/lam**2, mat[:,2]*lam]).T)
        
        # Define cube points
        all_pts = [
            [[0,0,0],[w_tag,0,0]],
            [[0,0,-w_tag],[w_tag,0,-w_tag]],
            [[0,h_tag,0],[w_tag,h_tag,0]],
            [[0,h_tag,-w_tag],[w_tag,h_tag,-w_tag]],
            
            [[0,0,0],[0,h_tag,0]],
            [[0,0,-w_tag],[0,h_tag,-w_tag]],
            [[w_tag,0,0],[w_tag,h_tag,0]],
            [[w_tag,0,-w_tag],[w_tag,h_tag,-w_tag]],
            
            [[0,0,0],[0,0,-w_tag]],
            [[0,h_tag,0],[0,h_tag,-w_tag]],
            [[w_tag,0,0],[w_tag,0,-w_tag]],
            [[w_tag,h_tag,0],[w_tag,h_tag,-w_tag]]
        ]
        
        # Disp cube
        int_ = lambda val: int(round(val))
        for p in P:
            for pair in all_pts:
                pt0 = p @ np.array([[pair[0][0]], [pair[0][1]], [pair[0][2]], [1]])
                pt1 = p @ np.array([[pair[1][0]], [pair[1][1]], [pair[1][2]], [1]])
                loc0 = pt0/pt0[-1]
                loc1 = pt1/pt1[-1]
                tup0 = (int_(loc0[0][0]), int_(loc0[1][0]))
                tup1 = (int_(loc1[0][0]), int_(loc1[1][0]))
                cv2.line(frame_color, tup0, tup1, (0,0,255), 2)
                        
if __name__ == '__main__':
    from os import system
    system('cls')
    ar_tag = Overlay3D.resize_frame(cv2.imread("clean_ar_tag.png"), 40)
    ar_tag = cv2.cvtColor(ar_tag, cv2.COLOR_BGR2GRAY)
    
    K = np.array([[1406.08415449821,                0, 0],
                  [2.20679787308599, 1417.99930662800, 0],
                  [1014.13643417416, 566.347754321696, 1]]).T
    
    overlay = Overlay3D(K)  
    
    # Change video name here
    video_name = "Tag1"
    video_name = "multipleTags"
        
    video_feed = cv2.VideoCapture(video_name+".mp4")
    wait = 0
    try:
        while True:
            ret, frame = video_feed.read()

            if not ret:
                break
            
            frame = overlay.resize_frame(frame, 40)
            overlay.overlay_cube(frame, ar_tag)
            
            img = frame
            overlay.frames.append(img)
            cv2.imshow("img", img)
            k = cv2.waitKey(wait) & 0xff
            if k == ord('q'):
                break
            elif k == ord('s'):
                if wait == 0:
                    wait = 1
                else:
                    wait = 0
        video_feed.release()
        overlay.save(video_name+'_cube.avi', True)
    except KeyboardInterrupt:
        pass
