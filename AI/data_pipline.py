import cv2
import os



CONFIG = {
    "new_width" : 1280,
    "new_height": 720,
}

class Image():
    def __init__(self,path_to_data_folder:str):
        self.path_to_data_folder = path_to_data_folder
    
    def __len__(self):
        return len(os.walk(self.path_to_data_folder))
    
    def  preprocess_image(self,file_inside=None):
       if file_inside != None:
          print(os.path.join(self.path_to_data_folder,file_inside))
          img = cv2.imread(os.path.join(self.path_to_data_folder,file_inside))

          new_resolution = (CONFIG["new_width"],CONFIG["new_height"])

          resized_img = cv2.resize(img, new_resolution, interpolation=cv2.INTER_AREA) 

          gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

          return gray_img

       else:
           Warning(f"file_inside == None ")


           