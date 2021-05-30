from PIL import Image
import numpy as np
import glob
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

def get_imgs(loc ,ext ,to_file = 0):
    loc = glob.glob(f"{loc}/*.{ext}")[:100]
    images = []
    for img_loc in loc:
        pil_img = Image.open(img_loc)
        pil_img.draft('RGB',(227,227))
        pil_img = pil_img.resize((227,227))
        images.append(np.asarray(pil_img))        
    images = np.array(images)
    return images

class ToFillDataset_train(Dataset):
    def __init__(self, images, corrupted_images=None):
        self.images = images
        self.corrupted_images = corrupted_images
        
    def corrupt_image(self, img):
        img = img.copy()
        for i in range(8):
            img_perc_covered = img.flatten()
            img_perc_covered = np.count_nonzero( img_perc_covered == 255) / img_perc_covered.shape[0]
            if(img_perc_covered < 0.25):
                cover = [150,150]
                corner = [np.random.randint(0,537-cover[0]), np.random.randint(0,936-cover[1])] 
                img[corner[0]:cover[0]+corner[0],corner[1]:cover[1]+corner[1]] = 255
            else:
                img_perc_covered = img.flatten()
                break
        return img
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, item):
        image = self.images[item]
        
        if type(self.corrupted_images) == int :
            corrupt_images = self.corrupt_image(self.images[item])
        else:
            corrupt_images = self.corrupted_images[item]
            
        
        return {
          'image': image,
          'corrupt_image': corrupt_images ,
        }

def create_data_loader(images ,corrupt_images = 0 ):
    ds = ToFillDataset_train(images, corrupt_images)
    return DataLoader(ds, batch_size = 16) 


if __name__ == "__main__":
    train_loc = "paris_train_original.nosync"
    train_images = get_imgs(train_loc,'JPG')

    test_loc = "paris_eval.nosync/paris_eval_gt"
    test_images = get_imgs(test_loc,'png')

    test_loc_corrupt = "paris_eval.nosync/paris_eval_gt"
    test_images_corrupt = get_imgs(test_loc_corrupt,'png')

    print(train_images.shape,test_images.shape,test_images_corrupt.shape)

    train_data_loader = create_data_loader(train_images)
    test_data_loader = create_data_loader(test_images)

