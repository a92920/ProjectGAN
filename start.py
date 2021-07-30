from PIL import Image
import numpy as np
import glob
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch

def get_imgs(file_loc,reshape = (227,227)):

    train_images = get_img_file(file_loc["train"],'JPG',reshape)
    test_images = get_img_file(file_loc["test"],'png',reshape)
    test_images_corrupt = get_img_file(file_loc["test_corr"],'png',reshape)

    return train_images, test_images,test_images_corrupt
    
def get_img_file(loc ,ext ,reshape = (227,227)):
    # !REMOVE [:100] TO GET ALL FILES
    loc = glob.glob(f"{loc}/*.{ext}")
    images = []
    for img_loc in loc:
        pil_img = Image.open(img_loc)
        pil_img.draft('RGB',reshape)
        pil_img = pil_img.resize(reshape)
        images.append(np.asarray(pil_img))        
    images = np.array(images)
    return images

def white_out_center(img, cover):
    # Input image,cover shape
    # Output image with white in center , image original cropped image
    img_cp = img.copy()
    shape = img.shape
    corner = [(shape[0]-cover[0])//2,(shape[1]-cover[1])//2]
    img_cp[corner[0]:cover[0]+corner[0],corner[1]:cover[1]+corner[1]] = 255 
    
    return img_cp, img[corner[0]:cover[0]+corner[0],corner[1]:cover[1]+corner[1]]

class ToFillDataset_train(Dataset):
    def __init__(self, images, cover, corrupted_images=None):
        self.images = images
        self.corrupted_images = corrupted_images
        self.cover = cover
        
    # def corrupt_image(self, img):
    #     img = img.copy()
    #     corver = []
    #     for i in range(1):
    #         img_perc_covered = img.flatten()
    #         img_perc_covered = np.count_nonzero( img_perc_covered == 255) / img_perc_covered.shape[0]
    #         if(img_perc_covered < 0.25):
    #             cover = [50,50]
    #             corner = [np.random.randint(0,227-cover[0]), np.random.randint(0,227-cover[1])]
    #             corver.append(corner)
    #             img[corner[0]:cover[0]+corner[0],corner[1]:cover[1]+corner[1]] = 255
                
    #         else:
    #             img_perc_covered = img.flatten()
    #             if (i <= 8):
    #                 for left in range(8-i):
    #                     corver.append([0,0])
    #                 break
    #     corver = torch.tensor(corver)
    #     return img, corver
    
    def __len__(self):
        return len(self.images)
    
    def normalize(self,img):
        img = img / 255
        return img
    
    def __getitem__(self, item):
        image = self.images[item]
        #image = self.normalize(image)
        if type(self.corrupted_images) == int :
            corrupt_images, original_cropped = white_out_center(self.images[item],self.cover)
            
        else:
            corrupt_images = self.corrupted_images[item]
            
        return {
          # original image  
          'image': image,
          # with center white
          'corrupt_image': corrupt_images,
          # original image center cropped
          'corrupt_image_center': original_cropped,
        }

def create_data_loader(images ,cover,corrupt_images = 0, batch_size = 16 ):
    ds = ToFillDataset_train(images, cover,corrupt_images)
    return DataLoader(ds, batch_size) 


if __name__ == "__main__":
    train_loc = "paris_train_original.nosync"
    train_images = get_imgs(train_loc,'JPG')

    test_loc = "paris_eval.nosync/paris_eval_gt"
    test_images = get_imgs(test_loc,'png')

    test_loc_corrupt = "paris_eval.nosync/paris_eval_gt"
    test_images_corrupt = get_imgs(test_loc_corrupt,'png')

#    print(train_images.shape,test_images.shape,test_images_corrupt.shape)

    train_data_loader = create_data_loader(train_images)
    test_data_loader = create_data_loader(test_images)

    # cover = [150,150]
    # for count,data in enumerate(train_data_loader): 
    #     fig = plt.figure(figsize=(16,8))
    #     corner = data['covers']
    #     img = data["image"]
    #     print(corner)
    #     for k in range(len(img)):
    #         for w in range(len(corner[k])):
    #             image_selected = img[k]
    #             print(corner[k][w][0])
    #             print(corner[k][w][1])
    #             image_selected = image_selected[corner[k][w][0]:cover[0]+corner[k][w][0],corner[k][w][1]:cover[1]+corner[k][w][1]]
    #             fig.add_subplot(2, 16, k+1)
    #             plt.imshow(image_selected)
    #             fig.add_subplot(2, 16, k+2)
    #             plt.imshow(img[k])

    #     plt.show()