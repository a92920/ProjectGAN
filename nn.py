from torch import nn, optim
from torch.nn.modules.linear import Linear
from torchsummary import summary
import torch
from start import *
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

"""
Variables:  
            - image input cropped size (227/256/128)
            - image cover size (multiple covers in v2)
            - reshape 
            - cover 
            
Hyper Parameters: 
            - epochs
            - batch size
            - learning rate 
            - AdamOptimizer
            - autoencoder loss
            - adversarial loss 
"""


# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         nn.init.normal_(m.weight.data, 0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         nn.init.normal_(m.weight.data, 1.0, 0.02)
#         nn.init.constant_(m.bias.data, 0)



class autoencoder(nn.Module):
    def __init__(self, ngpu=0):
        super(autoencoder, self).__init__()
        self.ngpu = ngpu
        self.encoder_decoder = nn.Sequential(
            #Encoder
            
            nn.Conv2d(3,64, kernel_size=(4,4),stride=(2,2),padding=(1,1) ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(64,64, kernel_size=(4,4),stride=(2,2), padding=(1,1) ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(64,128, kernel_size=(4,4),stride=(2,2), padding=(1,1) ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(128,256, kernel_size=4,stride=(2,2), padding=(1,1) ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(256,512, kernel_size=4,stride=(2,2), padding=(1,1) ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(512,4000, kernel_size=4,stride=(2,2), padding=(1,1) ),
            nn.BatchNorm2d(4000),
            nn.LeakyReLU(0.2,inplace=True),

            #Decoder

            nn.ConvTranspose2d(4000, 512, kernel_size = (4,4), stride=(2,2), padding = (1,1) ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size = (4,4), stride=(2,2), padding = (1,1) ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2,inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size = (4,4), stride=(2,2), padding = (1,1) ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size = (4,4), stride=(2,2), padding = (1,1) ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size = (4,4), stride=(2,2), padding = (1,1) ),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.2,inplace=True),
        )
    def forward(self, input):
        #already must be converted to 128x128x3
        output = self.encoder_decoder(input)
        #print(output.shape)
        return output
    
class Adversarial_Discriminator(nn.Module):
    def __init__(self, ngpu=0):
        super(Adversarial_Discriminator, self).__init__()
        self.ngpu = ngpu
        self.discriminator = nn.Sequential(
            nn.Conv2d(3,64, kernel_size=(4,4),stride=(2,2),padding=(1,1) ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(64,128, kernel_size=(4,4),stride=(2,2), padding=(1,1) ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(128,256, kernel_size=4,stride=(2,2), padding=(1,1) ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(256,512, kernel_size=4, stride=(2,2), padding=(1,1) ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(512,1, kernel_size=4 ),
            nn.Sigmoid()
        )
    def forward(self, input):
        output = self.discriminator(input)
        return output



import torch.nn as nn




class train_context_encoder():
    def __init__(self, encoder_decoder,discriminator, train_data):
        self.encoder_decoder = encoder_decoder
        self.train_data = train_data
        self.discriminator = discriminator

    def fit(self,loss_fn_autoe, loss_fn_discriminator, optimizerAe,optimizerD, n_epochs=10, eval_data = None):
        autoe_losses = []
        D_losses = []
        encoder_decoder = self.encoder_decoder
        netD = self.discriminator

        real_label = 1
        fake_label = 0
        for epoch in range(1,int(n_epochs)+1):
            with tqdm(self.train_data, unit='batch') as tepoch:
                for data in tepoch:
                    tepoch.set_description(f"Epoch {epoch}")
                    
                    images = data["image"]
                    images = torch.reshape(images,(images.shape[0],3,128,128))
                    images = images.type(torch.FloatTensor)

                    cropped_original_image = data["corrupt_image_center"].type(torch.FloatTensor)
                    cropped_original_image = torch.reshape(cropped_original_image,(cropped_original_image.shape[0],3,64,64))
                    cropped_original_image = cropped_original_image / 255


                    ############################
                    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                    ###########################

                    ## Train with all-real batch
                    netD.zero_grad()
                    # Forward pass real batch through D
                    output = netD(cropped_original_image).view(-1)
                    # Calculate loss on all-real batch

                    label = torch.full((cropped_original_image.shape[0],), real_label, dtype=torch.float)

                    errD_real = loss_fn_discriminator(output, label)
                    # Calculate gradients for D in backward pass
                    errD_real.backward()
                    D_x = output.mean().item()

                    ## Train with all-fake batch
                    
                    # Generate fake image batch with G
                    fake = encoder_decoder(images)
                    label.fill_(fake_label)
                    # Classify all fake batch with D
                    output = netD(fake.detach()).view(-1)
                    # Calculate D's loss on the all-fake batch
                    errD_fake = loss_fn_discriminator(output, label)
                    # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                    errD_fake.backward()
                    D_G_z1 = output.mean().item()
                    # Compute error of D as sum over the fake and the real batches
                    errD = errD_real + errD_fake
                    # Update D
                    optimizerD.step()

                    ############################
                    # (2) Update G network: maximize log(D(G(z)))
                    ###########################
                    
                    encoder_decoder.zero_grad()
                    label.fill_(real_label)  # fake labels are real for generator cost
                    # Since we just updated D, perform another forward pass of all-fake batch through D
                    output = netD(fake).view(-1)
                    # Calculate G's loss based on this output
                    errG = loss_fn_autoe(output, label)
                    # Calculate gradients for G
                    errG.backward()
                    loss_ae = output.mean().item()
                    # Update G
                    optimizerAe.step()


                    # out = encoder_decoder(images)
                    # out = torch.reshape(out, (out.shape[0],64,64,3))

                    #plt.imshow(data["corrupt_image_center"][0].type(torch.FloatTensor)/255)
                    #plt.show()


                    
                    # losses.append(loss.item())
                    # loss.backward()
                    # optimizer.step()
                    # optimizer.zero_grad()
                    tepoch.set_postfix(loss_discriminator = errD.item(), loss_ae = loss_ae)        
    
if __name__ == "__main__":

    # Hparams
    file_locs = {
        "train": "paris_train_original.nosync",
        "test": "paris_eval.nosync/paris_eval_gt",
        "test_corr": "paris_eval.nosync/paris_eval_gt",
    }
    reshape = (128,128) #reshape image to size (m,n)
    cover = (64,64) # size of white shaded area
    batch_size = 16 
    epochs = 40
    learning_rate = 0.001

    autoe = autoencoder()
    discriminator = Adversarial_Discriminator()
    train_images, test_images,test_images_corrupt  = get_imgs(file_locs,reshape) 
    train_data_loader = create_data_loader(train_images, cover, batch_size = batch_size)

    summary(autoe,(3,128,128))
    summary(discriminator,(3,64,64))

    trainer = train_context_encoder(autoe,discriminator,train_data_loader)
    trainer.fit(nn.MSELoss(),nn.BCELoss(),optim.AdamW(autoe.parameters(),lr=learning_rate), optim.AdamW(discriminator.parameters(),lr=learning_rate),epochs)
    