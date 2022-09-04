import torch
import torch.nn as nn
from torchvision.utils import save_image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2

class Generator(nn.Module):
    def __init__(self, img_size, latent_dim, output_channels):
        super(Generator,self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim, 128 * self.init_size ** 2)
        )

        # Can try using other upsample methods instead of bicubic
        # Bicubic leads to best looking results
        # Nearest is faster to train
        self.conv_blocks = nn.Sequential(
            nn.Upsample(scale_factor=2,mode="bicubic",align_corners=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2,mode="bicubic",align_corners=True),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, output_channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self,input):
        out = self.l1(input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        return self.conv_blocks(out)

class Discriminator(nn.Module):
    def __init__(self,img_size, input_channels):
        super(Discriminator,self).__init__()

        def discrim_block(in_filters, out_filters, batch_norm=True):
            block = [
                nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25)
            ]

            if batch_norm:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.blocks = nn.Sequential(
            *discrim_block(input_channels, 16, False),
            *discrim_block(16,32),
            *discrim_block(32,64),
            *discrim_block(64,128),
        )

        down_sampled_size = img_size // 2 ** 4
        # Instead getting a value between 0 and 1, we use MSE loss instead of BCE
        self.adversarial_layer = nn.Linear(128 * down_sampled_size ** 2, 1)

    def forward(self,img):
        out = self.blocks(img)
        out = out.view(out.shape[0], -1)
        return self.adversarial_layer(out)

if __name__ == '__main__':
    # Hyper parameters
    n_epochs = 400
    batch_size = 128
    lr = 2e-4
    # Beta decays for Adam
    b1 = 0.5
    b2 = 0.99
    # Dimension of the noise
    latent_dim = 100
    # Size of the image
    img_size = 64
    # Channels in the image
    channels = 3
    # If we want to view the image get generated
    view_progress = False

    # Device to put everything on
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load the images
    # Turns the image into a tensor and then put between 0 and 1
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    # Loads the dataset and will apply the transform when we iterate
    dataset = datasets.ImageFolder("./cat", transform=transform)
    # Loads the data
    data_loader = DataLoader(dataset, batch_size = batch_size, shuffle=True)

    # Create the generator and the discriminator
    gen = Generator(img_size, latent_dim, channels).to(device)
    discrim = Discriminator(img_size, channels).to(device)

    # Optimizers
    generator_optim = torch.optim.Adam(gen.parameters(), lr=lr, betas=(b1,b2))
    discriminator_optim = torch.optim.Adam(discrim.parameters(), lr=lr, betas=(b1,b2))

    # Using MSE loss over BCE loss
    loss = nn.MSELoss()

    # tqdm for progress tracking
    for epoch in tqdm(range(n_epochs), position = 0, desc = "Epoch"):
        for i, (images, _) in enumerate(tqdm(data_loader, position = 1, desc = "Batch", leave = False)):
            images = images.to(device)
            # Ground truths
            real = torch.ones((images.shape[0], 1), requires_grad=False).to(device)
            fake = torch.zeros((images.shape[0], 1), requires_grad=False).to(device)

            # Zero gradietns for generator
            generator_optim.zero_grad()

            # Noise for generator input
            noise = torch.randn((images.shape[0], latent_dim)).to(device)

            # Generated images
            generated_images = gen(noise)

            # Generator loss calculation
            generator_loss = loss(discrim(generated_images), real)

            # Back prop
            generator_loss.backward()
            generator_optim.step()

            # Zero gradients for discriminator
            discriminator_optim.zero_grad()

            # Caluclate real and fake loss
            real_loss = loss(discrim(images), real)
            fake_loss = loss(discrim(generated_images.detach()), fake)

            # Average the loss
            discriminator_loss = 0.5 * (real_loss + fake_loss)

            # Back prop
            discriminator_loss.backward()
            discriminator_optim.step()

            # If we wanna see an image being while working
            if view_progress:
                view_image = gen(torch.randn(1, latent_dim).to(device)).clone().detach().cpu().numpy()
                view_image = view_image.squeeze().transpose((1, 2, 0))
                view_image = cv2.cvtColor(view_image, cv2.COLOR_RGB2BGR)
                view_image = cv2.resize(view_image, (512, 512), cv2.INTER_CUBIC)
                norm_image = cv2.normalize(view_image, None, alpha=0, beta=1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
                cv2.imshow("View", view_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break

        # Generate 36 random images and save them to a folder
        view_images = gen(torch.randn(36, latent_dim).to(device))
        save_image(view_images.data, f"./images/{epoch:04}.png", nrow = 6, normalize= True)
