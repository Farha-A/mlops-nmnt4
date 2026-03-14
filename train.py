import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
from mlflow.models import infer_signature

torch.manual_seed(13)
batch_size = 128

dataloader = DataLoader(
    FashionMNIST(".", download=True, transform=transforms.ToTensor()),
    batch_size=batch_size,
    shuffle=True,
)


def tensor_to_images(image_tensor, num_images=30, size=(1, 28, 28)):
    return


class Generator(nn.Module):
    def __init__(
        self, noise_dimension=64, image_dimension=784, hidden_dimension=128
    ):
        super(Generator, self).__init__()

        self.gen = nn.Sequential(
            self.generator_block(noise_dimension, hidden_dimension),
            self.generator_block(hidden_dimension, hidden_dimension * 2),
            self.generator_block(hidden_dimension * 2, hidden_dimension * 4),
            self.generator_block(hidden_dimension * 4, hidden_dimension * 8),
            nn.Linear(hidden_dimension * 8, image_dimension),
            nn.Tanh(),
        )

    def forward(self, noise):
        return self.gen(noise)

    def generator_block(self, in_dimension, out_dimension):
        return nn.Sequential(
            nn.Linear(in_dimension, out_dimension),
            nn.BatchNorm1d(out_dimension),
            nn.ReLU(inplace=True),
        )


def get_noise(n_samples, noise_vector_dimension, device="cpu"):
    return torch.randn(n_samples, noise_vector_dimension, device=device)


class Discriminator(nn.Module):
    def __init__(self, image_dimension=784, hidden_dimension=128):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            self.discriminator_block(image_dimension, hidden_dimension * 4),
            self.discriminator_block(
                hidden_dimension * 4, hidden_dimension * 2
            ),
            self.discriminator_block(hidden_dimension * 2, hidden_dimension),
            nn.Linear(hidden_dimension, 1),
        )

    def forward(self, image):
        return self.disc(image)

    def discriminator_block(self, in_dimension, out_dimension):
        return nn.Sequential(
            nn.Linear(in_dimension, out_dimension), 
            nn.LeakyReLU(0.2, inplace=True)
        )


n_epochs = 3
noise_dimension = 64
lr = 0.00001
display_step = 500
hidden_dimension = 128

criterion = nn.BCEWithLogitsLoss()
device = "cuda" if torch.cuda.is_available() else "cpu"


# Generator & Optimizer for Generator
gen = Generator(noise_dimension).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)

# Discriminator & Optimizer for Discriminator
disc = Discriminator().to(device)
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)


def get_disc_loss(
    gen, disc, criterion, real, num_images, noise_dimension, device
):
    # Generate noise and pass to generator
    fake_noise = get_noise(num_images, noise_dimension, device=device)
    fake = gen(fake_noise)

    # Pass fake features to discriminator
    # All of them will got label as 0
    # .detach() ensures that only discriminator parameters will get update
    disc_fake_pred = disc(fake.detach())
    disc_fake_loss = criterion(disc_fake_pred, 
    torch.zeros_like(disc_fake_pred))

    # Pass real features to discriminator
    # All of them will got label as 1
    disc_real_pred = disc(real)
    disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))

    # Average of loss from both real and fake features
    disc_loss = (disc_fake_loss + disc_real_loss) / 2
    return disc_loss


def get_gen_loss(gen, disc, criterion, num_images, noise_dimension, device):
    # Generate noise and pass to generator
    fake_noise = get_noise(num_images, noise_dimension, device=device)
    fake = gen(fake_noise)

    # Pass fake features to discriminator
    # But all of them will got label as 1
    disc_fake_pred = disc(fake)
    gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
    return gen_loss


cur_step = 0
mean_generator_loss = 0
mean_discriminator_loss = 0
gen_loss = False

for epoch in range(n_epochs):
    epoch_gen_loss = 0
    epoch_disc_loss = 0
    num_batches = 0

    for real, _ in tqdm(dataloader):
        # Get number of batch size (number of image)
        # And get tensor for each image in batch
        cur_batch_size = len(real)
        real = real.view(cur_batch_size, -1).to(device)

        # Training discriminator
        disc_opt.zero_grad()
        disc_loss = get_disc_loss(
            gen, disc, criterion, real, cur_batch_size, noise_dimension, device
        )
        disc_loss.backward(retain_graph=True)
        disc_opt.step()

        # Training generator
        gen_opt.zero_grad()
        gen_loss = get_gen_loss(
            gen, disc, criterion, cur_batch_size, noise_dimension, device
        )
        gen_loss.backward()
        gen_opt.step()

        # Accumulate losses for epoch-level logging
        epoch_gen_loss += gen_loss.item()
        epoch_disc_loss += disc_loss.item()
        num_batches += 1

        mean_discriminator_loss += disc_loss.item() / display_step
        mean_generator_loss += gen_loss.item() / display_step

        if cur_step % display_step == 0 and cur_step > 0:
            print(
                f"Step {cur_step}: Generator loss: {mean_generator_loss}, "
                f"discriminator loss: {mean_discriminator_loss}"
            )
            fake_noise = get_noise(
                cur_batch_size, noise_dimension, device=device
            )
            fake = gen(fake_noise)
            tensor_to_images(fake)
            tensor_to_images(real)
            mean_generator_loss = 0
            mean_discriminator_loss = 0

        cur_step += 1

    # Live Logging: log average loss at end of every epoch
    avg_gen_loss = epoch_gen_loss / num_batches
    avg_disc_loss = epoch_disc_loss / num_batches
    print(
        f"Epoch {epoch+1}/{n_epochs} - Avg Gen Loss: {avg_gen_loss:.4f}, "
        f"Avg Disc Loss: {avg_disc_loss:.4f}"
    )

gen.eval()
disc.eval()

gen_input_example = torch.randn(1, noise_dimension).to(device)
# Get a sample output to infer the schema
with torch.no_grad():
    gen_output = gen(gen_input_example)

gen_signature = infer_signature(
    gen_input_example.cpu().numpy(), gen_output.cpu().numpy()
)

# 3. Prepare Discriminator Example & Signature
disc_input_example = torch.randn(1, 784).to(device)
with torch.no_grad():
    disc_output = disc(disc_input_example)

disc_signature = infer_signature(
    disc_input_example.cpu().numpy(), disc_output.cpu().numpy()
)
