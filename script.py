
import numpy as np
import torch
# get mnist data

from torchvision import datasets, transforms
# get mnist data and transform to tensor, to the right device.
transform = transforms.Compose([
        transforms.ToTensor(),
        ])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

labels = mnist_trainset.targets.numpy()
# remove half of the 9s from the training set, to have a more balanced dataset
# we will use this dataset to train the encoder
# get the indices of the 9s
nines = np.where(labels==9)[0]
# shuffle the indices
np.random.shuffle(nines)
# keep only half of the indices
nines = nines[:len(nines)//2]
# remove the 9s from the training set
mnist_trainset.data = torch.cat([mnist_trainset.data[labels!=9], mnist_trainset.data[nines]])
mnist_trainset.targets = torch.cat([mnist_trainset.targets[labels!=9], mnist_trainset.targets[nines]])

# plot the distribution of labels, with one color for each label
labels = mnist_trainset.targets.numpy()

# keep only percentage of the training set
percentage = 0.1
mnist_trainset.data = mnist_trainset.data[:int(len(mnist_trainset.data)*percentage)]
mnist_trainset.targets = mnist_trainset.targets[:int(len(mnist_trainset.targets)*percentage)]
mnist_testset.data = mnist_testset.data[:int(len(mnist_testset.data)*percentage)]
mnist_testset.targets = mnist_testset.targets[:int(len(mnist_testset.targets)*percentage)]
print(f"Number of images in the training set: {len(mnist_trainset)}")
print(f"Number of images in the test set: {len(mnist_testset)}")

class SamplingLayer(torch.nn.Module):
    def __init__(self):
        super(SamplingLayer, self).__init__()

    def forward(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

class VariationalEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VariationalEncoder, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(8)  # Batch Normalization after the first convolution
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(16)  # Batch Normalization after the second convolution
        self.conv3 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(32)  # Batch Normalization after the third convolution

        self.fc1 = torch.nn.Linear(512, hidden_dim)
        self.mu = torch.nn.Linear(hidden_dim, latent_dim)
        self.logvar = torch.nn.Linear(hidden_dim, latent_dim)
        self.sampling = SamplingLayer()
        # init logvar to 0
        self.logvar.weight.data.fill_(0)
        self.logvar.bias.data.fill_(0)

    def forward(self, x):
        original_x = x
        x = self.conv1(x)
        x = self.bn1(x)  # Apply Batch Normalization
        x = torch.nn.functional.leaky_relu(x, 0.2)
        x = self.conv2(x)
        x = self.bn2(x)  # Apply Batch Normalization
        x = torch.nn.functional.leaky_relu(x, 0.2)
        x = self.conv3(x)
        x = self.bn3(x)  # Apply Batch Normalization
        x = torch.nn.functional.leaky_relu(x, 0.2)

        x = torch.nn.Flatten(start_dim=1)(x)
        # x: batch_size * 64*7*7
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        # x: batch_size * hidden_dim
        mu = self.mu(x)
        sigma = self.logvar(x)
        z = self.sampling(mu, sigma)
        # print the shape
        """print(f'original_x: {original_x.shape}')
        print(f'mu: {mu.shape}')
        print(f'sigma: {sigma.shape}')
        print(f'z: {z.shape}')"""

        # x: batch_size * latent_dim
        return z, mu, sigma, original_x

class VariationalDecoder(torch.nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(VariationalDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = torch.nn.Linear(latent_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 3 * 3 * 32)

        # Transpose Convolutional layers
        self.t_conv1 = torch.nn.ConvTranspose2d(32, 16, kernel_size=7, stride=2, padding=0)
        self.bn1 = torch.nn.BatchNorm2d(16)  # Batch Normalization after the first transposed convolution
        self.t_conv2 = torch.nn.ConvTranspose2d(16, 8, kernel_size=7, stride=2, padding=0)
        self.bn2 = torch.nn.BatchNorm2d(8)  # Batch Normalization after the second transposed convolution
        self.t_conv3 = torch.nn.ConvTranspose2d(8, 1, kernel_size=4, stride=1, padding=1)



    def forward(self, x):
        batch_size = x.size(0)
        x = self.fc1(x)
        x = torch.nn.functional.leaky_relu(x, 0.2)
        x = self.fc2(x)
        x = torch.nn.functional.leaky_relu(x, 0.2)
        x = torch.nn.Unflatten(1, (32, 3, 3))(x)
        x = self.t_conv1(x)
        x = self.bn1(x)  # Apply Batch Normalization
        x = torch.nn.functional.leaky_relu(x, 0.2)
        x = self.t_conv2(x)
        x = self.bn2(x)  # Apply Batch Normalization
        x = torch.nn.functional.leaky_relu(x, 0.2)
        x = self.t_conv3(x)
        x = torch.nn.functional.sigmoid(x)

        return x


def KL_loss(mu, sigma):
    return -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())

def reconstruction_loss(original_x, x):
    return torch.nn.functional.binary_cross_entropy(x, original_x, reduction='sum')

def loss_function(x, original_x, mu, sigma, k1=1, k2=1e-4):
    return k1 * reconstruction_loss(original_x, x) + k2 * KL_loss(mu, sigma)


# take inly 10 images
#mnist_trainset.data = mnist_trainset.data[:nb_image_to_overfit]
#mnist_trainset.targets = mnist_trainset.targets[:nb_image_to_overfit]

# shuffle the data and the targets in the same way
indices = torch.randperm(len(mnist_trainset.data))
mnist_trainset.data = mnist_trainset.data[indices]
mnist_trainset.targets = mnist_trainset.targets[indices]
batch_size = 128
mnist_trainloader = torch.utils.data.DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True)

# lets use optuna to find the best hyperparameters
import optuna
from optuna.trial import TrialState

def define_model(trial):
    # create the Variational Autoencoder
    hidden_dim = 256
    latent_size = trial.suggest_int('latent_size', 2, 200)

    model = VariationalAutoEncoder(input_dim=1,hidden_dim=hidden_dim, latent_dim=latent_size)
    return model



def objective(trial):
    # Generate the model.
    model = define_model(trial).to(device)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam"])

    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)
    k1 = trial.suggest_float('k1', 1e-5, 1)
    k2 = trial.suggest_float('k2', 1e-5, 1)
    # Training of the model.
    for epoch in range(10):
        epoch_loss = 0
        model.train()
        for images, _ in mnist_trainloader:
            images = images.to(device)
            x, mu, sigma, original_x = model(images)
            original_x = original_x.detach()
            loss = loss_function(x, original_x, mu, sigma,k1=k1,k2=k2)

            epoch_loss += loss.mean().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        trial.report(epoch_loss, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return epoch_loss/len(mnist_trainloader.dataset)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10, timeout=600)

pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
