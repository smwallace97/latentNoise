import math
import numpy as np
import torch
import gpytorch
from matplotlib import pyplot as plt
from gpytorch.mlls import SumMarginalLogLikelihood


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([30]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(batch_shape=torch.Size([30])),
            batch_shape=torch.Size([30])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def model_setup(train_x, train_y_a, train_y_b, train_y_c):
    likelihood_a = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([30]))
    model_a = ExactGPModel(train_x, train_y_a, likelihood_a)
    likelihood_b = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([30]))
    model_b = ExactGPModel(train_x, train_y_b, likelihood_b)
    likelihood_c = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([30]))
    model_c = ExactGPModel(train_x, train_y_c, likelihood_c)

    model = gpytorch.models.IndependentModelList(model_a, model_b, model_c)
    likelihood = gpytorch.likelihoods.LikelihoodList(model_a.likelihood, model_b.likelihood, model_c.likelihood)
    mll = SumMarginalLogLikelihood(likelihood, model)
    model, likelihood = train(model, likelihood, mll)
    return model, likelihood

def train(model, likelihood, mll):
    training_iterations = 150

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model(*model.train_inputs)
        loss = -mll(output, model.train_targets)
        loss.sum().backward()
        # print the loss
        if i % 25 == 0 or i==training_iterations-1:
            print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.sum().item()))
        optimizer.step()

    return model, likelihood

def evaluate(model, likelihood, train_x):
    model.eval()
    likelihood.eval()

    f, axs = plt.subplots(1, 3, figsize=(12, 3))

    # make predictions using the same test points
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictions = likelihood(*model(train_x, train_x, train_x))

    for submodel, prediction, ax in zip(model.models, predictions, axs):
        mean = prediction.mean
        lower, upper = prediction.confidence_region()
        tr_y = submodel.train_targets.mean(0).detach().numpy()

        # plot training data as black stars
        ax.plot(train_x, tr_y, 'k*')
        # predictive mean as blue line
        ax.plot(train_x.numpy(), mean.mean(0).numpy(), 'b')
        # shade in confidence
        ax.fill_between(train_x.numpy(), lower.mean(0).detach().numpy(), upper.mean(0).detach().numpy(), alpha=0.5)
        ax.set_ylim([-2, 4])
        ax.legend(['Observed Data (mean)', 'Mean', 'Confidence'])
        ax.set_title('Observed Values (Likelihood)')
    plt.show()


def main():
    # training data
    train_x = torch.linspace(0, 1, 21)
    train_y_a = torch.from_numpy(np.load('30samps_seed1_a.npy')).float()
    train_y_b = torch.from_numpy(np.load('30samps_seed1_b.npy')).float()
    train_y_c = torch.from_numpy(np.load('30samps_seed1_c.npy')).float()
    model, likelihood = model_setup(train_x, train_y_a, train_y_b, train_y_c)
    evaluate(model, likelihood, train_x)



if __name__ == '__main__':
    main()