import torch
import torch.nn as nn
import torch.nn.functional as F

# class abmil_(nn.Module):
#     def __init__(self):
#         super().__init__()
        
#         self.attention = nn.Sequential(
#             nn.Linear(512, 128),
#             nn.Tanh(),  ## Tanh / leaky relu
#             nn.Linear(128, 1))
        
#         self.classifier = nn.Sequential(
#             nn.Linear(512, 1),
#             nn.Sigmoid())

#     def forward(self, x):
    
#         A = self.attention(x)
#         A = torch.transpose(A, 2, 1)
#         A = F.softmax(A, dim=2)

#         M = torch.bmm(A, x)
        
#         prob = self.classifier(M)
#         pred = torch.ge(prob, 0.5).float()

#         return prob, pred
    
#     def loss_score(self, prob, y, class_weights = [1.0, 1.0]):

#         regularization = 0.0
#         for param in self.parameters():
#             regularization += torch.norm(param, 2)
#         y = y.float()
#         prob = torch.clamp(prob.squeeze(), 1e-5, 1.0 - 1e-5)

#         neg_log_like = -class_weights[1] * y * torch.log(prob) - class_weights[0]*(1 - y) * torch.log(1 - prob)
#         #neg_log_like += 0.008 * regularization 
#         return neg_log_like
    
#     def score(self, pred, y):
#         y = y.float()        
#         return torch.eq(pred, y).float()
    


class abmil(nn.Module):
    def __init__(self,input_size=512):
        super(abmil, self).__init__()
        self.L = input_size
        self.D = 128
        self.K = 1

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1)
        )

    def forward(self, x):
        # x = x.squeeze(0)

        A = self.attention(x)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, x)  # KxL

        Y_prob = self.classifier(M)

        return Y_prob, A


    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A



class mcabmil(nn.Module):
    def __init__(self,input_size=512, output_size=1):
        super(mcabmil, self).__init__()
        self.L = input_size
        self.D = 128
        self.K = 1
        self.output_size=output_size

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, self.output_size)
        )

    def forward(self, x):
        # x = x.squeeze(0)

        A = self.attention(x)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, x)  # KxL

        Y_prob = self.classifier(M)
        
        return Y_prob, A



class gabmil(nn.Module):
    def __init__(self,input_size=512):
        super(gabmil, self).__init__()
        self.L = input_size
        self.D = 128
        self.K = 1

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh() 
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)

        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A