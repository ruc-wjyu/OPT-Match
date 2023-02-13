# import torch
# import torch.nn as nn
# import math
#
#
# class ConineSimilarity(nn.Module):
#     def forward(self, tensor_1, tensor_2):
#         normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=-1, keepdim=True)
#         normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=-1, keepdim=True)
#         return (normalized_tensor_1 * normalized_tensor_2).sum(dim=-1)
#
#
# # input_1 = torch.randn(3, 5, requires_grad=True)
# # print(input_1)
# # input_2 = torch.randn(3, 5, requires_grad=True)
# # print(input_2)
# # con = ConineSimilarity()
# # CS = con(input_1, input_2)
# # CS2 = torch.cosine_similarity(input_1, input_2)
# # print(CS)
# # print(CS2)
#
#
