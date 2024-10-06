import torch


# checkpoint = torch.load("checkpoint/epoch_160.pth")

# #print(checkpoint['state_dict'].keys())

# print(checkpoint['state_dict']["net_g"].keys())


# print(3333)


checkpoint = torch.load("checkpoint/render.pth")

print(checkpoint.keys())

# print(checkpoint['state_dict']["net_g"].keys())