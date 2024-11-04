import torch
#加载权至
checkpoint = torch.load('./checkpoint/epoch_11000.pth')
#提舰netg
net_g_static = checkpoint['state_dict']['net_g']


#保存新权童
torch.save(net_g_static,'./checkpoint/new.pth')

print("ok")