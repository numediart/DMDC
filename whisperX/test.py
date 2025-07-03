import torch
print(torch.cuda.is_available())      # doit afficher True
print(torch.version.cuda)             # doit afficher '11.8'
print(torch.cuda.get_device_name(0))  # doit afficher 'NVIDIA GeForce GTX 1070'
