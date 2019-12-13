import torch
from torch import nn
from torch.autograd import Variable

def train_codec(train_loader, encoder, decoder):
    num_epochs = 2
    learning_rate = 1e-3
    criterion = nn.MSELoss()
    nets = [encoder, decoder]
    params = [{'params': net.parameters()} for net in nets]
    optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=1e-5)
    
    for epoch in range(num_epochs):
        itr = 0
        for img, fn in train_loader:
            img = img.permute(0, 4, 2, 3, 1)
            img = Variable(img).cuda()
    
            # ===================forward=====================
            encoded_output = encoder(img)
            decoded_output = decoder(encoded_output)

            if itr % 500 == 0:
                save_model(encoder, decoder, epoch, itr)

            loss = criterion(decoded_output, img)
    
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            itr += 1
            print('Iteration:{}, loss:{:.4f}'.format(itr, loss.data))
