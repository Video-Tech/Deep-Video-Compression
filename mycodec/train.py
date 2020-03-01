import torch
from torch import nn
from torch.autograd import Variable

from util import init_lstm

def train_codec(train_loader, encoder, binarizer, decoder):
    num_epochs = 2
    learning_rate = 1e-3
    criterion = nn.MSELoss()
    nets = [encoder, decoder]
    params = [{'params': net.parameters()} for net in nets]
    optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=1e-5)
    
    for epoch in range(num_epochs):
        itr = 0
        for img, fn in train_loader:

            optimizer.zero_grad()

            (enc1, enc2, enc3, dec1, dec2, dec3, dec4) = init_lstm(3, 352, 640)#batch_size, height, width)
            
            #img = img.permute(0, 4, 2, 3, 1)
            img = Variable(img).cuda()
    
            # ===================forward=====================
            encoded_output, enc1, enc2, enc3 = encoder(img, enc1, enc2, enc3)
            
            code = binarizer(encoded_output)

            decoded_output, dec1, dec2, dec3, dec4 = decoder(code, dec1, dec2, dec3, dec4)

            #if itr % 500 == 0:
            #    save_model(encoder, decoder, epoch, itr)

            loss = criterion(decoded_output, img)
    
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            itr += 1
            print('Iteration:{}, loss:{:.4f}'.format(itr, loss.data))
