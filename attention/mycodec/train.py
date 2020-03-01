import torch
from torch import nn
from torch.autograd import Variable

from util import init_lstm

def train_codec(train_loader, encoder, binarizer, decoder):
    learning_rate = 0.00025
    criterion = nn.MSELoss()
    nets = [encoder, binarizer, decoder]
    params = [{'params': net.parameters()} for net in nets]
    optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=1e-5)
    
    itr = 0
    for img, fn in train_loader:
        batch_size, height, width = 2, 352, 640
        optimizer.zero_grad()
        (enc1, enc2, enc3, dec1, dec2, dec3, dec4) = init_lstm(batch_size, height, width)
        img = Variable(img).cuda()
    
        # ===================forward=====================
        encoded_output, enc1, enc2, enc3 = encoder(img, enc1, enc2, enc3)
        code = binarizer(encoded_output) # Need to use entropy coding to compress the bitstream
        decoded_output, dec1, dec2, dec3, dec4 = decoder(code, dec1, dec2, dec3, dec4)

        #if itr % 500 == 0:
        #    save_model(encoder, decoder, epoch, itr)

        loss = img - decoded_output
        loss = loss.abs().mean()
        #loss = criterion(decoded_output, img)
    
        # ===================backward====================
        loss.backward()
        for net in [encoder, binarizer, decoder]:
            if net is not None:
                torch.nn.utils.clip_grad_norm(net.parameters(), 0.5)
        optimizer.step()
        itr += 1
        print('Loss:{:.4f}'.format(loss.data))
