import os
import numpy as np
from torch.autograd import Variable
from torchvision.utils import save_image
from scipy.misc import imread, imresize, imsave

if not os.path.exists('./output'):
    os.mkdir('./output')

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 3, 352, 640)
    return x

def save_numpy_array_as_image(filename, arr):
    imsave(filename, np.squeeze(arr * 255.0).astype(np.uint8).transpose(1, 2, 0))

def save_output(output, filenames):
    for b1, b2 in zip(output, filenames):
        for img, filename in zip(b1, b2):
            fn = filename[-12:]
            fn = 'output/frame_'+fn
            save_numpy_array_as_image(fn, img)

def test_codec(test_loader, encoder, decoder):
    for img, fn in test_loader:
        img = img.permute(0, 4, 2, 3, 1)
        img = Variable(img).cuda()
        code = encoder(img)
        output = decoder(code)
        out_img = output.data.cpu()
        out_img_np = out_img.numpy().clip(0, 1)
        fn = np.array(fn)
        #out_img_np = np.swapaxes(out_img_np, 0, 1)
        fn = np.swapaxes(fn, 0, 1)
        save_output(out_img_np, fn)
        #pic = to_img(output.cpu().data)
        #save_image(pic[0], './output/image_temp.png')
