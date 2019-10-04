from django.shortcuts import render
import base64
import numpy as np
from PIL import Image
import re
from keras.models import load_model
from resize_mnist import imageprepare
from django.core.files.storage import FileSystemStorage
import librosa
from keras import backend as K

# Create your views here.

def getHome(request):
    return render(request,'home.html')

def getDraw(request):
    return render(request,'canvas.html',{})

def image_solve(request):
    K.clear_session()
    image_data = request.POST['image_data']
    image_data = re.sub("^data:image/png;base64,", "", image_data)
    image_data = base64.b64decode(image_data)
    fh = open("imageToSave.png", "wb")
    fh.write(image_data)
    fh.close()
    print("save success!")
    im = Image.open("imageToSave.png")
    result = Image.new('RGB', (im.width, im.height), color=(255, 255, 255))
    result.paste(im, im)
    result.save('colors.jpg')
    x = [imageprepare('colors.jpg')]
    newArr = [[0 for d in range(28)] for y in range(28)]
    k = 0
    for i in range(28):
        for j in range(28):
            newArr[i][j] = x[0][k]
            k = k + 1

    im2arr = np.array(newArr)
    im2arr = im2arr.reshape(1, 28, 28, 1)
    model = load_model('my_model.h5')
    
    pred = model.predict(im2arr)
   
    d = {}
    for i in range(10):
        d[i] = pred[0][i]
        d[i] = round(d[i] * 100, 2)
    print(pred.argmax())
    kp = pred.argmax()
    K.clear_session()
    return render(request, 'display.html', {'ans': kp,'zero':d[0],'one':d[1],'two':d[2],'three':d[3],'four':d[4],'five':d[5],'six':d[6],'seven':d[7],'eight':d[8],'nine':d[9]})

def getVoice(request):
    if request.method == 'POST' and request.FILES.get('document', False):
        K.clear_session()
        uploaded_file=request.FILES['document']
        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name, uploaded_file)
        url=fs.url(filename)

        y, sr = librosa.load(url[1:], sr=8000, mono=True)

        mfcc = librosa.feature.mfcc(y, sr=8000, n_mfcc=40)
        if mfcc.shape[1] > 40:
            mfcc = mfcc[:, 0:40]
        else:
            mfcc = np.pad(mfcc, ((0, 0), (0, 40 - mfcc.shape[1])), mode='constant', constant_values=0)


        mfcc = np.expand_dims(mfcc, axis=0)
        mfcc = mfcc.reshape((1, 40, 40, 1))

        model = load_model('mnist_sound.h5')
        
        pred = model.predict(mfcc)
        
        kp = pred.argmax()
        d = {}
        for i in range(10):
            d[i] = pred[0][i]
            d[i] = round(d[i] * 100, 2)
        K.clear_session()

        return render(request,'predict.html',{'ans': kp,'zero':d[0],'one':d[1],'two':d[2],'three':d[3],'four':d[4],'five':d[5],'six':d[6],'seven':d[7],'eight':d[8],'nine':d[9]})

    return render(request, 'upload.html')
