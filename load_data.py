import numpy as np
from keras_preprocessing import image
from PIL import Image
from numpy import hstack
from scipy import misc
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io as scio
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import normalize

path = './data'

def HW():
    from sklearn import preprocessing
    min_max_scaler = preprocessing.MinMaxScaler()
    data = scio.loadmat(path + "/HW.mat")
    x1 = data['x1']
    x2 = data['x2']
    Y = data['gt'][:,0]
    x1 = min_max_scaler.fit_transform(x1)
    x2 = min_max_scaler.fit_transform(x2)
    return [x1, x2], Y


def Get_MNIST_USPS_From_COMIC():
    data = 0
    if data == 1:
        x = scio.loadmat(path + "/MNIST-USPS.mat")
        print(x)
        x1 = x['X1']
        x2 = x['X2']
        Y = x['Y']
        print(x1.shape)
        print(x2.shape)
        print(Y.shape)
        print(x1[0])
        print(x2[0])
        print(Y[0])
        x1 = x1.reshape((5000, 28, 28))
        x2 = x2.reshape((5000, 16, 16), order='A')
        print(Y)
        Y = Y[0].reshape(5000,)
        print(Y)
        xu_reshape = np.zeros([len(x2), 28, 28], dtype=float)
        for i in range(len(x2)):
            for x in range(16):
                for y in range(16):
                    xu_reshape[i][x + 6][y + 6] = x2[i][x][y]

        print(x1.shape)
        print(xu_reshape.shape)
        print(Y.shape)
        z = np.linspace(0, len(Y) - 1, len(Y), dtype=int)
        np.random.shuffle(z)
        # print(z)
        # print(y_label)
        x_data_m = x1
        x_data_u = xu_reshape
        y_label = Y
        x_shuffle_m = np.copy(x_data_m)
        x_shuffle_u = np.copy(x_data_u)
        y_shuffle = np.copy(y_label)
        for i in range(len(y_label)):
            x_shuffle_m[i] = x_data_m[z[i]]
            x_shuffle_u[i] = x_data_u[z[i]]
            y_shuffle[i] = y_label[z[i]]
        x_shuffle_m = x_shuffle_m.reshape([-1, 28, 28, 1])
        x_shuffle_u = x_shuffle_u.reshape([-1, 28, 28, 1])/255
        print(x_shuffle_m.shape)
        print(x_shuffle_u.shape)
        print(y_shuffle.shape)
        print(x_shuffle_m[0])
        print(x_shuffle_u[0])
        # print(y_shuffle[0])
        scio.savemat(path + '/2V_MNIST_USPS.mat', {'X1': x_shuffle_m, 'X2': x_shuffle_u, 'Y': y_shuffle})
    data = scio.loadmat(path + "/2V_MNIST_USPS.mat")
    x1 = data['X1']
    x2 = data['X2']
    Y = data['Y'][0]
    ge = np.random.randint(0, len(x1), 1, dtype=int)
    image1 = np.reshape(x1[ge], (28, 28))
    image2 = np.reshape(x2[ge], (28, 28))
    print(Y[ge][0])
    plt.figure('Mnist')
    plt.imshow(image1)
    plt.show()
    plt.figure('USPS')
    plt.imshow(image2)
    plt.show()
    print(x1.shape)
    print(x2.shape)
    print(Y.shape)

    return [x1, x2], Y

def DigitProduct():
    import scipy.io as scio
    from sklearn import preprocessing
    min_max_scaler = preprocessing.MinMaxScaler()
    data = scio.loadmat(path + "/Digit-Product.mat")
    x1 = data['X1']
    x2 = data['X2']
    x1 = min_max_scaler.fit_transform(x1)
    x2 = min_max_scaler.fit_transform(x2)
    Y = data['Y'][0]
    return [x1, x2], Y

def ALOI():
    import scipy.io as scio
    from sklearn import preprocessing
    from sklearn.preprocessing import normalize
    min_max_scaler = preprocessing.MinMaxScaler()
    data = scio.loadmat(path + "/ALOI.mat")
    x1 = data['X1']
    x2 = data['X2']
    x3 = data['X3']
    x4 = data['X4']
    x1 = min_max_scaler.fit_transform(x1)
    x2 = min_max_scaler.fit_transform(x2)
    x3 = min_max_scaler.fit_transform(x3)
    x4 = min_max_scaler.fit_transform(x4)
    Y = data['Y'][0]
    return [x1, x2, x3, x4], Y

def AWA():
    import scipy.io as scio
    from sklearn import preprocessing
    min_max_scaler = preprocessing.MinMaxScaler()
    data = scio.loadmat(path + "/AWA.mat")
    x1 = data['X1']
    x2 = data['X2']
    x3 = data['X3']
    x4 = data['X4']
    x5 = data['X5']
    x6 = data['X6']
    x7 = data['X7']
    # x1 = min_max_scaler.fit_transform(x1)
    # x2 = min_max_scaler.fit_transform(x2)
    # x3 = min_max_scaler.fit_transform(x3)
    # x4 = min_max_scaler.fit_transform(x4)
    # x5 = min_max_scaler.fit_transform(x5)
    # x6 = min_max_scaler.fit_transform(x6)
    # x7 = min_max_scaler.fit_transform(x7)
    x1 = normalize.fit_transform(x1)
    x2 = normalize(x2)
    x3 = normalize(x3)
    x4 = normalize(x4)
    x5 = normalize(x5)
    x6 = normalize(x6)
    x7 = normalize(x7)
    Y = data['Y'][0]
    return [x1, x2, x3, x4, x5, x6, x7], Y


def NoisyMNIST():
    import scipy.io as scio
    from sklearn import preprocessing
    min_max_scaler = preprocessing.MinMaxScaler()
    data = scio.loadmat(path + "/NoisyMNIST.mat")
    x1 = data['X1']
    x2 = data['X2']
    x1 = min_max_scaler.fit_transform(x1)
    x2 = min_max_scaler.fit_transform(x2)
    Y = data['Y'][0]
    return [x1, x2], Y

def Scene_15():
    import scipy.io as scio
    from sklearn import preprocessing
    min_max_scaler = preprocessing.MinMaxScaler()
    data = scio.loadmat(path + "/Scene-15.mat")
    x1 = data['X1']
    x2 = data['X2']
    x3 = data['X3']
    x1 = min_max_scaler.fit_transform(x1)
    x2 = min_max_scaler.fit_transform(x2)
    x3 = min_max_scaler.fit_transform(x3)
    Y = data['Y'][0]
    return [x1, x2, x3], Y

def Hdigit():
    import scipy.io as scio
    from sklearn import preprocessing
    min_max_scaler = preprocessing.MinMaxScaler()
    data = scio.loadmat(path + "/Hdigit.mat")
    x1 = data['X1']
    x2 = data['X2']
    x1 = min_max_scaler.fit_transform(x1)
    x2 = min_max_scaler.fit_transform(x2)
    Y = data['Y'][0]
    return [x1, x2], Y

def Caltech101_4V():
    import scipy.io as scio
    from sklearn import preprocessing
    min_max_scaler = preprocessing.MinMaxScaler()
    data = scio.loadmat(path + "/Caltech-4V.mat")
    x1 = data['X1']
    x2 = data['X2']
    x3 = data['X3']
    x4 = data['X4']
    x1 = min_max_scaler.fit_transform(x1)
    x2 = min_max_scaler.fit_transform(x2)
    x3 = min_max_scaler.fit_transform(x3)
    x4 = min_max_scaler.fit_transform(x4)
    Y = data['Y'][0]
    return [x1, x2, x3, x4], Y

def Caltech101_5V():
    import scipy.io as scio
    from sklearn import preprocessing
    min_max_scaler = preprocessing.MinMaxScaler()
    data = scio.loadmat(path + "/Caltech-5V.mat")
    x1 = data['X1']
    x2 = data['X2']
    x3 = data['X3']
    x4 = data['X4']
    x5 = data['X5']
    x1 = min_max_scaler.fit_transform(x1)
    x2 = min_max_scaler.fit_transform(x2)
    x3 = min_max_scaler.fit_transform(x3)
    x4 = min_max_scaler.fit_transform(x4)
    x5 = min_max_scaler.fit_transform(x5)
    Y = data['Y'][0]
    return [x1, x2, x3, x4, x5], Y

def MSRC():
    import scipy.io as scio
    from sklearn import preprocessing
    min_max_scaler = preprocessing.MinMaxScaler()
    data = scio.loadmat(path + "/MSRC.mat")
    x1 = data['X1']
    x2 = data['X2']
    x3 = data['X3']
    x4 = data['X4']
    x5 = data['X5']
    x1 = min_max_scaler.fit_transform(x1)
    x2 = min_max_scaler.fit_transform(x2)
    x3 = min_max_scaler.fit_transform(x3)
    x4 = min_max_scaler.fit_transform(x4)
    x5 = min_max_scaler.fit_transform(x5)
    Y = data['Y'][0]
    return [x1, x2, x3, x4, x5], Y

def Caltech101_all():
    import scipy.io as scio
    from sklearn import preprocessing
    min_max_scaler = preprocessing.MinMaxScaler()
    data = scio.loadmat(path + "/Caltech-all.mat")
    x1 = data['X1']
    x2 = data['X2']
    x3 = data['X3']
    x4 = data['X4']
    x5 = data['X5']
    x6 = data['X6']
    x1 = min_max_scaler.fit_transform(x1)
    x2 = min_max_scaler.fit_transform(x2)
    x3 = min_max_scaler.fit_transform(x3)
    x4 = min_max_scaler.fit_transform(x4)
    x5 = min_max_scaler.fit_transform(x5)
    x6 = min_max_scaler.fit_transform(x6)
    Y = data['Y'][0]
    return [x1, x2, x3, x4, x5, x6], Y

def YTF10():
    import scipy.io as scio
    from sklearn import preprocessing
    from sklearn.preprocessing import normalize
    min_max_scaler = preprocessing.MinMaxScaler()
    data = scio.loadmat(path + "/YTF10.mat")
    x1 = data['X1']
    x2 = data['X2']
    x3 = data['X3']
    x4 = data['X4']
    x1 = min_max_scaler.fit_transform(x1)
    x2 = min_max_scaler.fit_transform(x2)
    x3 = min_max_scaler.fit_transform(x3)
    x4 = min_max_scaler.fit_transform(x4)
    # x1 = normalize(x1)
    # x2 = normalize(x2)
    # x3 = normalize(x3)
    # x4 = normalize(x4)
    Y = data['Y'][0]
    return [x1, x2, x3, x4], Y

def NUS():
    import scipy.io as scio
    from sklearn import preprocessing
    min_max_scaler = preprocessing.MinMaxScaler()
    data = scio.loadmat(path + "/NUS.mat")
    x1 = data['X1']
    x2 = data['X2']
    x3 = data['X3']
    x4 = data['X4']
    x5 = data['X5']
    x1 = min_max_scaler.fit_transform(x1)
    x2 = min_max_scaler.fit_transform(x2)
    x3 = min_max_scaler.fit_transform(x3)
    x4 = min_max_scaler.fit_transform(x4)
    x5 = min_max_scaler.fit_transform(x5)
    Y = data['Y'][0]
    return [x1, x2, x3, x4, x5], Y

def Caltech101_20():
    data = 0
    if data == 1:
        import scipy.io as scio
        data = scio.loadmat(path + "/Caltech101-20.mat")
        Y = data['Y'] - 1
        # print(Y.shape)
        X = data['X']
        print(X[0][0].shape)
        print(X[0][1].shape)
        print(X[0][2].shape)
        print(X[0][3].shape)
        print(X[0][4].shape)
        print(X[0][5].shape)
        x1 = X[0][0]
        x2 = X[0][1]
        x3 = X[0][2]
        x4 = X[0][3]
        x5 = X[0][4]
        x6 = X[0][5]
        t = np.linspace(0, Y.shape[0] - 1, Y.shape[0], dtype=int)
        print(t)
        import random
        random.shuffle(t)
        # np.save("./Caltech101_20_t.npy", t)
        t = np.load("./Caltech101_20_t.npy")
        print(t)
        xx1 = np.copy(x1)
        xx2 = np.copy(x2)
        xx3 = np.copy(x3)
        xx4 = np.copy(x4)
        xx5 = np.copy(x5)
        xx6 = np.copy(x6)
        YY = np.copy(Y)
        for i in range(Y.shape[0]):
            x1[i] = xx1[t[i]]
            x2[i] = xx2[t[i]]
            x3[i] = xx3[t[i]]
            x4[i] = xx4[t[i]]
            x5[i] = xx5[t[i]]
            x6[i] = xx6[t[i]]
            Y[i] = YY[t[i]]
        from sklearn import preprocessing
        min_max_scaler = preprocessing.MinMaxScaler()
        x1 = min_max_scaler.fit_transform(x1)
        x2 = min_max_scaler.fit_transform(x2)
        x3 = min_max_scaler.fit_transform(x3)
        x4 = min_max_scaler.fit_transform(x4)
        x5 = min_max_scaler.fit_transform(x5)
        x6 = min_max_scaler.fit_transform(x6)
        print(x1[0])
        Y = Y.reshape(Y.shape[0])
        print(Y.shape)
        scio.savemat(path + '/6V_Caltech101_20.mat', {'X1': x1, 'X2': x2, 'X3': x3, 'X4': x4, 'X5': x5, 'X6': x6, 'Y': Y})
    import scipy.io as scio
    data = scio.loadmat(path + "/6V_Caltech101_20.mat")
    x1 = data['X1']
    x2 = data['X2']
    x3 = data['X3']
    x4 = data['X4']
    x5 = data['X5']
    x6 = data['X6']
    Y = data['Y'][0]
    print(x1.shape)
    print(x2.shape)
    print(x3.shape)
    print(x4.shape)
    print(x5.shape)
    print(x6.shape)
    print(Y.shape)

    return [x1, x2, x3, x4, x5, x6], Y


def BDGP():
    from sklearn import preprocessing
    min_max_scaler = preprocessing.MinMaxScaler()
    data = scio.loadmat(path + "/BDGP.mat")
    x1 = data['X1']
    x2 = data['X2']
    # x1 = min_max_scaler.fit_transform(x1)
    # x2 = min_max_scaler.fit_transform(x2)
    Y = data['Y'][0]
    print(x1.shape)
    print(x2.shape)
    print(Y.shape)
    return [x1, x2], Y

def Wiki_fea():
    import scipy.io as scio
    from sklearn import preprocessing
    min_max_scaler = preprocessing.MinMaxScaler()
    data = scio.loadmat(path + "/Wiki_fea.mat")
    x1 = data['X1']
    x2 = data['X2']
    x1 = min_max_scaler.fit_transform(x1)
    x2 = min_max_scaler.fit_transform(x2)
    Y = data['Y'][0]
    return [x1, x2], Y

def CCV():
    import scipy.io as scio
    from sklearn import preprocessing
    from sklearn.preprocessing import normalize
    min_max_scaler = preprocessing.MinMaxScaler()
    data = scio.loadmat(path + "/CCV.mat")
    x1 = data['X1']
    x2 = data['X2']
    x3 = data['X3']
    # x1 = min_max_scaler.fit_transform(x1)
    # x2 = min_max_scaler.fit_transform(x2)
    # x3 = min_max_scaler.fit_transform(x3)
    x1 = normalize(x1, norm='l2')  
    x2 = normalize(x2, norm='l2')  
    x3 = normalize(x3, norm='l2') 
    Y = data['Y'][0]
    return [x1, x2, x3], Y

def AWA():
    import scipy.io as scio
    from sklearn import preprocessing
    from sklearn.preprocessing import normalize
    min_max_scaler = preprocessing.MinMaxScaler()
    data = scio.loadmat(path + "/AWA.mat")
    x1 = data['X1']
    x2 = data['X2']
    x3 = data['X3']
    x4 = data['X4']
    x5 = data['X5']
    x6 = data['X6']
    x7 = data['X7']
    x1 = min_max_scaler.fit_transform(x1)
    x2 = min_max_scaler.fit_transform(x2)
    x3 = min_max_scaler.fit_transform(x3)
    x4 = min_max_scaler.fit_transform(x4)
    x5 = min_max_scaler.fit_transform(x5)
    x6 = min_max_scaler.fit_transform(x6)
    x7 = min_max_scaler.fit_transform(x7)
    Y = data['Y'][0]
    return [x1, x2, x3, x4, x5, x6, x7], Y


def load_data(dataset):
    print("load:", dataset)
    if dataset == 'CCV':               # HW
        return CCV()
    elif dataset == 'ALOI':
        return ALOI()
    elif dataset == 'Fashion_MV':                   # Fashion-10K-3views
        return Fashion_MV()
    elif dataset == 'MNIST_USPS':                 # MNIST-USPS
        return Get_MNIST_USPS_From_COMIC()
    elif dataset == 'NoisyMNIST':
        return NoisyMNIST()
    elif dataset == 'YTF10':
        return YTF10()
    elif dataset == 'NUS':
        return NUS()
    elif dataset == 'Caltech-5V':              # Caltech101_20
        return Caltech101_5V()
    elif dataset == 'Caltech-4V':
        return Caltech101_4V()
    elif dataset == 'Caltech-all':
        return Caltech101_all()
    elif dataset == 'BDGP':                       # BDGP
        return BDGP()
    elif dataset == 'Digit-Product':
        return DigitProduct()
    elif dataset == 'Hdigit':
        return Hdigit()
    elif dataset == 'Scene-15':
        return Scene_15()
    elif dataset == 'Wiki_fea':
        return Wiki_fea()
    elif dataset =='MSRC':
        return MSRC()
    elif dataset == 'AWA':
        return AWA()
    else:
        raise ValueError('Not defined for loading %s' % dataset)
