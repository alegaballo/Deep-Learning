import numpy as np
import gzip
import pickle
import matplotlib.pyplot as plt


def load_data():
    np.random.seed(1990)
    print("Loading MNIST data .....")

    # Load the MNIST dataset
    with gzip.open('Data/mnist.pkl.gz', 'r') as f:
        train_set, valid_set, test_set = pickle.load(f,encoding="iso-8859-1")
        learn_data       = [(train_set[0][i], [1 if j == train_set[1][i] else 0 for j in range(10)]) \
                    for i in np.arange(len(train_set[0]))]
        test_data        = [(test_set[0][i], [1 if j == test_set[1][i] else 0 for j in range(10)]) \
                    for i in np.arange(len(test_set[0]))]
        validation_data  = [(valid_set[0][i], [1 if j == valid_set[1][i] else 0 for j in range(10)]) \
                    for i in np.arange(len(valid_set[0]))]
        
        
    print("Done.")
    return learn_data , test_data, validation_data 




   
        
        
def plot_curve(t,s,metric):
    plt.plot(t, s)
    plt.ylabel(metric) # or ERROR
    plt.xlabel('Epoch')
    plt.title('Learning Curve_'+str(metric))
    #curve_name=str(metric)+"LC.png"
    #plt.savefig(Figures/curve_name)
    plt.show()
    

