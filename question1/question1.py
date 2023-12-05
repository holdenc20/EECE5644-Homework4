import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def write_data(samples, path):
    """Writes the sample data to the file path provided.

    Args:
        samples (pandas.DataFrame): the sample data
        path (string): the path to the file containing sample data
    """
    samples.to_csv(path, sep=',', header=True, index=True)


def read_sample_data(file_path):
    """Reads the sample data from the file path provided.

    Args:
        file_path (string): the path to the file containing sample data

    Returns:
        pandas.DataFrame: the sample data
    """
    data = pd.read_csv(file_path, sep=',', header=0, index_col=0)
    return data

def generate_data(dataset, N):
    data = pd.DataFrame(columns=['x1', 'x2', 'label'])
    for n in range(1, N + 1):
        r = np.random.choice([-1, 1])
        rad = dataset['rp1'].values[0]
        if r == -1:
            rad = dataset['rn1'].values[0]
        theta = np.random.uniform(-np.pi, np.pi)
        
        x1 = rad * np.cos(theta) + np.random.normal(0, dataset['sigma'].values[0])
        x2 = rad * np.sin(theta) + np.random.normal(0, dataset['sigma'].values[0])
        data = data._append({'x1': x1, 'x2': x2, 'label': r}, ignore_index=True)
          
    return data

def plot_pred_scatter(predictions, title):
    fig = plt.figure()
    correct = predictions[predictions['predictions'] == predictions['label']]
    incorrect = predictions[predictions['predictions'] != predictions['label']]
    plt.scatter(correct['x1'], correct['x2'], marker='o', c='green', s=1)
    
    plt.scatter(incorrect['x1'], incorrect['x2'], marker='o', c='red', s=1)
    plt.xlabel('X1')   
    plt.ylabel('X2')
    plt.title('2D Scatter Plot for ' + title + ' Samples')
    plt.legend()
    plt.savefig('./homework4/question1/' + title + '.png')

def plot_pred_scatter2(predictions, title):
    fig = plt.figure()
    correct = predictions[predictions['predictions'] == predictions['label']]
    correct1 = correct[correct['label'] == 1]
    correct2 = correct[correct['label'] == -1]
    
    plt.scatter(correct1['x1'], correct1['x2'], marker='o', c='green', s=10)
    plt.scatter(correct2['x1'], correct2['x2'], marker='o', c='blue', s=10)
    plt.xlabel('X1')   
    plt.ylabel('X2')
    plt.title('2D Scatter Plot for ' + title + ' Samples - removing incorrect predictions')
    plt.legend()
    plt.savefig('./homework4/question1/boundary-' + title + '.png')

def plot_scatter(data, title):
    plt.figure()
    plt.scatter(data['x1'], data['x2'], marker='o', c=data['label'])
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('2D Scatter Plot for ' + title + ' Samples')
    plt.legend()
    plt.savefig('./homework4/question1/' + title + '.png')


def mlp(numPerc, k, x, labels, numLabels):
    N = len(x)
    numValIters = 10
    y = np.zeros((numLabels, len(x)))

    for i in range(numLabels):
        y[i, :] = (labels == i + 1)

    partSize = N // k
    partInd = np.concatenate((np.arange(0, N, partSize), [len(x)]))

    avgPFE = np.zeros(numPerc)
    
    #M is the number of perceptrons in the hidden layer
    #this is for the model order selection
    for M in range(1, numPerc + 1):
        pFE = np.zeros(k)
        #k-fold cross validation
        #k is the number of folds
        for part in range(k):
            index_val = np.arange(partInd[part], partInd[part + 1])

            index_train = np.setdiff1d(np.arange(N), index_val)
            min_score = 1e6
            for i in range(3):
                net = MLPClassifier(hidden_layer_sizes=(M,), max_iter=10000, activation='relu')
                net.fit(x[index_train], labels[index_train])
                y_val = net.predict(x[index_val])
                score = 1 - accuracy_score(labels[index_val], y_val)
                min_score = min(min_score, score)
                
            pFE[part] = min_score
        #used for model order selection
        avgPFE[M - 1] = np.mean(pFE)

    #minimum cross entropy loss model    
    optM = np.argmin(avgPFE) + 1

    finalnet = MLPClassifier(hidden_layer_sizes=(optM,), max_iter=10000)
    finalnet.fit(x, labels)

    pFEFinal = np.zeros(numValIters)

    for i in range(numValIters):
        y_val = finalnet.predict(x)
        pFEFinal[i] = 1 - accuracy_score(labels, y_val)

    minPFE = np.min(pFEFinal)

    return finalnet, minPFE, optM, {'M': np.arange(1, numPerc + 1), 'mPFE': avgPFE}

if __name__ == '__main__':
    
    dataset = pd.DataFrame([[2, 4, 1]], 
                           columns=['rn1', 'rp1', 'sigma'])
    '''
    training = generate_data(dataset, 1000)
    write_data(training, './homework4/question1/training.csv')
    validation = generate_data(dataset, 10000)
    write_data(validation, './homework4/question1/validation.csv')
    
    plot_scatter(training, 'Training')
    plot_scatter(validation, 'Validation')
    '''
    
    training = read_sample_data('./homework4/question1/training.csv')
    validation = read_sample_data('./homework4/question1/validation.csv')
    
    #MLP
    numPerceptrons = 20
    k = 10
    valData = {}

    X_train = training[['x1', 'x2']].to_numpy()
    y_train = training['label'].to_numpy()
    net, minPFE, optM, stats = mlp(numPerceptrons, k, X_train, y_train, 2)
    X_val = validation[['x1', 'x2']].to_numpy()
    ypred = net.predict(X_val)
    validation['predictions'] = ypred
    pFE = np.sum(validation['predictions'] != validation['label']) / len(validation)
    M = stats['M']
    mPFE = stats['mPFE']
    print(f"NN pFE, N={len(training)}: Error={100 * pFE:.2f}%")
    
    #ploting the mean probability of error vs number of perceptrons
    fig, ax = plt.subplots()
    M = list(range(1, 21))
    ax.plot(M, mPFE)
    ax.set_xlabel('Number of Perceptrons')
    ax.set_ylabel('Mean Probability of Error')
    ax.set_title('Mean Probability of Error vs Number of Perceptrons')
    ax.legend()
    plt.savefig('./homework4/question1/perceptrons_vs_error.png')
    
    print("Optimal number of perceptrons: ", optM)
    
    #plotting the validation data
    plot_pred_scatter(validation, 'NN')
    plot_pred_scatter2(validation, 'NN')
    
    #SVM
    training = read_sample_data('./homework4/question1/training.csv')
    y_train = training['label'].to_numpy()
    training = training.drop(['label'], axis=1)
    validation = read_sample_data('./homework4/question1/validation.csv')
    y_val = validation['label'].to_numpy()
    validation = validation.drop(['label'], axis=1)
    
    
    C = np.logspace(-3, 3, 15)
    C = np.round(C, 3)
    gamma = np.logspace(-3, 3, 15)
    gamma = np.round(gamma, 3)
    param_grid = {'C': C,  
              'gamma': gamma, 
              'kernel': ['rbf']}
    
    svc = SVC()
    
    grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, cv=10)
    grid_search.fit(training, y_train)
    
    best = grid_search.best_params_
    print(best)
    results = grid_search.cv_results_
    scores = np.array(results['mean_test_score']).reshape(len(param_grid['C']), len(param_grid['gamma']))
    
    plt.figure(figsize=(8, 6))
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot)
    plt.xlabel('Gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(param_grid['gamma'])), param_grid['gamma'], rotation=45)
    plt.yticks(np.arange(len(param_grid['C'])), param_grid['C'])
    plt.title('SVM Hyperparameter Tuning (K-fold Cross-Validation)')
    plt.savefig('./homework4/question1/svm_hyperparameter_tuning.png')
    
    svc_final = SVC(kernel='rbf', C=best['C'], gamma=best['gamma'])
    svc_final.fit(training, y_train)
    
    ypred = svc_final.predict(validation)
    validation['predictions'] = ypred
    validation['label'] = y_val
    plot_pred_scatter(validation, 'SVM')
    plot_pred_scatter2(validation, 'SVM')
    print("Probability of error SVM: ", 1 - accuracy_score(y_val, ypred))