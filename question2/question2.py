import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score

def custom_scoring(gmm, X): 
    return -np.log(np.sum(gmm.score_samples(X)))

if __name__ == '__main__':
    image = plt.imread('./homework4/question2/image2.jpg')
    height, width, _ = image.shape

    # Preprocess the image
    feature_vectors = []
    for i in range(height):
        for j in range(width):
            pixel = image[i, j]
            row_index, col_index = i / height, j / width
            red, green, blue = pixel / 255.0
            feature_vector = [row_index, col_index, red, green, blue]
            feature_vectors.append(feature_vector)
    feature_vectors = np.array(feature_vectors)

    def split_into_k(a, n):
        k, m = divmod(len(a), n)
        return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

    data = feature_vectors[:, 2:]
    k = 10
    models = [GaussianMixture(n_components=n, max_iter=1000, tol=1e-3, covariance_type='full') for n in range(2, 9)]
    scores_total = []
    N = len(data)
    samples = list(split_into_k(np.arange(N), k))
    for model in models:
        scores = []
        for x in range(k):
            total = 0
            for i in range(1):
                train = np.array([data[i] for i in range(N) if i not in samples[x]])
                val = np.array([data[i] for i in range(N) if i in samples[x]])
                model.fit(train)
                total += model.score(val)
            scores.append(total)
        mean_score = np.mean(scores)
        scores_total.append(-1*mean_score)
        # print('Model has mean score of: ', mean_score)

    plt.plot(np.arange(2, 9), scores_total, label='negative log likelihood')
    plt.xlabel('Components')
    plt.ylabel('Negative Log Likelihood Score')
    plt.legend(loc='best')
    plt.title('Negative Log Likelihood Score for different GMM Models')
    #plt.show()
    
    #side by side comparison best model and original image and the overlay image
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(image)
    axs[0].set_title('Original Image')
    
    best_model = models[np.argmin(scores_total)]
    print('Best model has ' + str(np.argmin(scores_total) + 2) + ' components')
    best_model.fit(data)
    predictions = best_model.fit_predict(data)
    predictions = np.reshape(predictions, (height, width))
    colors = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
              (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
    temp = np.zeros((height, width, 3))
    for i in range(height):
        for j in range(width):
            temp[i, j] = colors[predictions[i, j]]
    axs[1].imshow(temp)
    
    
    plt.show()
    
