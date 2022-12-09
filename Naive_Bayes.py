import numpy as np

class Gaussian_Naive_Bayes():
    def fit(self, X, y):
        self.n_samples, self.n_features = X.shape
        self.n_classes = len(np.unique(y))
        self.classes = np.unique(y)
        
        self.mean = np.zeros((self.n_classes, self.n_features))
        self.variance = np.zeros((self.n_classes, self.n_features))
        self.priors = np.zeros(self.n_classes)
        
        for c in range(self.n_classes):
            X_c = X[y == self.classes[c]]
            self.mean[c, :] = np.mean(X_c, axis = 0)
            self.variance[c, :] = np.var(X_c, axis = 0)
            self.priors[c] = X_c.shape[0] / self.n_samples
    
    def gaussian_density(self, x, mean, var):
        const = 1 / np.sqrt(var * 2  * np.pi)
        prob = np.exp(-0.5 * ((x - mean) ** 2 / var))
        return const * prob

    def class_probability(self, x):
        posteriors = list()
        for c in range(self.n_classes):
            mean = self.mean[c]
            variance = self.variance[c]
            prior = np.log(self.priors[c])

            posterior = np.sum(np.log(self.gaussian_density(x, mean, variance)))
            posterior = prior + posterior
            posteriors.append(posterior)
        
        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        y_hat = [self.class_probability(x) for x in X]
        return np.array(y_hat)