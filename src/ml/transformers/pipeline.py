

class TransformPipeline:
    def __init__(self, transforms):
        self.transforms = transforms

    def fit(self, X, y=None):
        for t in self.transforms:
            t.fit(X, y)
        return self

    def transform(self, X):
        for t in self.transforms:
            X = t.transform(X)
        return X

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
