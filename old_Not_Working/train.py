from sklearn.datasets import load_iris
import bentoml
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
X = iris.data[:, :4]
y = iris.target


model = KNeighborsClassifier()
model.fit(X, y)

bento_model = bentoml.sklearn.save_model("knn",model)

print(bento_model)

