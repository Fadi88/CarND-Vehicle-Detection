from obj_detect_aux import treat_training_image
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import glob
import time



cars_list = glob.glob("../training_samples/vehicles/*/*")
not_cars_list = glob.glob("../training_samples/non-vehicles/*/*")

print("training set for cars is     : " , len(cars_list) , " samples")
print("training set for non cars is : " , len(not_cars_list), " samples")

car_features     = []
not_car_features = []

t0 = time.time()
for car_file in cars_list:
	car_features.append(treat_training_image(car_file))
print("time for extracting features for cars is    :" , time.time() - t0)

t0 = time.time()
for not_car_file in not_cars_list:
	not_car_features.append(treat_training_image(not_car_file))
print("time for extracting features for not cars is :" , time.time() - t0)

X = np.vstack((car_features, not_car_features)).astype(np.float64)
y = np.hstack((np.ones(len(car_features)), np.zeros(len(not_car_features))))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=np.random.randint(0, 100))

print()
svc = LinearSVC()

t0 = time.time()
svc.fit(X_train, y_train)

print("training model took : " , time.time() - t0)
print(svc.score(X_test , y_test))

pickle.dump(svc , open("clf.p","wb"))
