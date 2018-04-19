print(__doc__)
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy




x = pd.read_csv("train.csv")
# x=x.values
# X = np.zeros((len(x), 48*48))
# Y = np.zeros((len(x), 1))
# for i in range(len(x)):
#     X[i,:]=x[i,1].split(' ')
#     Y[i]=x[i,0]

# avg=np.mean(X, axis=0)
# st=np.std(X, axis=0)
# X=(X-avg)/st

# img_rows, img_cols = 48, 48
# batch_size = 1
# nb_classes = 7
# nb_epoch = 1

# Train_x=X[:25838,:]
# Val_x=X[25838:,:]

# Train_x = numpy.asarray(Train_x)
# Train_x = Train_x.reshape(Train_x.shape[0],img_rows,img_cols,1)

# Val_x = numpy.asarray(Val_x)
# Val_x = Val_x.reshape(Val_x.shape[0],img_rows,img_cols,1)

# Train_x = Train_x.astype('float32')
# Val_x = Val_x.astype('float32')

# Train_y=Y[:25838,:]


y=np.array(x['label'][25838:]).astype('float')

# y=Y[25838:,:]
pred_y=pd.read_csv("outputval.csv")
pred_y=np.array(pred_y['label'][25838:]).astype('float')
class_names = np.array(['angry','disgust','fear','happy','sad','suprised','neutral'])
# y=pd.read_csv("output.csv")
# print(y)
# print(pred_y)
# print(Train_y)
# Train_y = np_utils.to_categorical(Train_y, nb_classes)
# Val_y = np_utils.to_categorical(Val_y, nb_classes)






















# import some data to play with
'''
iris = datasets.load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names
print(type(class_names))
# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Run classifier, using a model that is too regularized (C too low) to see
# the impact on the results
classifier = svm.SVC(kernel='linear', C=0.01)
y_pred = classifier.fit(X_train, y_train).predict(X_test)
print(y_test)
print(type(y_pred))
'''
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y, pred_y)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
