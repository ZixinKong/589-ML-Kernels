# Import python modules
import numpy as np
import kaggle
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import poly
import tri
import model_selection

############################################################################
# Read in train and test synthetic data
def read_synthetic_data():
	print('Reading synthetic data ...')
	train_x = np.loadtxt('../../Data/Synthetic/data_train.txt', delimiter = ',', dtype=float)
	train_y = np.loadtxt('../../Data/Synthetic/label_train.txt', delimiter = ',', dtype=float)
	test_x = np.loadtxt('../../Data/Synthetic/data_test.txt', delimiter = ',', dtype=float)
	test_y = np.loadtxt('../../Data/Synthetic/label_test.txt', delimiter = ',', dtype=float)

	return (train_x, train_y, test_x, test_y)

############################################################################
# Read in train and test credit card data
def read_creditcard_data():
	print('Reading credit card data ...')
	train_x = np.loadtxt('../../Data/CreditCard/data_train.txt', delimiter = ',', dtype=float)
	train_y = np.loadtxt('../../Data/CreditCard/label_train.txt', delimiter = ',', dtype=float)
	test_x = np.loadtxt('../../Data/CreditCard/data_test.txt', delimiter = ',', dtype=float)

	return (train_x, train_y, test_x)

############################################################################
# Read in train and test tumor data
def read_tumor_data():
	print('Reading tumor data ...')
	train_x = np.loadtxt('../../Data/Tumor/data_train.txt', delimiter = ',', dtype=float)
	train_y = np.loadtxt('../../Data/Tumor/label_train.txt', delimiter = ',', dtype=float)
	test_x = np.loadtxt('../../Data/Tumor/data_test.txt', delimiter = ',', dtype=float)

	return (train_x, train_y, test_x)

############################################################################
# Compute MSE
def compute_MSE(y, y_hat):
	# mean squared error
	return np.mean(np.power(y - y_hat, 2))

############################################################################


######################## 1.d

train_x, train_y, test_x, test_y = read_synthetic_data()
print('Train=', train_x.shape)
print('Test=', test_x.shape)

poly_i = [1, 2, 4, 6]
# KRRS
predictions_poly_krrs = []
MSE_poly_krrs = []
for i in poly_i:
    y_hat = poly.krrs(train_x, train_y, test_x, i)
    predictions_poly_krrs.append(y_hat)
    MSE_poly_krrs.append(compute_MSE(test_y, y_hat))
    
errors_poly_krrs = dict(zip(poly_i, MSE_poly_krrs))
print("MSE of polynomial KRRS", errors_poly_krrs)

# BERR
predictions_poly_berr = []
MSE_poly_berr = []
for i in poly_i:
    y_hat = poly.berr(train_x, train_y, test_x, i)
    predictions_poly_berr.append(y_hat)
    MSE_poly_berr.append(compute_MSE(test_y, y_hat))
    
errors_poly_berr = dict(zip(poly_i, MSE_poly_berr))
print("MSE of trigonometric BERR", errors_poly_berr)


tri_i = [3, 5, 10]
# KRRS
predictions_tri_krrs = []
MSE_tri_krrs = []
for i in tri_i:
    y_hat = tri.berr(train_x, train_y, test_x, i)
    predictions_tri_krrs.append(y_hat)
    MSE_tri_krrs.append(compute_MSE(test_y, y_hat))
    
errors_tri_krrs = dict(zip(tri_i, MSE_tri_krrs))
print("MSE of polynomial KRRS", errors_tri_krrs)

# BERR
predictions_tri_berr = []
MSE_tri_berr = []
for i in tri_i:
    y_hat = tri.berr(train_x, train_y, test_x, i)
    predictions_tri_berr.append(y_hat)
    MSE_tri_berr.append(compute_MSE(test_y, y_hat))
    
errors_tri_berr = dict(zip(tri_i, MSE_tri_berr))
print("MSE of trigonometric BERR", errors_tri_berr)

#Plot a line graph
fig, axes = plt.subplots(4, 2, figsize=(15, 20))

axes[0, 0].plot(test_x, predictions_poly_krrs[1], 'or') 
axes[0, 0].set_title("KRRS, Polynomial, degree=2, lamda=0.1")

axes[0, 1].plot(test_x, predictions_poly_berr[1], 'or') 
axes[0, 1].set_title("BERR, Polynomial, degree=2, lamda=0.1")

axes[1, 0].plot(test_x, predictions_poly_krrs[3], 'or') 
axes[1, 0].set_title("KRRS, Polynomial, degree=6, lamda=0.1")

axes[1, 1].plot(test_x, predictions_poly_berr[3], 'or')
axes[1, 1].set_title("BERR, Polynomial, degree=6, lamda=0.1")

axes[2, 0].plot(test_x, predictions_tri_krrs[1], 'or') 
axes[2, 0].set_title("KRRS, Trigonometric, degree=5, lamda=0.1")

axes[2, 1].plot(test_x, predictions_tri_berr[1], 'or')
axes[2, 1].set_title("BERR, Trigonometric, degree=5, lamda=0.1")

axes[3, 0].plot(test_x, predictions_tri_krrs[2], 'or') 
axes[3, 0].set_title("KRRS, Trigonometric, degree=10, lamda=0.1")

axes[3, 1].plot(test_x, predictions_tri_berr[2], 'or')
axes[3, 1].set_title("BERR, Trigonometric, degree=10, lamda=0.1")

for plot in axes:
    for ax in plot:
        ax.plot(test_x, test_y, '*b')
        ax.set_xlabel("Test X")
        ax.set_ylabel("True/Predicted Y")


#Save the chart
plt.savefig("../Figures/1d_line_plot.pdf")
plt.show()

######################## 1.e

train_x, train_y, test_x  = read_creditcard_data()
print('Train=', train_x.shape)
print('Train=', train_y.shape)
print('Test=', test_x.shape)


#model_selection.credit_card(train_x, train_y)
result_cc = model_selection.credit_card(train_x, train_y)
print(result_cc, "\n")
print("Best parameter is: ", min(result_cc, key = result_cc.get) )


clf = KernelRidge(alpha=0.0001, kernel='rbf', gamma=None)
clf.fit(train_x, train_y)
predicted_y = clf.predict(test_x)

# Output file location
file_name = '../Predictions/CreditCard/best.csv'
# Writing output in Kaggle format
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name, True)


######################### 2.a

train_x, train_y, test_x  = read_tumor_data()
print('Train=', train_x.shape)
print('Test=', test_x.shape)

result_t = model_selection.tumor(train_x, train_y)
print(result_t, "\n")
print("Best parameter is: ", max(result_t, key = result_t.get))

clf = SVC(C=1.0, kernel='rbf', gamma=0.001)
clf.fit(train_x, train_y)
predicted_y = clf.predict(test_x)

# Output file location
file_name = '../Predictions/Tumor/best.csv'
# Writing output in Kaggle format
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name, False)

########################## 3

train_x, train_y, test_x  = read_creditcard_data()
print('Train=', train_x.shape)
print('Train=', train_y.shape)
print('Test=', test_x.shape)

clf = KernelRidge(alpha=0.0001, kernel='rbf', gamma=None)
clf.fit(train_x, train_y)
predicted_y = clf.predict(test_x)

# Output file location
file_name = '../Predictions/CreditCard/best_extra_credit.csv'
# Writing output in Kaggle format
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name, True)

########################## 4
train_x, train_y, test_x  = read_tumor_data()
print('Train=', train_x.shape)
print('Test=', test_x.shape)

clf = SVC(C=0.01, kernel='poly', degree=3, coef0 = 0.5, gamma=0.01)
clf.fit(train_x, train_y)
predicted_y = clf.predict(test_x)

# Output file location
file_name = '../Predictions/Tumor/best_extra_credit.csv'
# Writing output in Kaggle format
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name, False)
