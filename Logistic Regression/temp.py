import numpy as np
import matplotlib.pyplot as plt

def plotData(X,y):
  # plots the data points with o for the positive examples and x for the negative examples. output is saved to file graph.png
  fig, ax = plt.subplots(figsize=(12,8))
  ##### insert your code here #####
  positive = y>0
  negative = y<0
  ax.scatter(X[positive,0], X[positive,1], c='b', marker='o', label='Healthy')
  ax.scatter(X[negative,0], X[negative,1], c='r', marker='x', label='Not Healthy')

  ax.set_xlabel('Test 1')
  ax.set_ylabel('Test 2')  
  fig.savefig('graph.png') 
  
def predict(X,theta):
  # calculates the prediction h_theta(x) for input(s) x contained in array X
  ##### replace the next line with your code #####
  z=np.matmul(theta,X.T)
  if z>=0:
      pred=1
  else: pred=-1;

  return pred

def computeCost(X, y, theta):
  # function calculates the cost J(theta) and returns its value
  ##### replace the next line with your code #####
  cost = 0
  z=y*(np.matmul(theta,X.T))
  m=y.shape[0]
  cost=(1/m)*np.sum(np.log(1+np.exp(z*-1)))
  return cost

def computeGradient(X,y,theta):
  # calculate the gradient of J(theta) and return its value
  ##### replace the next lines with your code #####
  n=len(theta)
  grad = np.zeros(n)
  z=-1*y*(np.matmul(theta,X.T))
  m=y.shape[0]
  
  grad=(m**-1)*np.sum(np.matmul((np.matmul(y,X.T)),(np.exp(z)/np.exp(1+z))))
  
  return grad

def gradientDescent(X, y):
  # iteratively update parameter vector theta

  # initialize variables for learning rate and iterations
  alpha = 0.1
  iters = 10000
  cost = np.zeros(iters)
  (m,n)=X.shape
  theta= np.zeros(n)

  for i in range(iters):
    theta = theta - alpha * computeGradient(X,y,theta)
    cost[i] = computeCost(X, y, theta)

  return theta, cost

def normaliseData(x):
  # rescale data to lie between 0 and 1
  scale = x.max(axis=0)
  return (x/scale, scale)

def addQuadraticFeature(X):
  # Given feature vector [x_1,x_2] as input, extend this to
  # [x_1,x_2,x_1*x_1] i.e. add a new quadratic feature
  ##### insert your code here #####
  return X

def computeScore(X,y,preds):
  # for training data X,y it calculates the number of correct predictions made by the model 
  ##### replace the next line with your code #####
  score = 0
  return score

def plotDecisionBoundary(Xt,y,Xscale,theta):
  # plots the training data plus the decision boundary in the model
  fig, ax = plt.subplots(figsize=(12,8))
  # plot the data
  positive = y>0
  negative = y<0
  ax.scatter(Xt[positive,1]*Xscale[1], Xt[positive,2]*Xscale[2], c='b', marker='o', label='Healthy')
  ax.scatter(Xt[negative,1]*Xscale[1], Xt[negative,2]*Xscale[2], c='r', marker='x', label='Not Healthy')
  # calc the decision boundary
  x=np.linspace(Xt[:,2].min()*Xscale[2],Xt[:,2].max()*Xscale[2],50)
  if (len(theta) == 3):
    # linear boundary
    x2 = -(theta[0]/Xscale[0]+theta[1]*x/Xscale[1])/theta[2]*Xscale[2]
  else:
    # quadratic boundary
    x2 = -(theta[0]/Xscale[0]+theta[1]*x/Xscale[1]+theta[3]*np.square(x)/Xscale[3])/theta[2]*Xscale[2]
  # and plot it
  ax.plot(x,x2,label='Decision boundary')
  ax.legend()
  ax.set_xlabel('Test 1')
  ax.set_ylabel('Test 2')  
  fig.savefig('pred.png')   
  
def main():
  # load the training data
  data=np.loadtxt('health.csv')
  X=data[:,[0,1]]
  y=data[:,2]
  # add a column of ones to input data
  m=len(y)
  Xt = np.column_stack((np.ones((m, 1)), X))
  (m,n)=Xt.shape # m is number of data points, n number of features

  # rescale training data to lie between 0 and 1
  (Xt,Xscale) = normaliseData(Xt)
  t1=np.array([[2,2,2],[0,0,0],[4,4,4]]);
  t2=np.array([2,3,4])
  print(np.column_stack((t1,t1**2)))
if __name__ == '__main__':
  main()