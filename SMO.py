
# Referred from https://www.youtube.com/watch?v=ZwGaLJbKHiQ


import numpy as np
import prettytable as prettytable
import matplotlib.pyplot as pyplot
import matplotlib.patches as patches
import time

MAX_NUMB_OF_ITERATIONS = 100
EPSILON = 0.00001

C = 10
MIN_ALPHA_OPTIMIZATION = 0.00001

Data = [[[1, 1], [-1]], [[2, 2], [-1]], [[2, 0], [-1]], [[2, 1], [-1]], [[0, 0], [1]], [[1, 0], [1]], [[0, 1], [1]],
        [[-1, -1], [1]]]


class SVM:
    def __init__(self, x, y):
        self._x = x
        self._y = y
        self._alpha = np.mat(np.zeros((np.shape(x)[0], 1)))
        self._b = np.mat([[0]])
        i = 0
        while i < MAX_NUMB_OF_ITERATIONS:
            if self.perform_SMO() == 0:
                i = i + 1
            else:
                i = 0
        self._w = self.calc_w(self._alpha, self._x, self._y)

    def perform_SMO(self):
        numbOfAlphaPairsOptimized = 0
        for i in range(np.shape(self._x)[0]):
            Ei = np.multiply(self._y, self._alpha).T * self._x * self._x[i].T + self._b - self._y[i]
            if self.check_if_alpha_violates_kkt(self._alpha[i], Ei):
                j = self.select_index_of_2nd_aplha_to_optimize(i, np.shape(self._x)[0])
                Ej = np.multiply(self._y, self._alpha).T * self._x * self._x[j].T + self._b - self._y[j]
                alphaIold = self._alpha[i].copy()
                alphaJold = self._alpha[j].copy()
                bounds = self.bound_alpha(self._alpha[i], self._alpha[j], self._y[i], self._y[j])
                ETA = 2.0 * self._x[i] * self._x[j].T - self._x[i] * self._x[i].T - self._x[j] * self._x[j].T
                if bounds[0] != bounds[1] and ETA < 0:
                    if self.optimize_alpha_pair(i, j, Ei, Ej, ETA, bounds, alphaIold, alphaJold):
                        numbOfAlphaPairsOptimized += 1

        return numbOfAlphaPairsOptimized

    def optimize_alpha_pair(self, i, j, Ei, Ej, ETA, bounds, alphaIold, alphaJold):
        flag = False
        self._alpha[j] -= self._y[j] * (Ei - Ej) / ETA
        self.clip_alpha_j(j, bounds)
        if abs(self._alpha[j] - alphaJold) >= MIN_ALPHA_OPTIMIZATION:
            self.optimize_alphai_same_as_alphaj_opposite_direction(i, j, alphaJold)
            self.optimize_b(Ei, Ej, alphaIold, alphaJold, i, j)
            flag = True
        return flag

    def optimize_b(self, Ei, Ej, alphaIold, alphaJold, i, j):
        b1 = self._b - Ei - self._y[i] * (self._alpha[i] - alphaIold) * self._x[i] * self._x[i].T - \
             self._y[j] * (self._alpha[j] - alphaJold) * self._x[i] * self._x[j].T
        b2 = self._b - Ej - self._y[i] * (self._alpha[i] - alphaIold) * self._x[i] * self._x[j].T - \
             self._y[j] * (self._alpha[j] - alphaJold) * self._x[j] * self._x[j].T
        if (0 < self._alpha[i]) and (C > self._alpha[i]):
            self._b = b1
        elif (0 < self._alpha[j]) and (C > self._alpha[j]):
            self._b = b2

        else:
            self._b = (b1 + b2) / 2.0
        return self._b

    def select_index_of_2nd_aplha_to_optimize(self, indexOf1stAlpha, numbOfRows):
        indexOf2ndAlpha = indexOf1stAlpha
        while indexOf1stAlpha == indexOf2ndAlpha:
            indexOf2ndAlpha = int(np.random.uniform(0, numbOfRows))
        return indexOf2ndAlpha

    def optimize_alphai_same_as_alphaj_opposite_direction(self, i, j, alphaJold):
        self._alpha[i] += self._y[j] * self._y[i].T * (alphaJold - self._alpha[j])

    def clip_alpha_j(self, j, bounds):
        if self._alpha[j] < bounds[0]: self._alpha[j] = bounds[0]
        if self._alpha[j] > bounds[1]: self._alpha[j] = bounds[1]

    def check_if_alpha_violates_kkt(self, alpha, E):
        return (alpha > 0 and np.abs(E) < EPSILON) or (alpha < C and np.abs(E) > EPSILON)

    def bound_alpha(self, alphai, alphaj, yi, yj):
        bounds = [2]
        if yi == yj:
            bounds.insert(0, max(0, alphaj + alphai - C))
            bounds.insert(1, min(C, alphaj + alphai))
        else:
            bounds.insert(0, max(0, alphaj - alphai))
            bounds.insert(1, min(C, alphaj - alphai + C))
        return bounds

    def calc_w(self, alpha, x, y):
        w = np.zeros((np.shape(x)[1], 1))
        for i in range(np.shape(x)[0]):
            w += np.multiply(y[i] * alpha[i], x[i].T)
        return w

    def classify(self, x):
        classification = "Classified as -1"
        if np.sign((x @ self._w + self._b).item(0, 0)) == 1:
            classification = "Classified as 1"
        return classification

    def get_alpha(self):
        return self._alpha

    def get_b(self):
        return self._b

    def get_w(self):
        return self._w


def display_info_tables():
    svTable = prettytable.PrettyTable(['Support vector', 'Label', 'Alpha'])
    for i in range(len(xArray)):
        if svm.get_alpha()[i] > 0.0 and svm.get_alpha()[i] != C:
            svTable.add_row([xArray[i], yArray[i], svm.get_alpha()[i].item(0, 0)])
        print(svTable)
        wbTable = prettytable.PrettyTable(['wT', 'b'])
        wbTable.add_row([svm.get_w().T, svm.get_b()])
        print(wbTable)


def plot(alpha, b, w):
    x0 = []
    y0 = []
    x1 = []
    y1 = []
    for i in range(0, len(Data)):
        if Data[i][1][0] == -1:
            x0.append(Data[i][0][0]);
            y0.append(Data[i][0][1])
        else:
            x1.append(Data[i][0][0]);
            y1.append(Data[i][0][1])
    plot = pyplot.figure()
    ax = plot.add_subplot(1, 1, 1)
    ax.scatter(x0, y0, marker='o', s=30, c='orange')
    ax.scatter(x1, y1, marker='o', s=30, c='green')
    for i in range(len(xArray)):
        if alpha[i] > 0.0 and alpha[i] != C:
            ax.add_patch(patches.CirclePolygon(
                (xArray[i][0], xArray[i][0]), 0.25, facecolor='none', edgecolor=(0, 0, 0), linewidth=1, alpha=0.9
            ))
    x = np.arange(-5.0, 20.0, 0.1)
    y = (-w[0] * x - b) / w[1]
    ax.plot(x, y)
    ax.axis([-5, 10, -5, 10])
    pyplot.show()


start_time = time.time()
xArray = []
yArray = []
for i in range(0, len(Data)):
    xArray.append(Data[i][0])
    yArray.append(Data[i][1][0])
print(xArray, yArray)
svm = SVM(np.mat(xArray), np.mat(yArray).T)
display_info_tables()
print("Close")

print("--- %s seconds ---" % (time.time() - start_time))
plot(svm.get_alpha(), svm.get_b().item(0, 0), svm.get_w())

