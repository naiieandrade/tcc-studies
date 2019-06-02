import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
import matplotlib.tri as tri

corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])
triangle = tri.Triangulation(corners[:, 0], corners[:, 1])

refiner = tri.UniformTriRefiner(triangle)
trimesh = refiner.refine_triangulation(subdiv=4)

# plt.figure(figsize=(8, 4))
# for (i, mesh) in enumerate((triangle, trimesh)):
#     plt.subplot(1, 2, i + 1)
#     plt.triplot(mesh)
#     plt.axis('off')
#     plt.axis('equal')


# Mid-points of triangle sides opposite of each corner
midpoints = [(corners[(i + 1) % 3] + corners[(i + 2) % 3]) / 2.0 \
             for i in range(3)]


def xy2bc(xy, tol=1.e-3):
    '''Converts 2D Cartesian coordinates to barycentric.'''
    s = [(corners[i] - midpoints[i]).dot(xy - midpoints[i]) / 0.75 \
         for i in range(3)]
    return np.clip(s, tol, 1.0 - tol)


class Dirichlet(object):
    def __init__(self, alpha):
        from math import gamma
        from operator import mul
        self._alpha = np.array(alpha)
        self._coef = gamma(np.sum(self._alpha)) / reduce(mul, [gamma(a) for a in self._alpha])

    def pdf(self, x):
        '''Returns pdf value for `x`.'''
        from operator import mul
        return self._coef * reduce(mul, [xx ** (aa - 1)
                                         for (xx, aa)in zip(x, self._alpha)])


def draw_pdf_contours(dist, nlevels=200, subdiv=8, **kwargs):
    import math

    refiner = tri.UniformTriRefiner(triangle)
    trimesh = refiner.refine_triangulation(subdiv=subdiv)
    pvals = [dist.pdf(xy2bc(xy)) for xy in zip(trimesh.x, trimesh.y)]

    plt.tricontourf(trimesh, pvals, nlevels, **kwargs)
    plt.axis('equal')
    plt.xlim(0, 1)
    plt.ylim(0, 0.75**0.5)
    plt.axis('off')


def plot_alphas_distribution(alpha):
    a = alpha
    s = np.random.dirichlet((a, a, a, a, a, a, a, a, a, a), 10)
    fig = plt.figure()
    fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = plt.subplots(2, 5)

    ax1.plot([0,0], [0, s[0][0]])
    ax1.plot([1,1], [0, s[0][1]])
    ax1.plot([2,2], [0, s[0][2]])
    ax1.plot([3,3], [0, s[0][3]])
    ax1.plot([4,4], [0, s[0][4]])
    ax1.plot([5,5], [0, s[0][5]])
    ax1.plot([6,6], [0, s[0][6]])
    ax1.plot([7,7], [0, s[0][7]])
    ax1.plot([8,8], [0, s[0][8]])
    ax1.plot([9,9], [0, s[0][9]])
    ax1.axes.get_xaxis().set_ticks([])
    ax1.set_xlim([0,10])
    ax1.set_ylim([0,1])


    ax2.plot([0,0], [0, s[1][0]])
    ax2.plot([1,1], [0, s[1][1]])
    ax2.plot([2,2], [0, s[1][2]])
    ax2.plot([3,3], [0, s[1][3]])
    ax2.plot([4,4], [0, s[1][4]])
    ax2.plot([5,5], [0, s[1][5]])
    ax2.plot([6,6], [0, s[1][6]])
    ax2.plot([7,7], [0, s[1][7]])
    ax2.plot([8,8], [0, s[1][8]])
    ax2.plot([9,9], [0, s[1][9]])
    ax2.axes.get_xaxis().set_ticks([])
    ax2.axes.get_yaxis().set_ticks([])
    ax2.set_xlim([0,10])
    ax2.set_ylim([0,1])


    ax3.plot([0,0], [0, s[2][0]])
    ax3.plot([1,1], [0, s[2][1]])
    ax3.plot([2,2], [0, s[2][2]])
    ax3.plot([3,3], [0, s[2][3]])
    ax3.plot([4,4], [0, s[2][4]])
    ax3.plot([5,5], [0, s[2][5]])
    ax3.plot([6,6], [0, s[2][6]])
    ax3.plot([7,7], [0, s[2][7]])
    ax3.plot([8,8], [0, s[2][8]])
    ax3.plot([9,9], [0, s[2][9]])
    ax3.axes.get_xaxis().set_ticks([])
    ax3.axes.get_yaxis().set_ticks([])
    ax3.set_xlim([0,10])
    ax3.set_ylim([0,1])


    ax4.plot([0,0], [0, s[3][0]])
    ax4.plot([1,1], [0, s[3][1]])
    ax4.plot([2,2], [0, s[3][2]])
    ax4.plot([3,3], [0, s[3][3]])
    ax4.plot([4,4], [0, s[3][4]])
    ax4.plot([5,5], [0, s[3][5]])
    ax4.plot([6,6], [0, s[3][6]])
    ax4.plot([7,7], [0, s[3][7]])
    ax4.plot([8,8], [0, s[3][8]])
    ax4.plot([9,9], [0, s[3][9]])
    ax4.axes.get_xaxis().set_ticks([])
    ax4.axes.get_yaxis().set_ticks([])
    ax4.set_xlim([0,10])
    ax4.set_ylim([0,1])

    ax5.plot([0,0], [0, s[4][0]])
    ax5.plot([1,1], [0, s[4][1]])
    ax5.plot([2,2], [0, s[4][2]])
    ax5.plot([3,3], [0, s[4][3]])
    ax5.plot([4,4], [0, s[4][4]])
    ax5.plot([5,5], [0, s[4][5]])
    ax5.plot([6,6], [0, s[4][6]])
    ax5.plot([7,7], [0, s[4][7]])
    ax5.plot([8,8], [0, s[4][8]])
    ax5.plot([9,9], [0, s[4][9]])
    ax5.axes.get_xaxis().set_ticks([])
    ax5.axes.get_yaxis().set_ticks([])
    ax5.set_xlim([0,10])
    ax5.set_ylim([0,1])

    ax6.plot([0,0], [0, s[5][0]])
    ax6.plot([1,1], [0, s[5][1]])
    ax6.plot([2,2], [0, s[5][2]])
    ax6.plot([3,3], [0, s[5][3]])
    ax6.plot([4,4], [0, s[5][4]])
    ax6.plot([5,5], [0, s[5][5]])
    ax6.plot([6,6], [0, s[5][6]])
    ax6.plot([7,7], [0, s[5][7]])
    ax6.plot([8,8], [0, s[5][8]])
    ax6.plot([9,9], [0, s[5][9]])
    ax6.axes.get_xaxis().set_ticks([])
    #ax6.axes.get_yaxis().set_ticks([])
    ax6.set_xlim([0,10])
    ax6.set_ylim([0,1])

    ax7.plot([0,0], [0, s[6][0]])
    ax7.plot([1,1], [0, s[6][1]])
    ax7.plot([2,2], [0, s[6][2]])
    ax7.plot([3,3], [0, s[6][3]])
    ax7.plot([4,4], [0, s[6][4]])
    ax7.plot([5,5], [0, s[6][5]])
    ax7.plot([6,6], [0, s[6][6]])
    ax7.plot([7,7], [0, s[6][7]])
    ax7.plot([8,8], [0, s[6][8]])
    ax7.plot([9,9], [0, s[6][9]])
    ax7.axes.get_xaxis().set_ticks([])
    ax7.axes.get_yaxis().set_ticks([])
    ax7.set_xlim([0,10])
    ax7.set_ylim([0,1])

    ax8.plot([0,0], [0, s[7][0]])
    ax8.plot([1,1], [0, s[7][1]])
    ax8.plot([2,2], [0, s[7][2]])
    ax8.plot([3,3], [0, s[7][3]])
    ax8.plot([4,4], [0, s[7][4]])
    ax8.plot([5,5], [0, s[7][5]])
    ax8.plot([6,6], [0, s[7][6]])
    ax8.plot([7,7], [0, s[7][7]])
    ax8.plot([8,8], [0, s[7][8]])
    ax8.plot([9,9], [0, s[7][9]])
    ax8.axes.get_xaxis().set_ticks([])
    ax8.axes.get_yaxis().set_ticks([])
    ax8.set_xlim([0,10])
    ax8.set_ylim([0,1])

    ax9.plot([0,0], [0, s[8][0]])
    ax9.plot([1,1], [0, s[8][1]])
    ax9.plot([2,2], [0, s[8][2]])
    ax9.plot([3,3], [0, s[8][3]])
    ax9.plot([4,4], [0, s[8][4]])
    ax9.plot([5,5], [0, s[8][5]])
    ax9.plot([6,6], [0, s[8][6]])
    ax9.plot([7,7], [0, s[8][7]])
    ax9.plot([8,8], [0, s[8][8]])
    ax9.plot([9,9], [0, s[8][9]])
    ax9.axes.get_xaxis().set_ticks([])
    ax9.axes.get_yaxis().set_ticks([])
    ax9.set_xlim([0,10])
    ax9.set_ylim([0,1])

    ax10.plot([0,0], [0, s[9][0]])
    ax10.plot([1,1], [0, s[9][1]])
    ax10.plot([2,2], [0, s[9][2]])
    ax10.plot([3,3], [0, s[9][3]])
    ax10.plot([4,4], [0, s[9][4]])
    ax10.plot([5,5], [0, s[9][5]])
    ax10.plot([6,6], [0, s[9][6]])
    ax10.plot([7,7], [0, s[9][7]])
    ax10.plot([8,8], [0, s[9][8]])
    ax10.plot([9,9], [0, s[9][9]])
    ax10.axes.get_xaxis().set_ticks([])
    ax10.axes.get_yaxis().set_ticks([])
    ax10.set_xlim([0,10])
    ax10.set_ylim([0,1])


    plt.show()
