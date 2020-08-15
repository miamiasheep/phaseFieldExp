from fipy import Variable, CellVariable, Grid2D, TransientTerm, DiffusionTerm, ImplicitSourceTerm, \
    Matplotlib2DGridViewer
from fipy.tools import numerix
from builtins import range
import pylab

class DentriteViewer(Matplotlib2DGridViewer):
    def __init__(self, dT, title=None, limits={}, **kwlimits):
        self.phase = phase
        self.contour = None
        Matplotlib2DGridViewer.__init__(self, vars=(dT, ), title=title, cmap=pylab.cm.hot, limits=limits,
                                        **kwlimits)

    def _plot(self):
        Matplotlib2DGridViewer._plot(self)

        if self.contour is not None:
            for c in self.contour.collections:
                c.remove()

        mesh = self.phase.mesh
        shape = mesh.shape
        x, y = mesh.cellCenters
        z = self.phase.value
        x, y, z = [a.reshape(shape, order='F') for a in (x, y, z)]
        self.contour = self.axes.contour(x, y, z, (0.5,))


if __name__ == '__main__':
    # initialize mesh
    nx = ny = 500
    dx = dy = 0.025
    mesh = Grid2D(dx, dy, nx, ny)

    # dt is used for the interval to calculate differential
    dt = 5e-4

    # heat flux equation (D is the diffusion constant)
    D = 2.25
    phase = CellVariable(name=r'$\phi$', mesh=mesh, hasOld=True)
    dT = CellVariable(name=r'$\Delta T$', mesh=mesh, hasOld=True)
    heatEq = (TransientTerm() == DiffusionTerm(D) + (phase - phase.old) / dt)

    #
    alpha = 0.015
    c = 0.02
    N = 6.0
    theta = numerix.pi / 8.0
    psi = theta + numerix.arctan2(phase.faceGrad[1], phase.faceGrad[0])
    Phi = numerix.tan(N * psi / 2)
    PhiSq = Phi ** 2
    beta = (1. - PhiSq) / (1 + PhiSq)
    DbetaDpsi = -N * 2 * Phi / (1 + PhiSq)
    Ddia = (1. + c * beta)
    Doff = c * DbetaDpsi
    I0 = Variable(value=((1, 0), (0, 1)))
    I1 = Variable(value=((0, -1), (1, 0)))
    D = alpha ** 2 * (1. + c * beta) * (Ddia * I0 + Doff * I1)

    tau = 3e-4
    kappa1 = 0.9
    kappa2 = 20.
    phaseEq = (TransientTerm(tau)
               == DiffusionTerm(D)
               + ImplicitSourceTerm((phase - 0.5 - kappa1 / numerix.pi * numerix.arctan(kappa2 * dT))
                                    * (1 - phase)))

    radius = dx * 5.
    C = (nx * dx / 2, ny * dy / 2)
    x, y = mesh.cellCenters
    phase.setValue(1., where=((x - C[0]) ** 2 + (y - C[1]) ** 2) < radius ** 2)
    dT.setValue(-0.5)

    steps = 201
    granularity = 10
    fileName = 'snapshot/exp3'
    viewer = DentriteViewer(phase=phase, dT=dT, title=r"%s & %s" % (phase.name, dT.name), datamin=-0.1, datamax=0.05)
    for i in range(steps):
        phase.updateOld()
        dT.updateOld()
        phaseEq.solve(phase, dt=dt)
        heatEq.solve(dT, dt=dt)
        if i % granularity == 0:
            viewer.plot(filename='{}_{}.png'.format(fileName, i))
