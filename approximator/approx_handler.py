
from prometheus_client import Enum
from traitsui.api import Handler, ViewHandler
from traits.api import List
import os
import subprocess
import shutil
import glob

import numpy as np

home = os.path.expanduser("~")
base_dir = os.path.join(home, ".sphapprox")
thispath = os.path.dirname(os.path.abspath(__file__))
print(thispath)

RESOLUTIONS = [50, 100, 200, 250, 400, 500]
DIMENSION = {'1D':1, '2D':2, '3D':3}
SPH_MEHTOD = { 'standard':'stan' ,
               'order1':'order1',
               'custom':'custom'}
PERTURB = { 'no':0, 'perturb':1, 'pack':2}
PERIODIC = {'No':0, "Yes":1}
DERIVATIVES = {'Function':0, 'gradient':1, 'Laplacian':2, 'divergence':3}


def is_dir_empty():
    listdir = os.listdir(base_dir)
    if len(listdir) == 0:
        return True
    return False

class ApproxHandler(Handler):
    def setattr(self, info, object, name, value):
        Handler.setattr(self, info, object, name, value)
        info.object._updated = True

    def object_ui_changed(self, info):
        if info.initialized:
            info.ui.title += "*"

    def object_hdx_cp_changed(self, info):
        print("hdx changedi")

    def _enable_vis(self, info):
        if not is_dir_empty():
            listdir = os.listdir(base_dir)
            print(info.object.trait_names())
            info.object.values = listdir


    def object_run_changed(self, info):
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
        hdx = info.hdx_cp.value
        dim = info.dimension_cp.value
        method = info.sph_method.value
        pert = info.perturb.value
        derv = info.derv.value
        periodic = info.periodic.value
        norm = info.norm.value

        print(hdx, dim, method, pert, derv, periodic, norm)
        filename = "_".join([method,dim,pert,derv,periodic,norm,"%.2f"%hdx])
        path = os.path.join(base_dir, filename)
        if os.path.exists(path):
            print("run already performed in path %s"%path)
        else:
            os.mkdir(path)
            self.run_cases(info, hdx, dim, method, pert, derv, periodic, norm, path)
        self._enable_vis(info)

    def run_cases(self, info, hdx, dim, method, pert, derv, periodic, norm, dirpath):
        if method == 'custom':
            filename = info.custom_file.value
            methodname = info.methodname.value
            shutil.copy(filename, os.path.join(thispath, 'custom.py'))

        resolutions = RESOLUTIONS
        command = 'python control.py --openmp --dim ' + str(DIMENSION[dim]) + ' --hdx ' + str(hdx) +\
            ' --use-sph ' + SPH_MEHTOD[method] + ' --perturb ' + str(PERTURB[pert]) +\
            ' --derv ' + str(DERIVATIVES[derv]) + ' --periodic ' + str(PERIODIC[periodic]) +\
            ' --norm ' + norm

        for res in resolutions:
            _command = command + ' --N ' + str(res) + ' -d ' + os.path.join(dirpath, 'nx%d'%res)
            print(_command.split(" "))
            subprocess.call(_command.split(" "))
            print("Done")

    def object_reset_changed(self, info):
        print("reset")
        listdir = os.listdir(base_dir)
        if len(listdir) > 0:
            for file in listdir:
                path = os.path.join(base_dir, file)
                print('removed %s'%path)
                shutil.rmtree(path)
                info.object.values = []

    def object_plot_changed(self, info):
        if not is_dir_empty():
            from matplotlib import pyplot as plt
            listdir = os.listdir(base_dir)
            norm = info.norm.value
            print(norm)
            error = None
            for file in listdir:
                path = os.path.join(base_dir, file)
                label = file
                error = []
                for res in RESOLUTIONS:
                    folder = os.path.join(path , "nx%d"%res)
                    datafile = os.path.join(folder, 'results.npz')
                    data = np.load(datafile)
                    fc = data['fc']
                    fe = data['fe']
                    err = None
                    if norm == 'l1':
                        err = sum(abs(fc-fe))/len(fc)
                    elif norm == 'l2':
                        err = np.sqrt(sum((fc-fe)**2)/len(fc))
                    elif norm == 'linf':
                        err = max(abs(fc-fe))
                    error.append(err)
                print(error, res)
                plt.loglog(RESOLUTIONS, error, label=label)

            res = np.array(RESOLUTIONS)
            plt.loglog(res, error[0]*res[0]/res, "--k", label=r'$O(h)$')
            plt.loglog(res, error[0]*res[0]**2/res**2, ":k", label=r'$O(h^2)$')

            plt.grid()
            plt.legend()
            plt.xlabel('N')
            plt.ylabel('Error')
            plt.show()
        else:
            print("No runs found!")

    def object_render_changed(self, info):
        res = info.resolutions.value
        case = info.runs.value
        fullpath = os.path.join(base_dir, case)
        path = os.path.join(fullpath, 'nx'+res)
        datafile = os.path.join(path, 'results.npz')
        data = np.load(datafile)
        x = data['x']
        y = data['y']
        z = data['z']
        fc = data['fc']
        print(info.scene, info.trait_names(), info.object.scene)
        scene = info.object.scene
        scene.mlab.clf()
        out = scene.mlab.points3d(x, y, z, fc, scale_factor=0.01, scale_mode='none')
        print(out)
        scene.mlab.show()

