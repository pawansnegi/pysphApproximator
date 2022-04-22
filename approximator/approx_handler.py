
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
if not os.path.exists(base_dir):
    os.mkdir(base_dir)
thispath = os.path.dirname(os.path.abspath(__file__))

RESOLUTIONS = [50, 100, 200, 250, 400, 500]
DIMENSION = {'1D':1, '2D':2, '3D':3}
SPH_MEHTOD = { 'standard':'stan' ,
               'order1':'order1',
               'order2':'order2',
               'cleary':'cleary',
               'custom':'custom',}
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

    def init(self, info):
        self._enable_vis(info)

    def _enable_vis(self, info):
        if not is_dir_empty():
            listdir = os.listdir(base_dir)
            info.object.values = listdir

    def object_run_changed(self, info):
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
        hdx = info.hdx_cp.value
        dim = info.dimension_cp.value
        method = info.sph_method.value
        pert = info.perturb.value
        derv = info.derv.value
        frac = info.frac.value
        periodic = info.periodic.value
        norm = info.norm.value

        methodname = self._get_method_name(info, method)
        filename = "_".join([methodname,dim,pert,derv,periodic,norm,"%.2f"%frac,"%.2f"%hdx])
        path = os.path.join(base_dir, filename)
        complete = False
        if os.path.exists(path):
            print("run already performed in path %s"%path)
        else:
            os.mkdir(path)
            complete = self.run_cases(info, hdx, dim, method, pert, derv, periodic, norm, frac, path)

        if complete:
            self._enable_vis(info)

    def _get_method_name(self, info, method):
        if method == 'custom':
            methodname = info.methodname.value
            if (len(methodname) == 0):
                print("method name is set 'custom' as methodName was missing")
                return method
            return methodname
        return method

    def run_cases(self, info, hdx, dim, method, pert, derv, periodic, norm, frac, dirpath):
        resolutions = RESOLUTIONS
        command = 'python control.py --openmp --dim ' + str(DIMENSION[dim]) + ' --hdx ' + str(hdx) +\
            ' --use-sph ' + SPH_MEHTOD[method] + ' --perturb ' + str(PERTURB[pert]) +\
            ' --derv ' + str(DERIVATIVES[derv]) + ' --periodic ' + str(PERIODIC[periodic]) +\
            ' --norm ' + norm + ' --frac ' + str(frac)

        for res in resolutions:
            _command = command + ' --N ' + str(res) + ' -d ' + os.path.join(dirpath, 'nx%d'%res)
            print(_command)
            out = subprocess.call(_command.split(" "))
            if not (out == 0):
                print("One of the option is not valid...")
                shutil.rmtree(dirpath)
                return False
            print("Done")
        return True

    def object_custom_file_changed(self, info):
        method = info.sph_method.value
        if method == 'custom':
            filename = info.custom_file.value
            if not len(filename) == 0:
                newpath = os.path.join(thispath, 'custom.py')
                shutil.copy(filename, newpath)


    def object_reset_changed(self, info):
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
            fig = plt.figure(figsize=(12, 6))
            listdir = os.listdir(base_dir)
            norm = info.norm.value
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
                plt.loglog(RESOLUTIONS, error, label=label)

            res = np.array(RESOLUTIONS)
            plt.loglog(res, error[0]*res[0]/res, "--k", label=r'$O(h)$')
            plt.loglog(res, error[0]*res[0]**2/res**2, ":k", label=r'$O(h^2)$')

            plt.grid()
            plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
            plt.tight_layout(pad=1.5)
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
        scene = info.object.scene
        scene.mlab.clf()
        out = scene.mlab.points3d(x, y, z, fc, scale_factor=0.01, scale_mode='none')
        scene.mlab.show()

    def object_editcode_changed(self, info):
        print('edit code')
        print(info.trait_names())
        print(info.object._codeEditor.trait_names())
        filepath = ""
        if info.sph_method.value == 'custom':
            filepath = os.path.join(thispath, 'custom.py')
        else:
            approx_folder = os.path.join(thispath, 'approx')
            methodname = info.sph_method.value
            filepath = os.path.join(approx_folder, methodname + ".py")
        info.object._codeEditor.filepath = filepath
        fp = open(filepath)
        lines = "".join(fp.readlines())
        fp.close()
        info.object._codeEditor.codeeditor = lines
        info.object._codeEditor.configure_traits()

    def object_remove_changed(self, info):
        fname = info.runs.value
        folder = os.path.join(base_dir, fname)
        shutil.rmtree(folder)
        self._enable_vis(info)
        print("removed", folder)

class CodeEditorHandler(Handler):
    def setattr(self, info, object, name, value):
        Handler.setattr(self, info, object, name, value)
        info.object._updated = True

    def object_save_changed(self, info):
        filepath = info.object.filepath
        data = info.codeeditor.value
        fp = open(filepath, 'w')
        fp.writelines(data)
        fp.close()
        print("file saved")

    def object_close_changed(self, info):
        info.ui.dispose()
        print("Closed")
