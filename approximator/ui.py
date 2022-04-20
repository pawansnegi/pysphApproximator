import os

import numpy as np
from traits.api import HasTraits, Instance, Str, Int, Float, Bool, List, File
from traits.api import Range, Enum, Button
from traitsui.api import Item, View, Group, HSplit, VSplit, CheckListEditor
from mayavi.core.ui.api import MayaviScene, MlabSceneModel
from mayavi.core.ui.api import SceneEditor
from approx_handler import (
    ApproxHandler, DIMENSION, DERIVATIVES, SPH_MEHTOD, PERIODIC, PERTURB)

home = os.path.expanduser("~")
base_dir = os.path.join(home, ".sphapprox")

def is_dir_empty():
    listdir = os.listdir(base_dir)
    if len(listdir) == 0:
        return True
    return False

class ApproximationUI(HasTraits):

    scene = Instance(MlabSceneModel, args=())

    hdx_cp = Float(1.2)
    dimension_cp = Enum(*DIMENSION.keys())
    sph_method = Enum(*SPH_MEHTOD.keys())
    custom_file = File("", exists=True, filter=['*.py'])
    methodname = Str()
    perturb = Enum(*PERTURB.keys())
    derv = Enum(*DERIVATIVES.keys())
    periodic = Enum(*PERIODIC.keys())
    norm = Enum('l1', 'l2', 'linf')
    run = Button("Run")
    plot = Button("Plot")
    reset = Button("Reset")

    runs = Enum(" ")
    resolutions = Enum('50', '100', '200', '250', '500')
    render = Button()

    view = View(Group(
        HSplit(
            Group(
                Group(
                    Item(name='sph_method', label='Approximation method', springy=True),
                    Item(name="methodname", label="Custom name", springy=True,
                         enabled_when="sph_method.value == 'custom'"),
                    Item(name="custom_file", show_label=False, springy=True,
                         enabled_when="sph_method.value == 'custom'"),
                    show_border=True, orientation='horizontal', columns=2),
                Item(name='hdx_cp', label='Scaling parameter', springy=True),
                Item(name='dimension_cp',
                     label='Problem dimension',
                     springy=True),
                Item(name='perturb', label='Mesh kind', springy=True),
                Item(name='derv', label='Operator', springy=True),
                Item(name='periodic', label='Is periodic', springy=True),
                Item(name='norm', label='Norm', springy=True),
                Group(Item(name="run", show_label=False, springy=True),
                      Item(name="plot",
                           show_label=False,
                           springy=True,
                           enabled_when='len(runs.values) > 0'),
                      show_border=True,
                      layout='split',
                      orientation='horizontal'),
                Item(name='reset', show_label=False),
                show_border=True,
            ),
            Group(
                Group(Item(name='runs', springy=True, style='custom'),
                      Item('resolutions', springy=True),
                      Item('render',
                           springy=True,
                           enabled_when='len(runs.values) > 0'),
                      layout='split',
                      orientation='horizontal'),
                Item(name='scene',
                     editor=SceneEditor(scene_class=MayaviScene),
                     show_label=False,
                     resizable=True,
                     height=200,
                     width=400)))),
                resizable=True,
                title='Approximator',
                handler=ApproxHandler())



if __name__ == '__main__':
    ui = ApproximationUI()
    ui.configure_traits()