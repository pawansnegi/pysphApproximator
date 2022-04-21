import os

import numpy as np
from traits.api import HasTraits, Instance, Str, Int, Float, Bool, List, File
from traits.api import Range, Enum, Button
from traitsui.api import Item, View, Group, HSplit, VSplit, CheckListEditor, CodeEditor
from mayavi.core.ui.api import MayaviScene, MlabSceneModel
from mayavi.core.ui.api import SceneEditor

from approx_handler import (
    CodeEditorHandler, ApproxHandler,
    DIMENSION, DERIVATIVES, SPH_MEHTOD, PERIODIC, PERTURB)

home = os.path.expanduser("~")
base_dir = os.path.join(home, ".sphapprox")

def is_dir_empty():
    listdir = os.listdir(base_dir)
    if len(listdir) == 0:
        return True
    return False

class EditCode(HasTraits):
    filepath = Str()
    codeeditor = Str()
    save = Button('Save')
    close = Button('Close')

    view = View(
        Item(name='codeeditor', show_label=False, editor=CodeEditor()),
        Group(
            Item(name='save', show_label=False),
            Item(name='close', show_label=False),
            orientation='horizontal'
        ),
        handler=CodeEditorHandler()
    )

class ApproximationUI(HasTraits):

    _codeEditor = EditCode()
    scene = Instance(MlabSceneModel, args=())
    values = List([])

    hdx_cp = Float(1.2)
    frac = Float(0.1)
    dimension_cp = Enum(*DIMENSION.keys())
    sph_method = Enum(*SPH_MEHTOD.keys())
    custom_file = File("", exists=True, filter=['*.py'])
    methodname = Str()
    editcode = Button("Edit code")
    perturb = Enum(*PERTURB.keys())
    derv = Enum(*DERIVATIVES.keys())
    periodic = Enum(*PERIODIC.keys())
    norm = Enum('l1', 'l2', 'linf')
    run = Button("Run")
    plot = Button("Plot")
    reset = Button("Reset")
    remove = Button("Remove")

    runs = Enum(values='values')
    resolutions = Enum('50', '100', '200', '250', '500')
    render = Button()


    view = View(Group(
        HSplit(
            Group(
                Group(
                    Item(name='sph_method', label='Approximation method', springy=True),
                    Item(name='editcode', show_label=False),
                    Item(name="methodname", label="Custom name", springy=True,
                         enabled_when="sph_method == 'custom'"),
                    Item(name="custom_file", show_label=False, springy=True,
                         enabled_when="sph_method == 'custom'"),
                    show_border=True, orientation='horizontal', columns=2),
                Item(name='hdx_cp', label='Scaling parameter', springy=True),
                Item(name='dimension_cp', style='custom',
                     label='Problem dimension',
                     springy=True),
                Group(
                    Item(name='perturb', label='Mesh kind',  style='custom'),
                    Item(name="frac", label='fraction', enabled_when='perturb == "perturb"'),
                    orientation='horizontal',
                    show_border=True,
                    layout='split',
                ),
                Item(name='derv', label='Operator', style='custom', springy=True),
                Item(name='periodic', label='Is periodic', style='custom', springy=True),
                Item(name='norm', label='Norm', style='custom', springy=True),
                Group(Item(name="run", show_label=False, springy=True),
                      Item(name="plot",
                           show_label=False,
                           springy=True,
                           enabled_when='len(values) > 0'),
                      show_border=True,
                      layout='split',
                      orientation='horizontal'),
                Item(name='reset', show_label=False),
                show_border=True,
            ),
            VSplit(
                Group(Item(name='runs', springy=True),
                      Item('remove', springy=True, show_label=False),
                      Item('resolutions', springy=True),
                      Item('render', show_label=False,
                           springy=True,
                           enabled_when='len(values) > 0'),
                      layout='split',
                      orientation='horizontal'),
                Item(name='scene',
                     editor=SceneEditor(scene_class=MayaviScene),
                     show_label=False,
                     resizable=True,
                     height=400,
                     width=400)), )),
                resizable=True,
                title='Approximator',
                handler=ApproxHandler())



if __name__ == '__main__':
    ui = ApproximationUI()
    ui.configure_traits()