import sys
import os
import shutil

import numpy as np
import pyqtgraph as pg
import pandas as pd
import dill

from pyqtgraph.Qt import QtCore, QtGui

from components.CrosshairOverlay import CrosshairOverlay

def psave(path, variable):
    '''
    psave(path, variable)
    
    Takes a variable (given as string with its name) and saves it to a file as specified in the path.
    The path must at least contain the filename (no file ending needed), and can also include a 
    relative or an absolute folderpath, if the file is not to be saved to the current working directory.
    
    # ToDo: save several variables (e.g. take X args, store them to special DICT, and save to file)
    '''
    print("saving " + str(variable)[:15] + " to " + str(path) + "...")
    if(path.find('.pickledump')==-1):
        path = path + '.pickledump'
    path = path.replace('\\','/')
    if not os.path.exists(path[0:-17]):
        os.makedirs(path[0:-17])
    print(f"PATH PATH PATH {path}")
    file = open(path,mode="w+b")
    dill.dump(variable,file,protocol=4) #pickle.dump
    print("saved " + str(variable)[:15] + " to " + str(path) + ".")

#%%
def pload(path):
    '''
    variable = pload(path)
    
    Loads a variable from a file that was specified in the path. The path must at least contain the 
    filename (no file ending needed), and can also include a relative or an absolute folderpath, if 
    the file is not to located in the current working directory.
    
    # ToDo: load several variables (e.g. load special DICT from file and return matching entries)
    '''
    if(path.find('.pickledump')==-1):
        path = path + '.pickledump'
    path = path.replace('\\','/')
    # cwd = os.getcwd().replace('\\','/')
    #if(path[0:2]!=cwd[0:2]):
    #    path = os.path.abspath(cwd + '/' + path) # If relative path was given, turn into absolute path
    file = open(path, 'rb')
    return dill.load(file) #pickle.load

def print_dict(dictionary,filterlist=None,printit=True):
    ''' 
    textstring = print_dict(dictionary)
    
    This takes a dictonary as an input and returns its keys and values as text,
    such that overly long values are not shown. If a string 'filterlist' is 
    provided, only entries of that list will be shown
    '''
    text = ''
    keys = list(dictionary.keys())
    maxlen = len(max(keys, key=len)) + 1
    for key in keys:
        value = dictionary[key]
        infilter = True if filterlist is None else (key in filterlist)
        if(infilter and type(value)==list):
            if(0<len(value)<=3 and type(value[0])==int):
                text = text + key + ' '*(maxlen-len(key))+ ': '+str(value)+'\n'
            else:
                text = text + key + ' '*(maxlen-len(key))+ ': (list with '+str(len(value))+' elements)\n'
        elif(infilter and 'numpy.ndarray' in str(type(value))):
            if(len(value.shape)==1 and value.shape[0]<5):
                text = text + key + ' '*(maxlen-len(key))+ ': '+str(value)+'\n'
            else:
                text = text + key + ' '*(maxlen-len(key))+ ': (array of size '+str(value.shape)+')\n'
        elif(infilter and type(value)==dict):
            text = text + key + ' '*(maxlen-len(key))+ ':\n'
            deeptext = print_dict(dictionary[key],filterlist=filterlist,printit=False)
            for line in iter(deeptext.splitlines()):
                text = text + ' |-> ' + line + '\n'
        elif(infilter and('float' in str(type(value)) or 'int' in str(type(value)))):
            text = text + key + ' '*(maxlen-len(key))+ ': '  + str(round(value,2)) + '\n' # scalar metrics
        elif(infilter):
            text = text + key + ' '*(maxlen-len(key))+ ': '  + str(value)  + '\n' # vector metrics or text
    if(printit):
        print(text)
    else:
        return text

def load_region(path):
    global win
    region = pload(path)# + "region")
    image = region["thumbnails"]["MaxProjections_Z"]
    print_dict(region)
    win.setWindowTitle(region["name"])
    return region, image

def move_patches(event):
    global region, overlay_list, path_region
    path_candidates = QtGui.QFileDialog.getExistingDirectory(None, 'Open folder', '',  QtGui.QFileDialog.ShowDirsOnly)#[0]
    local_path = region["dataset"]["localfolder"]
    if not os.path.exists(local_path):
        local_path = path_region.replace("Sync/region.pickledump","Local/")
    print(f"Moving from \n\t{local_path}\nto\n\t{path_candidates}")
    for item in overlay_list:
        if item["patch"]["locationgroup"] == "Candidate":
            stb.showMessage("Moving {}".format(item["patch"]["id"]))
            shutil.copyfile(local_path + "patchvolume_{}.nii.gz".format(item["patch"]["id"]), path_candidates + "/patchvolume_{}.nii.gz".format(item["patch"]["id"]))
    print("Moved candidates")
    stb.showMessage("")

def delete_patches(event):
    global region, overlay_list
    empty_patches = [item["patch"] for item in overlay_list if item["patch"]["locationgroup"] == "Outside"]
    answer = QtGui.QMessageBox.question(None, "Delete empty patches", "Are you sure you want to delete {} empty patches? There is no Undo if you select \"Yes\".".format(len(empty_patches)), QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)
    if answer == QtGui.QMessageBox.Yes:
        for item in empty_patches:
            try:
                os.remove(region["dataset"]["localfolder"] + "patchvolume_{}.nii".format(item["id"]))
            except:
                # print "Unexpected error:", sys.exc_info()[0]
                print("Problem at {}: {}".format(item, sys.exc_info()[0]))
            stb.showMessage("Removing {}".format(item["id"]))
    print("Removed {} files.".format(len(empty_patches)))

def open_region(event):
    global overlay_list, imv, image_transposed, path_region, region, image, combo_slice
    path_region = QtGui.QFileDialog.getOpenFileName(None, 'Open file', '' , "Region files (*.pickledump)")[0]
    print(path_region)

    region, image = load_region(path_region)

    print("IMAGE SHAPE {}".format(image.shape))

    image_transposed = np.flip(np.rot90(np.swapaxes(image,0,2),axes=(1,2)),axis=1)
    imv.setImage(image_transposed)

    for item in overlay_list:
        imv.scene.removeItem(item["crosshair"])
    ovelay_list = []

    overlay_list, existing_threshold = generate_overlay_list(region, imv)
    if not existing_threshold:
        auto_threshold()
    # Set slices as combobox entries
    entry_list = ["Slice {}".format(i) for i in range(image_transposed.shape[0])]
    combo_slice.setItems(entry_list)

def save_region(event):
    stb.showMessage("Saving file to {}.pickledump ...".format(path_region))
    for item_overlay, item_region  in zip(overlay_list, region["patches"]):
        if item_region["id"] == item_overlay["patch"]["id"]:
            region["patches"][item_region["id"]] = item_overlay["patch"]
    psave(path_region, region)# + "region", region)
    stb.showMessage("Saved file to {}.pickledump!".format(path_region))
    print("Saved file to {}!".format(path_region))

#TODO Move segmentation candidates
#TODO Delete outside parts

def save_region_as(event):
    path_region = QtGui.QFileDialog.getSaveFileName(None, 'Save file', '' , "Region files (*.pickledump)")[0]
    for item_overlay, item_region  in zip(overlay_list, region["patches"]):
        if item_region["id"] == item_overlay["patch"]["id"]:
            region["patches"][item_region["id"]] = item_overlay["patch"]
    print(path_region)
    psave(path_region, region)# + "region", region)
    print("Saved file to {}!".format(path_region))

def generate_overlay_list(region, imageview):
    overlay_list = []
    existing_threshold = False
    region_len = len(region["patches"])
    print("Patch overlap = " + str(region["partitioning"]["patch_overlap"]))
    overlap_correction = 0
    seg_l = []
    if region["partitioning"]["patch_overlap"] > 0:
        overlap_correction = region["partitioning"]["patch_overlap"] * region["thumbnails"]["downsampling"]
        pass
    #TODO
    for i, patch in enumerate(region["patches"]):
        stb.showMessage("Generating overlay list for patch {} of {}".format(i, region_len))
        print("Generating overlay for {}".format(patch['id']), end="\r", flush=True)
        if('locationgroup' not in patch.keys()):
            patch['locationgroup'] = 'Outside' # create entry for any patches that don't have it yet.
        else:
            existing_threshold = True

        overlay_item = {}
        overlay_item['patch'] = patch

        # x_coord = patch['patchstep'][1] * 19 + 10
        # y_coord = patch['patchstep'][0] * 19 + 10
        # z_coord = patch['patchstep'][2]

        patch_dim = region["partitioning"]["patch_size"][0] * region["thumbnails"]["downsampling"]# - region["partitioning"]["patch_overlap"] * region["thumbnails"]["downsampling"]# * 2
        
        #TODO Fix, weird behaviour for legacy overlap code
        # x_coord = patch['patchstep'][1] * patch_dim + overlap_correction + patch_dim * 0.5
        # y_coord = patch['patchstep'][0] * patch_dim + overlap_correction + patch_dim * 0.5
        x_coord = patch['patchstep'][1] * (patch_dim)+ patch_dim * 0.5 - patch['patchstep'][1] * 0.5
        y_coord = patch['patchstep'][0] * (patch_dim)+ patch_dim * 0.5 - patch['patchstep'][0] * 0.5
        z_coord = patch['patchstep'][2]

        pen = QtGui.QPen(QtGui.QColor(0, 0, 0, 100))#color, 1)
        #XXX XXX
        crosshair_overlay = CrosshairOverlay(pos = (x_coord, y_coord), index=z_coord, size=4, pen=pen, movable=False)

        overlay_item['crosshair'] = crosshair_overlay
        overlay_item['pen'] = pen
        set_item_group(overlay_item, patch['locationgroup'], False)
        if patch['locationgroup'] == "Candidate":
            seg_l.append(patch["id"])
        # if patch['locationgroup'] == 'Outside':
        #     color = QtGui.QColor(255, 0, 0, 150) # Qt.red
        # elif patch['locationgroup'] == 'Boundary':
        #     color = QtGui.QColor(255, 255, 0, 150) # QtCore.Qt.yellow
        # elif patch['locationgroup'] == 'Core':
        #     color = QtGui.QColor(0, 255, 0, 150) # QtCore.Qt.green
        # elif patch['locationgroup'] == 'Candidate':
        #     color = QtGui.QColor(0, 0, 255, 150) #QtCore.Qt.blue
        imageview.addItem(crosshair_overlay)
        overlay_list.append(overlay_item)
    print("\nUpdating image...")
    imageview.updateImage()
    print("Done")
    print(f"Candidates\n{seg_l}")
    # exit()
    return overlay_list, existing_threshold

def set_item_group(item, group, refresh=True):
    global region
    region["patches"][item["patch"]["id"]]["locationgroup"] = group # XXX
    item['patch']['locationgroup'] = group
    if group == 'Outside':
        item['pen'] = QtGui.QPen(QtGui.QColor(255, 0, 0, 150))#QtCore.Qt.red)
    elif group == 'Boundary':
        item['pen'] = QtGui.QPen(QtGui.QColor(255, 255, 0, 150))#QtCore.Qt.red)
        # item['pen'] = QtGui.QPen(QtCore.Qt.yellow)
    elif group == 'Core':
        item['pen'] = QtGui.QPen(QtGui.QColor(0, 255, 0, 150))#QtCore.Qt.red)
        # item['pen'] = QtGui.QPen(QtCore.Qt.green)
    elif group == 'Candidate':
        item['pen'] = QtGui.QPen(QtGui.QColor(0, 0, 255, 150))#QtCore.Qt.red)
        # item['pen'] = QtGui.QPen(QtCore.Qt.blue)
    item['crosshair'].setPen(item['pen'])
    if refresh:
        time_change("")

def time_change(event):
    for item in overlay_list:
        pen = None
        if item['crosshair'].get_index() == imv.currentIndex:
            pen = item['pen']
            # print("in this index {} patch with index {} with patchstep {} and pen {}".format(imv.currentIndex, item['patch']['id'], item['patch']['patchstep'], item['pen'].color()))
        item['crosshair'].setPen(pen)
    imv.updateImage()

#TODO Rework
def click(event):
    event.accept()
    pos = event.pos()
    mapped_pos = imv.getView().mapToView(pos)
    coords = (int(mapped_pos.x()),int(mapped_pos.y()),imv.currentIndex)
    patch_dim = region["partitioning"]["patch_size"][0] * region["thumbnails"]["downsampling"]# - region["partitioning"]["patch_overlap"] * region["thumbnails"]["downsampling"]
    # if  0 < coords[0] < image.shape[1] - 20 and 0 < coords[1] < image.shape[0] - 20:
    upper_x = image.shape[1]# * region["thumbnails"]["downsampling"]
    upper_y = image.shape[0]# * region["thumbnails"]["downsampling"]
    print(f"Click coords {coords[0]} < {upper_x} | {coords[1]} < {upper_y}")
    if  0 < coords[0] < upper_x and 0 < coords[1] < upper_y:
        patchstep = get_patchstep_by_coords(coords)
        print(f"Returns {patchstep}")
        overlay_item = get_patch_by_patchstep(patchstep)
        set_item_group(overlay_item, click_mode)
        pd(overlay_item["patch"])
    else:
        print("{}, {} is out of {}, {}".format(coords[1], coords[0], image.shape[0], image.shape[1]))

def move(event):
    global sidebar_label, sidebar_label_template
    pos = event#.pos()
    mapped_pos = imv.getView().mapToView(pos)
    coords = (int(mapped_pos.x()),int(mapped_pos.y()),imv.currentIndex)
    patch_dim = region["partitioning"]["patch_size"][0] * region["thumbnails"]["downsampling"] #- region["partitioning"]["patch_overlap"] * region["thumbnails"]["downsampling"]
    # if  0 < coords[0] < image.shape[1] - 20 and 0 < coords[1] < image.shape[0] - 20:
    # if  0 < coords[0] < image.shape[1] - patch_dim and 0 < coords[1] < image.shape[0] - patch_dim:
    upper_x = image.shape[1]# * region["thumbnails"]["downsampling"]
    upper_y = image.shape[0]# * region["thumbnails"]["downsampling"]
    if  0 < coords[0] < upper_x and 0 < coords[1] < upper_y:
        patchstep = get_patchstep_by_coords(coords)
        overlay_item = get_patch_by_patchstep(patchstep)
        sidebar_label = sidebar_label_template.format(overlay_item["patch"]["id"], overlay_item["patch"]["locationgroup"],overlay_item["patch"]["patchstep"])
    else:
        sidebar_label = sidebar_label_template.format("None","None","None")
    sidebar_label += f"\nPOS\t{str(pos)}\tCOR\t{str(coords)}"
    info_label.setText(sidebar_label)

def get_patch_by_patchstep(patchstep):
    x_step = region["partitioning"]["patches_per_dim"][1] * region["partitioning"]["patches_per_dim"][2]
    y_step = region["partitioning"]["patches_per_dim"][2]
    patch_id = 0
    # patch_id = 240 * patchstep[0] + 3* patchstep[1] + patchstep[2]
    patch_id = x_step * patchstep[0] + y_step * patchstep[1] + patchstep[2]

    ppd = region["partitioning"]["patches_per_dim"]
    patch_id = patchstep[0] * ppd[1] * ppd[2] + patchstep[1] * ppd[2] + patchstep[2]
    patch = overlay_list[0]
    if patch_id < len(overlay_list):
        patch = overlay_list[patch_id]
    else:
        print("Cant find patch at {} : {}".format(patchstep, patch_id))
    return patch

def get_patchstep_by_coords(coords):
    patch_dim = region["partitioning"]["patch_size"][0] * region["thumbnails"]["downsampling"]# - region["partitioning"]["patch_overlap"] * region["thumbnails"]["downsampling"]
    patch_dim = int(patch_dim)
    if region["partitioning"]["patch_overlap"] > 0:
        overlap_correction = region["partitioning"]["patch_overlap"] * region["thumbnails"]["downsampling"]
    x_coord = np.floor(coords[1] / (patch_dim - overlap_correction)).astype(np.int64)
    y_coord = np.floor(coords[0] / (patch_dim - overlap_correction)).astype(np.int64)
    # x_coord = np.floor(coords[1] / 19).astype(np.int)
    # y_coord = np.floor(coords[0] / 19).astype(np.int)
    z_coord = imv.currentIndex
    patchstep = [x_coord, y_coord, z_coord]
    return patchstep

# get the region of the image of the image view
def get_patch_region(overlay_item):
    patch_dim = region["partitioning"]["patch_size"][0] * region["thumbnails"]["downsampling"]# - region["partitioning"]["patch_overlap"] * 2 * region["thumbnails"]["downsampling"]
    patch_dim = int(patch_dim)
    # x_start = overlay_item['patch']['patchstep'][1] * 19
    # y_start = overlay_item['patch']['patchstep'][0] * 19
    x_start = overlay_item['patch']['patchstep'][1] * patch_dim
    y_start = overlay_item['patch']['patchstep'][0] * patch_dim
    z = overlay_item['patch']['patchstep'][2]
    # print("Getting info for {} ({}, {}, {})".format(overlay_item['patch']['id'], x_start, y_start, z))
    try:
        # image_subregion = image_transposed[z, y_start: y_start + 19, x_start:x_start + 19]
        image_subregion = image_transposed[z, y_start: y_start + patch_dim, x_start:x_start + patch_dim]
    except IndexError as ex:
        # image_subregion = np.zeros((19,19))
        image_subregion = np.zeros((patch_dim, patch_dim))
        print("IndexError for {} ({}, {}, {}) : {}".format(overlay_item['patch']['id'], x_start, y_start, z, ex))
    except TypeError as tex:
        image_subregion = np.zeros((0,0))
        print("TypeError for {} | {}".format(patch_dim, tex))

    return image_subregion

def auto_threshold(core=0.5,boundary=0.35):
    global_mean = np.mean(image)
    overlay_list_len = len(overlay_list)
    for i, item in enumerate(overlay_list):
        stb.showMessage("Generating overlay list for patch {} of {}".format(i, overlay_list_len))
        patch_region = get_patch_region(item)
        patch_mean = np.mean(patch_region)
        # print(f"{i} {core} {patch_mean/global_mean} {boundary}")
        if core < patch_mean / global_mean:
            set_item_group(item, "Core", False)
        elif boundary < patch_mean / global_mean < core:
            set_item_group(item, "Boundary", False)
        else:
            set_item_group(item, "Outside", False)
    print(global_mean)
    time_change("")

def change_slice(evt):
    imv.setCurrentIndex(evt)
    time_change(evt)

def set_outside_checked(evt):
    global click_mode
    click_mode = "Outside"
    print("Click mode {}".format(click_mode))

def set_boundary_checked(evt):
    global click_mode
    click_mode = "Boundary"
    print("Click mode {}".format(click_mode))

def set_core_checked(evt):
    global click_mode
    click_mode = "Core"
    print("Click mode {}".format(click_mode))

def set_candidate_checked(evt):
    global click_mode
    click_mode = "Candidate"
    print("Click mode {}".format(click_mode))

def proccess_event(evt):
    print(evt)

# All image related stuff
path_region = -1
region = -1
overlay_list = -1
image = -1

# Sidebar content
click_mode              = "Core"
sidebar_label_template  = "Patch\t\t:\t{}\nLocationgroup\t:\t{}\nPatchstep\t:\t{}"
sidebar_label = sidebar_label_template.format(0,0,0)

# Create core app
pg.setConfigOptions(imageAxisOrder='row-major')
app     = QtGui.QApplication([])
win     = QtGui.QMainWindow()
stb     = QtGui.QStatusBar(win)

# Create Toolbar
menu    = win.addToolBar("Main Toolbar")
open_f  = menu.addAction("&Open File")
save_as = menu.addAction("Save File &As")
save_f  = menu.addAction("&Save File")
menu.addSeparator()
undo    = menu.addAction("&Undo").setDisabled(True)
redo    = menu.addAction("&Redo").setDisabled(True)
menu.addSeparator()
thresh  = menu.addAction("Auto Threshold")
move_p  = menu.addAction("Move Candidates")
del_e   = menu.addAction("Delete Empty Patches")

imv     = pg.ImageView()

# Create buttons for segmentation mode
button_seg_layout   = QtGui.QWidget()
button_seg_layout.setLayout(QtGui.QVBoxLayout())

button_outside  = QtGui.QRadioButton("Outside")
button_boundary = QtGui.QRadioButton("Boundary")
button_inside   = QtGui.QRadioButton("Inside")
button_seg_can  = QtGui.QRadioButton("Segmentation candidate")

button_inside.setChecked(True)

# Upper divider
divider_up      = QtGui.QFrame()
divider_up.setFrameShape(QtGui.QFrame.HLine)
divider_up.setStyleSheet("background-color: #c0c0c0;")

# Slice selector
combo_slice     = pg.ComboBox(button_seg_layout, ["des","pa","cito"])

# Lower divider
divider_down    = QtGui.QFrame()
divider_down.setFrameShape(QtGui.QFrame.HLine)
divider_down.setStyleSheet("background-color: #c0c0c0;")

# Label
info_label      = QtGui.QLabel(sidebar_label.format("","",""))

# Side bar - patch group selection
button_seg_layout.layout().addWidget(button_outside)
button_seg_layout.layout().addWidget(button_boundary)
button_seg_layout.layout().addWidget(button_inside)
button_seg_layout.layout().addWidget(button_seg_can)

# Side bar - slice selector
button_seg_layout.layout().addWidget(divider_up)
button_seg_layout.layout().addWidget(combo_slice)
button_seg_layout.layout().addWidget(divider_down)

# Side bar - information panel
button_seg_layout.layout().addWidget(info_label)

# Create widget and set layout
widget = QtGui.QWidget()
widget.setLayout(QtGui.QGridLayout())

widget.layout().addWidget(imv, 0, 0, 6, 6)
widget.layout().addWidget(button_seg_layout, 0, 7, 1, 1)

if __name__ == '__main__':
    # path_region = "/media/ramial-maskari/WDElement3TB/Cell Annotation Project/Sensory/Sync/"
    win.show()
    win.setCentralWidget(widget)#imv)
    win.setStatusBar(stb)
    win.resize(QtGui.QDesktopWidget().availableGeometry(win).size()*0.8)

    # Click on a patch
    imv.scene.sigMouseClicked.connect(click)

    # Hover over a patch
    imv.scene.sigMouseMoved.connect(move)

    # Scroll through the image
    imv.sigTimeChanged.connect(time_change)
    imv.sigProcessingChanged.connect(proccess_event)

    # Click on toolbar items 
    open_f.triggered.connect(open_region)
    save_as.triggered.connect(save_region_as)
    save_f.triggered.connect(save_region)

    thresh.triggered.connect(lambda: auto_threshold(0.5,0.35))
    move_p.triggered.connect(move_patches)
    del_e.triggered.connect(delete_patches)

    # Toggle click mode
    button_outside.toggled.connect(set_outside_checked)
    button_boundary.toggled.connect(set_boundary_checked)
    button_inside.toggled.connect(set_core_checked)
    button_seg_can.toggled.connect(set_candidate_checked)

    path_region = QtGui.QFileDialog.getOpenFileName(None, 'Open file', '' , "Region files (*.pickledump)")[0]
    print(path_region)

    region, image = load_region(path_region)

    print("IMAGE SHAPE {}".format(image.shape))

    image_transposed = np.flip(np.rot90(np.swapaxes(image,0,2),axes=(1,2)),axis=1)
    imv.setImage(image_transposed)

    overlay_list, existing_threshold = generate_overlay_list(region, imv)
    if not existing_threshold:
        auto_threshold()

    # Set slices as combobox entries
    entry_list = ["Slice {}".format(i) for i in range(image_transposed.shape[0])]
    combo_slice.setItems(entry_list)
    combo_slice.currentIndexChanged.connect(change_slice)
    imv.timeLine.setMovable(False)

    change_slice(0)
    
    # for item in overlay_list:
    #     p = item["patch"]
    #     if p["id"] in [2095, 1169, 625, 1962, 1818, 1847, 1779, 1064, 1749, 570]:
    #         set_item_group(item, "Candidate", refresh=False)
        # ps_y = p["patchstep"][1]
        # if ps_y > 38:
        #     set_item_group(item, "Outside", refresh=False)
        
    time_change("")

    QtGui.QApplication.instance().exec_()
