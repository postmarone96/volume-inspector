#TODO fix imports
#TODO fix params

def time_change(event):
    if str(imv.currentIndex) in roi_list:
        r = roi_list[str(imv.currentIndex)]
        r.size = 0
        r.setState({"pos":(0,0),"size":0,"angle":0})
    imv.updateImage()

def click(event):
    event.accept()
    pos = event.pos()
    mapped_pos = imv.getView().mapToView(pos)
    coords = (int(mapped_pos.x()),int(mapped_pos.y()),imv.currentIndex)
    # print("coords\t\tx.shape\t\tpos")
    # print("[0]{}\t\t[0]{}\t\tx{}".format(coords[0], x.shape[0], pos.x()))
    # print("[1]{}\t\t[1]{}\t\ty{}".format(coords[1], x.shape[1], pos.y()))
    # coords: 
    #   - 0 = x 
    #   - 1 = y
    # x:
    #   - 0 = y
    #   - 1 = x


    if  0 < coords[0] < x.shape[1] - 20 and 0 < coords[1] < x.shape[0] - 20:
        # print(imv.getView().mapToView(pos))
        # print("X {} Y {} Z {}".format(coords[0],coords[1],coords[2]))
        patch = get_patch((coords[0], coords[1]))
        if patch == -1:
            pass
        else:
            pen.setWidthF(1)
            crosshair_pos_x = patch["patchstep"][1]*19 + 10# - np.floor(patch["patchstep"][1]/20)
            crosshair_pos_y = patch["patchstep"][0]*19 + 10# - np.floor(patch["patchstep"][0]/20)
            crosshair_pos = [crosshair_pos_x, crosshair_pos_y]

            r = MyCrosshairOverlay(pos=crosshair_pos, size=4, pen=pen, movable=False)
            roi_list[str(coords[2])] = r
            imv.addItem(r)
            imv.updateImage()
    else:
        print("{}, {} is out of {}, {}".format(coords[1], coords[0], x.shape[0], x.shape[1]))

