import numpy as np
import pandas as pd
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from matplotlib import pyplot as plt
import os
# from libtiff import TIFF
import re
from scipy.spatial import ConvexHull
import json
import vtk
import SimpleITK as sitk
# import alphashape
import scipy


def read_swc(path, mode="t", scale=None, comments=False):
    '''

    :param path:文件路径
    :param mode: "a"--Axon  "t"--total(all of it)  "d"--Dendrite
    :return: a list like [
                          [id type x y z radius pid],
                          [id type x y z radius pid],
                          ...
                          [id type x y z radius pid],
                                                     ]
    '''
    swc_matrix = []
    comments_list = []
    with open(path) as f:
        while True:
            linelist = []
            line = f.readline()
            if not line:
                break
            if line[0] == "#" or line[0] == 'i' or line[0] == "\n":
                comments_list.append(line)
                continue
            if line.count("\t") >= line.count(" "):
                str_split = "\t"
            elif line.count("\t") <= line.count(" "):
                str_split = " "
            elem = line.strip("\n").strip(" ").split(str_split)
            if mode == "t" or mode == 'T':
                pass
            elif mode == "a" or mode == "A":  # 1s 2a 3d
                if elem[1] not in ['1', '2']:
                    continue
            elif mode == "d" or mode == "D":
                if elem[1] not in ['1', '3', '4']:
                    continue
            for i in range(len(elem)):
                if i == 0 or i == 1 or i == 6:
                    linelist.append(int(elem[i]))
                elif i in [2, 3, 4]:
                    if scale is not None:
                        linelist.append(float(elem[i]) * scale)
                    else:
                        linelist.append(float(elem[i]))
                else:
                    linelist.append(float(elem[i]))

            swc_matrix.append(linelist)

    if mode == 'bifur':
        bifur_matrix = []
        for i in range(len(swc_matrix)):
            count = 0
            for j in range(len(swc_matrix)):
                if swc_matrix[i][0] == swc_matrix[j][6]:
                    count += 1
                if count >= 2:
                    bifur_matrix.append(swc_matrix[i])
                    break
        return bifur_matrix

    if comments:
        return swc_matrix, comments_list
    else:
        return swc_matrix


def Euc_calc(x, y, z, xx, yy, zz):
    '''
    计算(x,y,z) (xx,yy,zz)的欧氏距离
    :param x:
    :param y:
    :param z:
    :param xx:
    :param yy:
    :param zz:
    :return:
    '''
    return np.sqrt((x - xx) ** 2 + (y - yy) ** 2 + (z - zz) ** 2)


def get_soma(swc):
    '''
    获取soma点
    :param swc: read_swc函数的返回值
    :return:[id type x y z radius pid]
    '''
    soma = []
    for i in range(len(swc)):
        if swc[i][1] == 1 and swc[i][6] == -1:
            soma = swc[i]
            break
    if len(soma) == 0:
        for i in range(len(swc)):
            if swc[i][6] == -1:
                soma = swc[i]
                break
    if len(soma) == 0:
        for i in range(len(swc)):
            if swc[i][1] == 1:
                soma = swc[i]
                break
    if len(soma) == 0:
        print("no soma detected...")
    return soma


def get_bifurs(swc):
    '''
    获取bifurcation点
    :param swc: read_swc函数的返回值
    :return:
    '''
    soma = get_soma(swc)
    if not soma:
        return []
    bifur_nodes = []
    df = pd.DataFrame(swc)
    df_vc = df.iloc[:, 6].value_counts()
    for i in range(len(df_vc.index)):
        if df_vc.iloc[i] == 2 and df_vc.index[i] != soma[0] and df_vc.index[i] != -1:
            try:
                idx = np.array(swc)[:, 0].tolist().index((df_vc.index[i]))
            except:
                continue
            bifur_nodes.append(swc[idx])

    return bifur_nodes


def get_tips(swc):
    tips = []
    swc = np.asarray(swc)
    tips = swc[np.in1d(swc[:, 0], np.setdiff1d(swc[:, 0], swc[:, 6]))].tolist()
    # for i in range(swc.shape[0]):
    #     if swc[i][0] not in swc[:, 6].tolist():
    #         tips.append(swc[i].tolist())
    return tips


def sort_swc_index(src):
    dst = []
    NeuronHash = {}
    indexChildren = []
    for i in range(len(src)):
        NeuronHash[src[i][0]] = i
        indexChildren.append([])
    for i in range(len(src)):
        pid = src[i][6]
        idx = NeuronHash.get(pid)
        if idx is None: continue
        indexChildren[idx].append(i)

    LUT_n2newn = {}
    count = 1

    # DBS
    root = get_soma(src)
    root_xyz = np.array(root[2:5])
    root_n = root[0]
    root_idx = NeuronHash[root_n]

    LUT_n2newn[root_n] = count
    tmpnode = list(root)
    tmpnode[0] = count
    count += 1
    dst.append(tmpnode)

    bifurs = indexChildren[root_idx]
    while bifurs:
        cur_node_idx = bifurs.pop()
        LUT_n2newn[src[cur_node_idx][0]] = count
        tmpnode = list(src[cur_node_idx])
        tmpnode[0] = count
        count += 1
        dst.append(tmpnode)
        # print(cur_node_idx)
        cur_node_child_idx = indexChildren[cur_node_idx]
        # one child
        while len(cur_node_child_idx) == 1:
            next_node_idx = cur_node_child_idx[0]
            cur_node_idx = next_node_idx
            LUT_n2newn[src[cur_node_idx][0]] = count
            tmpnode = list(src[cur_node_idx])
            tmpnode[0] = count
            count += 1
            dst.append(tmpnode)

            cur_node_child_idx = indexChildren[cur_node_idx]

        # two children or no children
        if len(cur_node_child_idx) > 1 or len(cur_node_child_idx) == 0:
            bifurs.extend(cur_node_child_idx)

    for i in range(len(dst)):
        node = dst[i]
        mapped_pid = LUT_n2newn.get(node[6])
        if mapped_pid is None:
            mapped_pid = -1
        dst[i][6] = mapped_pid

    return dst


def save_swc(path, swc, comments='', eswc=False):
    '''
    save swc file
    :param path:save path
    :param swc:swc list
    :param comments:some remarks in line 2 in swc file
    :return:none
    '''
    if not path.endswith(".swc") and not path.endswith(".eswc"):
        if eswc:
            path += ".eswc"
        else:
            path += ".swc"
    with open(path, 'w') as f:
        f.writelines('#' + comments + "\n")
        f.writelines("#n,type,x,y,z,radius,parent\n")
        for node in swc:
            string = ""
            for i in range(len(node)):
                item = node[i]
                if i in [0, 1, 6]:
                    item = int(item)
                elif i in [2, 3, 4]:
                    item = np.round(item, 3)
                string = string + str(item) + " "
                if not eswc:
                    if i == 6:
                        break
            string = string.strip(" ")
            string += "\n"
            f.writelines(string)


def nodes_to_soma(nodes, swc):
    soma = get_soma(swc)
    idlist = np.array(swc)[:, 0].tolist()
    waylists = []
    for node in nodes:
        waylist = []
        cur_node = node
        waylist.append(cur_node)
        while True:
            pid = cur_node[6]
            idx = idlist.index(pid)
            next_node = swc[idx]
            waylist.append(next_node)
            if next_node == soma:
                break
            cur_node = next_node
        waylists.append(waylist)
    return waylists



def Vaa3d_global_feature(swcpath):
    try:
        col_name = []
        vaa3d_path = "D:/Vaa3D_V3.601_Windows_MSVC_64bit/"
        a = os.popen("{}vaa3d_msvc.exe /x {}"
                     "plugins/neuron_utilities/global_neuron_feature/global_neuron_feature.dll "
                     "/f compute_feature "
                     "/i \"{}\"".format(vaa3d_path, vaa3d_path, swcpath.replace("\\", "/")))
        emm = a.readlines()
        count = 0
        temp = []
        value = []
        for i in range(len(emm)):
            if count == 1:
                temp.append(emm[i])
            if emm[i] == "compute Feature  \n":
                count = 1
        temp.pop(-1)
        for i in range(len(temp)):
            spli = temp[i].split(":")
            cn = spli[0]
            if cn == "Number of Bifurcatons":
                cn = "Number of Bifurcations"
            col_name.append(cn)
            aa = re.search("\d+.\d+e\+\d+", spli[1])
            bb = re.search("\d+e\+\d+", spli[1])
            cc = re.search("\d+\.\d+", spli[1])
            dd = re.search("\d+", spli[1])
            if aa is not None and bb is not None and cc is not None and dd is not None:
                value.append(float(aa.group()))
            elif aa is None and bb is not None and cc is None:
                value.append(float(bb.group()))
            elif aa is None and bb is None and cc is not None:
                value.append(float(cc.group()))
            else:
                value.append(float(dd.group()))
        if len(value) <= 22:
            return [None] * 28, col_name
        else:
            return value, col_name
    except:
        return [None] * 28, [None] * 28


def get_path(path):
    path_list = []
    iter_f = os.walk(path)
    root_path = ""
    file_path_list = ""
    try:
        while True:
            cur = next(iter_f)
            root_path = cur[0]
            file_path_list = cur[-1]
            if not file_path_list:
                continue
            for i in range(len(file_path_list)):
                path = os.path.join(root_path, file_path_list[i])
                path_list.append(path)

    except StopIteration:
        pass

    return path_list


def detail_to_rough_region(df, target=None, color=False):
    with open(r"E:\ZhixiYun\Projects\Neuron_Morphology_Table\Scripts\neuron_code\neuron\tree.json") as f:
        tree = json.load(f)

    if target is None:
        with open(r"E:\ZhixiYun\Projects\Neuron_Morphology_Table\Tables\acronym_list_1.txt") as f:
            rough_region = [x.strip("\n") for x in f.readlines()]

        # rough_region.remove("Isocortex")
        rough_region.remove("fiber tracts")
        # rough_region = ["MO"]+rough_region
    else:
        rough_region = target

    colordict = {}

    # 获取rough脑区的id
    rr_id = {}
    for item in tree:
        if item["acronym"] in rough_region:
            rr_id[item["id"]] = item["acronym"]
            colordict[item["acronym"]] = np.array(item["rgb_triplet"]) / 255.0

    mapdict = {}
    for ct in df["CellType"].value_counts().index:
        for item in tree:
            if item["acronym"] == ct:
                cur_path = item["structure_id_path"]
                for rrid in rr_id.keys():
                    if rrid in cur_path:
                        mapdict[ct] = rr_id[rrid]
                        break
                    else:
                        mapdict[ct] = "unknown"
    if not color:
        return mapdict
    else:
        return mapdict, colordict


def recon_mesh(img, fname_output, fname_tmp='./tmp.mhd', color=None):
    mask = img.copy()
    mask[img > 0] = 255
    snew = sitk.GetImageFromArray(mask)
    sitk.WriteImage(snew, fname_tmp)

    reader = vtk.vtkMetaImageReader()
    reader.SetFileName(fname_tmp)
    reader.Update()

    iso = vtk.vtkMarchingCubes()
    iso.SetInputConnection(reader.GetOutputPort())
    iso.SetValue(0, 1)
    iso.ComputeNormalsOff()
    iso.Update()
    mesh = iso.GetOutput()

    smoother = vtk.vtkSmoothPolyDataFilter()
    smoother.SetInputData(mesh)
    smoother.SetNumberOfIterations(500)
    smoother.SetRelaxationFactor(0.1)
    smoother.FeatureEdgeSmoothingOff()
    smoother.BoundarySmoothingOn()
    smoother.Update()
    mesh = smoother.GetOutput()

    decimate = vtk.vtkDecimatePro()
    decimate.SetInputData(mesh)
    decimate.SetTargetReduction(0.95)
    decimate.PreserveTopologyOn()
    decimate.Update()
    mesh = decimate.GetOutput()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(mesh)
    normals.SetFeatureAngle(100.0)
    normals.ComputePointNormalsOn()
    normals.SplittingOn()
    normals.Update()
    mesh = normals.GetOutput()

    if color is not None:
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        for _ in range(mesh.GetNumberOfPoints()):
            colors.InsertNextTypedTuple(color)
        mesh.GetPointData().SetScalars(colors)

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileVersion(42)
    writer.SetFileName(fname_output)
    writer.SetInputData(mesh)
    writer.Update()


def calc_projection(swc, anno, scale=1 / 25):
    def calc_2nodes_dist(node1, node2):
        dist = np.linalg.norm(np.array(node1)[2:5] - np.array(node2)[2:5])
        return dist

    mat = MouseAnatomyTree()
    proj_dict = {0: 0.0}
    for i in list(mat.lutidtoname.keys()):
        proj_dict[i] = 0.0
    # proj_dict = {0: 0.0}
    # for i in np.unique(anno):
    #     proj_dict[i] = 0.0

    NeuronHash = {}
    indexChildren = []
    for i in range(len(swc)):
        NeuronHash[swc[i][0]] = i
        indexChildren.append([])
    for i in range(len(swc)):
        pid = swc[i][6]
        idx = NeuronHash.get(pid)
        if idx is None: continue
        indexChildren[idx].append(i)

    # DBS
    root = get_soma(swc)
    if len(root) == 0: return proj_dict
    root_n = root[0]
    root_idx = NeuronHash[root_n]

    bifurs_chs = indexChildren[root_idx]
    cur_idx = root_idx
    cur_node = root
    while bifurs_chs:
        next_idx = bifurs_chs.pop()
        next_node = swc[next_idx]
        cur_idx = NeuronHash[next_node[6]]
        cur_node = swc[cur_idx]
        tmpdist = calc_2nodes_dist(cur_node, next_node)
        # print(f'tmpdist:{tmpdist}',cur_idx,next_idx,'outer')
        mid_pos_25um = np.round((np.array(cur_node)[2:5] + np.array(next_node)[2:5]) / 2.0 * scale).astype(int)

        if ((mid_pos_25um[0] >= 0) & (mid_pos_25um[0] < anno.shape[0]) &
                (mid_pos_25um[1] >= 0) & (mid_pos_25um[1] < anno.shape[1]) &
                (mid_pos_25um[2] >= 0) & (mid_pos_25um[2] < anno.shape[2])):
            key = anno[mid_pos_25um[0], mid_pos_25um[1], mid_pos_25um[2]]
            proj_dict[key] += tmpdist
        else:
            proj_dict[0] += tmpdist

        cur_idx = next_idx
        cur_node = next_node
        cur_child_idx_list = indexChildren[cur_idx]

        while len(cur_child_idx_list) == 1:
            next_idx = cur_child_idx_list[0]
            next_node = swc[next_idx]
            tmpdist = calc_2nodes_dist(cur_node, next_node)
            # print(f'tmpdist:{tmpdist}',cur_idx,next_idx,'inner')
            mid_pos_25um = np.round((np.array(cur_node)[2:5] + np.array(next_node)[2:5]) / 2.0 * scale).astype(int)

            if ((mid_pos_25um[0] >= 0) & (mid_pos_25um[0] < anno.shape[0]) &
                    (mid_pos_25um[1] >= 0) & (mid_pos_25um[1] < anno.shape[1]) &
                    (mid_pos_25um[2] >= 0) & (mid_pos_25um[2] < anno.shape[2])):
                key = anno[mid_pos_25um[0], mid_pos_25um[1], mid_pos_25um[2]]
                proj_dict[key] += tmpdist
            else:
                proj_dict[0] += tmpdist

            cur_idx = next_idx
            cur_node = next_node
            cur_child_idx_list = indexChildren[cur_idx]

        if len(cur_child_idx_list) > 1:
            bifurs_chs.extend(cur_child_idx_list)
        elif len(cur_child_idx_list) == 0:
            continue

    return proj_dict


def calc_voxel_occupying(swc, anno, scale=1 / 25):
    # def calc_2nodes_dist(node1, node2):
    #     dist = np.linalg.norm(np.array(node1)[2:5] - np.array(node2)[2:5])
    #     return dist
    def calc_2nodes_interp(node1, node2):
        nodelist = [np.array(node1)[2:5]]
        weightlist = [1]
        dist = np.linalg.norm(np.array(node1)[2:5] - np.array(node2)[2:5])
        normv = (np.array(node2)[2:5] - np.array(node1)[2:5]) / dist
        if dist // 1 >= 1:
            for tmpi in range(int(dist // 1)):
                nodelist.append(np.array(node1)[2:5] + (tmpi + 1) * normv)
        nodelist.append(np.array(node2)[2:5])
        return nodelist

    # mat = MouseAnatomyTree()
    # proj_dict = {0: 0.0}
    # for i in list(mat.lutidtoname.keys()):
    #     proj_dict[i] = 0.0
    # proj_dict = {0: 0.0}
    # for i in np.unique(anno):
    #     proj_dict[i] = 0.0
    allvoxels = []
    voxelfill = np.zeros(anno.shape, dtype=np.uint8)

    NeuronHash = {}
    indexChildren = []
    for i in range(len(swc)):
        NeuronHash[swc[i][0]] = i
        indexChildren.append([])
    for i in range(len(swc)):
        pid = swc[i][6]
        idx = NeuronHash.get(pid)
        if idx is None: continue
        indexChildren[idx].append(i)

    # DBS
    root = get_soma(swc)
    if len(root) == 0: return voxelfill
    root_n = root[0]
    root_idx = NeuronHash[root_n]

    bifurs_chs = indexChildren[root_idx]
    cur_idx = root_idx
    cur_node = root
    while bifurs_chs:
        next_idx = bifurs_chs.pop()
        next_node = swc[next_idx]
        cur_idx = NeuronHash[next_node[6]]
        cur_node = swc[cur_idx]
        nodelist = calc_2nodes_interp(cur_node, next_node)
        allvoxels.extend(nodelist)

        cur_idx = next_idx
        cur_node = next_node
        cur_child_idx_list = indexChildren[cur_idx]

        while len(cur_child_idx_list) == 1:
            next_idx = cur_child_idx_list[0]
            next_node = swc[next_idx]
            nodelist = calc_2nodes_interp(cur_node, next_node)
            allvoxels.extend(nodelist)

            cur_idx = next_idx
            cur_node = next_node
            cur_child_idx_list = indexChildren[cur_idx]

        if len(cur_child_idx_list) > 1:
            bifurs_chs.extend(cur_child_idx_list)
        elif len(cur_child_idx_list) == 0:
            continue

    allvoxels = np.asarray(allvoxels)
    if allvoxels.size == 0: return voxelfill
    allvoxels = np.round(allvoxels).astype(int)
    allvoxels = allvoxels[(allvoxels[:, 0] >= 0) & (allvoxels[:, 0] < anno.shape[0]) &
                          (allvoxels[:, 1] >= 0) & (allvoxels[:, 1] < anno.shape[1]) &
                          (allvoxels[:, 2] >= 0) & (allvoxels[:, 2] < anno.shape[2])]
    voxelfill[allvoxels[:, 0], allvoxels[:, 1], allvoxels[:, 2]] = 1

    return voxelfill


def genObj(input_p, ntype=None, scale=1.0, output_p='./tmp/tmp.obj'):
    swcraw = read_swc(input_p, scale=scale)
    swc = []
    if ntype is None:
        swc = list(swcraw)
        pass
    else:
        for node in swcraw:
            if node[1] in ntype:
                swc.append(node)
    objLines = []
    NeuronHash = {}
    for i in range(0, len(swc)):
        NeuronHash[swc[i][0]] = i
        line = 'v ' + str(swc[i][2]) + ' ' + str(swc[i][3]) + ' ' + str(swc[i][4]) + '\n'
        objLines.append(line)
    for i in range(0, len(swc)):
        p = swc[i][6]
        if p not in NeuronHash.keys():
            continue
        line = 'l ' + str(i + 1) + ' ' + str(NeuronHash[p] + 1) + '\n'
        objLines.append(line)

    with open(output_p, 'w') as f:
        f.writelines(objLines)

    

class SWC_Features:
    '''
    some new features. (see feature_name)
    (swc need resample)
    '''

    def __init__(self, swc, swcname, swc_reg=None):
        self.feature_name = ["Average Euclidean Distance", "25% Euclidean Distance", "50% Euclidean Distance",
                             "75% Euclidean Distance", "Average Path Distance", "25% Path Distance",
                             "50% Path Distance", "75% Path Distance", "Center Shift", "Relative Center Shift"]

        self.features = []
        self.swc = swc
        self.swcname = swcname
        self.soma = get_soma(swc)
        self.tips = get_tips(swc)
        self.swc_reg = swc_reg
        # self.bifurs = get_bifurs(swc)

        if not self.soma:
            print(swcname, "no soma detected...")
            self.features = []
        else:
            self.features += self.Euc_Dis(swc)
            self.features += self.Pat_Dis(swc)
            cs = self.center_shift(swc)
            ave_euc_dis = self.features[0]
            self.features += [cs, cs / ave_euc_dis]
            if self.swc_reg is not None:
                self.features += self.size_related_features(swc_reg)
                self.features += self.xyz_approximate(swc_reg)
                self.feature_name += [
                    "Area", 'Volume', "2D Density", "3D Density", "Width", "Height", "Depth", "Width_95ci",
                    "Height_95ci",
                    "Depth_95ci", "Slimness", "Flatness", "Slimness_95ci", "Flatness_95ci"]

    def Euc_Dis(self, swc):
        dislist = []
        soma = self.soma
        for node in swc:
            cur_dis = Euc_calc(soma[2], soma[3], soma[4], node[2], node[3], node[4])
            dislist.append(cur_dis)
        length = len(dislist)
        if length == 0:
            return [None] * 4
        euc_dis_ave = np.mean(dislist)
        dislist.sort()
        euc_dis_25 = dislist[int(np.floor(length * 1 / 4))]
        euc_dis_50 = np.median(dislist)
        euc_dis_75 = dislist[int(np.floor(length * 3 / 4))]
        return [euc_dis_ave, euc_dis_25, euc_dis_50, euc_dis_75]

    def Pat_Dis(self, swc):
        patlist = []
        soma = self.soma
        id_pathdist = {}
        idlist = np.array(swc)[:, 0].tolist()
        pidlist = np.array(swc)[:, 6].tolist()
        if soma[0] not in pidlist:
            # 此时说明没有连接到soma的通路，寻找最接近soma的root
            maxdist = 1000000
            for node in swc:
                if node == self.soma:
                    continue
                if node[6] not in idlist:
                    cur_dist = Euc_calc(self.soma[2], self.soma[3], self.soma[4], node[2], node[3], node[4])
                    if cur_dist < maxdist:
                        maxdist = cur_dist
                        soma = node

        if self.tips:
            nodes = self.tips
        else:
            nodes = swc
        for node in nodes:
            if node == soma or node == self.soma:
                continue
            cur_node = node
            cur_pathdist = 0
            passbynode = {}
            while True:
                pid = cur_node[6]
                if pid not in idlist:
                    break
                idx = idlist.index(pid)
                new_node = swc[idx]
                delta_pathdist = Euc_calc(cur_node[2], cur_node[3], cur_node[4], new_node[2], new_node[3], new_node[4])
                cur_pathdist += delta_pathdist
                if passbynode.keys():
                    passbynode = dict(
                        zip(list(passbynode.keys()), (np.array(list(passbynode.values())) + delta_pathdist).tolist()))
                if new_node == soma:
                    id_pathdist[node[0]] = cur_pathdist
                    id_pathdist.update(passbynode)
                    break
                elif new_node[0] in id_pathdist.keys():
                    id_pathdist[node[0]] = cur_pathdist + id_pathdist[new_node[0]]
                    if passbynode.keys():
                        passbynode = dict(
                            zip(list(passbynode.keys()),
                                (np.array(list(passbynode.values())) + id_pathdist[new_node[0]]).tolist()))
                    id_pathdist.update(passbynode)
                    break
                else:
                    cur_node = new_node
                    passbynode[new_node[0]] = 0

        pathdislist = list(id_pathdist.values())
        length = len(pathdislist)
        if length == 0:
            return [None] * 4
        path_dis_ave = np.mean(pathdislist)
        pathdislist.sort()
        path_dis_25 = pathdislist[int(np.floor(length * 1 / 4))]
        path_dis_50 = np.median(pathdislist)
        path_dis_75 = pathdislist[int(np.floor(length * 3 / 4))]
        return [path_dis_ave, path_dis_25, path_dis_50, path_dis_75]

    def center_shift(self, swc):
        soma = self.soma
        swc_ar = np.array(swc)
        centroid = np.mean(swc_ar[:, 2:5], axis=0)
        return Euc_calc(soma[2], soma[3], soma[4], centroid[0], centroid[1], centroid[2])

    def pixel_voxel_calc(self, swc_xyz):
        swcxyz = np.array(swc_xyz)
        x = np.round(swcxyz[:, 0])
        y = np.round(swcxyz[:, 1])
        z = np.round(swcxyz[:, 2])
        pixels = list(set(list(zip(z, y))))  # 投射到zy平面算pixel z是主方向 且去除了冗余pixel
        voxels = list(set(list(zip(x, y, z))))
        num_pixels = len(pixels)
        num_voxels = len(voxels)

        return num_pixels, num_voxels

    def size_related_features(self, swc):
        num_nodes = len(swc)
        if num_nodes < 3:
            return [None] * 4
        swc_zy = np.array(swc)[:, 3:5]
        swc_xyz = np.array(swc)[:, 2:5]
        CH2D = ConvexHull(swc_zy)
        CH3D = ConvexHull(swc_xyz)
        # CH2D.area  # 2D情况下这个是周长×
        # CH2D.volume  # 2D情况下这个是面积√
        # CH3D.area    # 3D情况下这个是表面积×
        # CH3D.volume  # 3D情况下这个是体积√
        area = CH2D.volume
        volume = CH3D.volume
        # interpolation of swc so that each pixel/voxel can be occupied on all pathway
        swc_xyz_new = list(swc_xyz)
        swc_arr = np.array(swc)
        idlist = list(swc_arr[:, 0])
        for node in swc:
            pid = node[6]
            x1, y1, z1 = node[2:5]
            if pid not in idlist:
                continue
            else:
                cur_id = idlist.index(pid)
                x2, y2, z2 = swc_xyz[cur_id]
                count = int(Euc_calc(x1, y1, z1, x2, y2, z2) // 1)
                if count != 0:
                    tmp = [[x1 + 1 * x, y1 + 1 * x, z1 + 1 * x] for x in
                           range(1, count + 1)]
                    swc_xyz_new.extend(tmp)

        num_pixels, num_voxels = self.pixel_voxel_calc(swc_xyz_new)
        density_2d = num_pixels / area
        density_3d = num_voxels / volume

        return [area, volume, density_2d, density_3d]

    def xyz_approximate(self, swc):
        '''
        shape related
        :param swc:
        :return:
        '''
        if not swc:
            return [None] * 10
        swcxyz = np.array(swc)[:, 2:5]
        x = swcxyz[:, 0]
        y = swcxyz[:, 1]
        z = swcxyz[:, 2]
        width = np.max(y) - np.min(y)  # y  zyx-registration   height=z-z' width=y-y' depth=x-x'
        height = np.max(z) - np.min(z)  # z
        depth = np.max(x) - np.min(x)  # x
        # confidence interval 95%
        width_95ci = abs(np.percentile(y, 97.5) - np.percentile(y, 2.5))
        height_95ci = abs(np.percentile(z, 97.5) - np.percentile(z, 2.5))
        depth_95ci = abs(np.percentile(x, 97.5) - np.percentile(x, 2.5))

        slimness = width / height  # slimness = width/height
        flatness = height / depth  # flatness = height/depth
        slimness_95ci = width_95ci / height_95ci
        flatness_95ci = height_95ci / depth_95ci

        return [width, height, depth, width_95ci, height_95ci, depth_95ci, slimness, flatness, slimness_95ci,
                flatness_95ci]

