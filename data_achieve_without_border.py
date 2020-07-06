import pymysql
import os
import shutil
import time
from PIL import Image, ImageOps
import numpy as np
import json
import math
from progressbar import *
import multiprocessing

'''
ocr_data achieve process
data source:mysql
for a keras yolov3 train
by liningbo
此版本边界处目标涂黑

'''

def cal_rotate_angle(x0, y0, x3, y3):

    """
    calulate rotate angle from new position
    x0, y0: position 0 rotated
    x3, y3: position 3 rotated
    """
    try:
        angle = -math.atan((x0-x3)*1.0/(y0-y3))*180/math.pi
        return angle
    except Exception as e:
        return 0


def rotate_points(points, rotate_degree, w, h):
    for point in points:
        x = point['x']
        y = point['y']
        x_shift = x - w/2.0
        y_shift = y - h/2.0
        rotate_rad = rotate_degree/180.0*math.pi
        nx = x_shift*math.cos(rotate_rad)+y_shift*math.sin(rotate_rad)+w/2.0
        ny = -x_shift*math.sin(rotate_rad)+y_shift*math.cos(rotate_rad)+h/2.0
        point['x']=nx
        point['y']=ny


def bounding_points(points):
    x_min = 10000000
    y_min = 10000000
    x_max = -10000000
    y_max = -10000000
    for point in points:
        x = point["x"]
        y = point["y"]
        x_min = min(x_min, x)
        x_max = max(x_max, x)
        y_min = min(y_min, y)
        y_max = max(y_max, y)
    points[0]["x"] = x_min
    points[0]["y"] = y_min
    points[1]["x"] = x_max
    points[1]["y"] = y_min
    points[2]["x"] = x_max
    points[2]["y"] = y_max
    points[3]["x"] = x_min
    points[3]["y"] = y_max



def divide_image(label_divided, image_label_set, image_with_border, image_ori_id, w_subnum, h_subnum, min_overlop, input_shape):

    """
    divide image into subimage
    image_with_border: image with border for inviding
    h_subnum: subimage rows num
    w_subnum: subimage cols num
    min_overlop: window sliding overlop
    input_shape: input shape of yolov3 model
    """
    area_thr = 0.75
    for sub_i in range(h_subnum):
        for sub_j in range(w_subnum):
            box = (sub_j*(input_shape[0]-min_overlop), sub_i*(input_shape[1]-min_overlop), sub_j*(input_shape[0]-min_overlop)+input_shape[0], sub_i*(input_shape[1]-min_overlop)+input_shape[1])
            sub_image = image_with_border.crop(box)
            for polygon_id in image_label_set:
                label = image_label_set[polygon_id]
                box_image = (sub_j*(input_shape[0]-min_overlop), sub_i*(input_shape[1]-min_overlop), sub_j*(input_shape[0]-min_overlop)+input_shape[0], sub_i*(input_shape[1]-min_overlop)+input_shape[1])
                rect_image = {'x':box_image[0], 'y':box_image[1], 'x_':box_image[2], 'y_':box_image[3], 'w':abs(box_image[0]-box_image[2]), 'h':abs(box_image[1]-box_image[3]), 'area':abs(box_image[0]-box_image[2])*abs(box_image[1]-box_image[3])}
                rect_info = get_rect_info(label[0], label[2])
                intersection = cal_intersection_ratio(rect_image, rect_info)
                if intersection["area"]>1:
                    (l, t, r, b) = intersection["inter_rect"]
                    l -= rect_image['x']
                    r -= rect_image['x']
                    t -= rect_image['y']
                    b -= rect_image['y']
                    if intersection["ratio_b"]>area_thr:
                        # map sub_label into rect in the subimage
                        key = "%d_%d_%d_%d" % (image_ori_id, sub_i, sub_j, min_overlop)
                        label_divided[key].append((l, t, r, b))
                    else:
                        sub_image.paste((0, 0, 0), (l, t, r, b))
            file_name = "./data/image_split/%d_%d_%d_%d.png" % (image_ori_id, sub_i, sub_j, min_overlop)
            sub_image.save(file_name)
    

def get_rect_info(point_a, point_b):
    try:
        x = int(min(point_a['x'], point_b['x']))
        x_ = int(max(point_a['x'], point_b['x']))
        y = int(min(point_a['y'], point_b['y']))
        y_ = int(max(point_a['y'], point_b['y']))
        w = int(abs(x_-x))
        h = int(abs(y_-y))
        area = int(w*h)
        return {'x':x, 'y':y, 'x_':x_, 'y_':y_, 'w':w, 'h':h, 'area':area}
    except Exception as e:
        print(e)


def cal_intersection_ratio(rect_a, rect_b):
    try:
        dist_x = max(rect_a['x_'], rect_b['x_']) - min(rect_a['x'], rect_b['x'])
        dist_y = max(rect_a['y_'], rect_b['y_']) - min(rect_a['y'], rect_b['y'])
        w_intersection = max(rect_a['w']+rect_b['w']-dist_x , 0)
        h_intersection = max(rect_a['h']+rect_b['h']-dist_y , 0)
        area_intersection = w_intersection * h_intersection
        intersection_ratio_a = area_intersection / (rect_a['area']*1.0)
        intersection_ratio_b = area_intersection / (rect_b['area']*1.0)
        inter_rect = None
        if area_intersection > 0:
            l = max(rect_a['x'], rect_b['x'])
            r = min(rect_a['x_'], rect_b['x_'])
            t = max(rect_a['y'], rect_b['y'])
            b = min(rect_a['y_'], rect_b['y_'])
            inter_rect = (l, t, r, b)
        return {'ratio_a':intersection_ratio_a, 'ratio_b':intersection_ratio_b, 'area':area_intersection, 'inter_rect':inter_rect}
    except Exception as e:
        return {'ratio_a':0, 'ratio_b':0, 'area':0, 'inter_rect':None}



def data_achieve(hostid, host_user, database, host_passwd, labels_contributor_set, input_shape, min_overlop, padding):

    """
    achieve pdf images and its labels then convert them into train data for the yolov3 model
    hostid: address of the mysql host
    host_user: user of mysql
    host_passwd: password of the appointed user 
    labels_contributor_set: user_names set of the contributors
    min_overlop: minimum overlop of the slid windows in a pdf image
    padding: padding mode
    """
    connection = pymysql.connect(host=hostid, user=host_user, password=host_passwd, charset="utf8", use_unicode=True)
    db_cursor = connection.cursor()
    connection.select_db(database)
    # achieve all labels and its image
    sql_polygon_achieve = 'select id, pdfImage_id, polygon from ocr_ocrlabelingpolygon where create_user_id in (%s)' % ','.join(['%s'] * len(labels_contributor_set)) 
    db_cursor.execute(sql_polygon_achieve, labels_contributor_set)
    query_polygon = db_cursor.fetchall()
    label_set_ori = dict()
    for polygon in query_polygon:
        (polygon_id, image_id, polygon_data)=polygon
        #print(polygon_id, image_id)
        try:
            label_set_ori[image_id][polygon_id] = json.loads(polygon_data.decode('utf-8'))
        except Exception as e:
            label_set_ori[image_id] = {polygon_id: json.loads(polygon_data.decode('utf-8'))}

    image_ori_dir = "./data/image_ori"
    if not os.path.exists(image_ori_dir):
        os.mkdir(image_ori_dir)
        print("original image direction:%s created." % image_ori_dir)
    else:
        shutil.rmtree(image_ori_dir)
        print("original image direction:%s removed." % image_ori_dir)
        os.mkdir(image_ori_dir)
        print("original image direction:%s created." % image_ori_dir)
    download_size = 0
    time_start = time.time()
    
    widgets = ['downloading: ', Percentage(), ' ', Bar('#'), ' ', Timer(), ' ']
    bar_download = progressbar.ProgressBar(widgets=widgets, maxval=len(label_set_ori))
    bar_download.start()
    file_count = 0
    for image_id in label_set_ori:
        # achieve the image_ori
        sql_image_achieve = "select data_byte from ocr_pdfimage where id=%s and id>533 and id<552" %  image_id 
        db_cursor.execute(sql_image_achieve)
        image = db_cursor.fetchone()
        file_name = "./data/image_ori/%s.png" % image_id
        if image==None:
            continue
        image_size = len(image[0])
        download_size += image_size
        with open(file_name, "wb") as f:
            f.write(image[0])
        file_count += 1
        bar_download.update(file_count)
            
    db_cursor.close()
    connection.close()
    duration = time.time() - time_start
    download_speed = download_size/1024.0/1024/duration
    print("druation: %.2f s" % duration)
    print("data amount: %.2f MB" % (download_size/1024.0/1024))
    print("download speed: %.2f MB/s, theoretical network speed: %s MB/s." % (download_speed, 125))

    image_split_dir = "./data/image_split"
    if not os.path.exists(image_split_dir):
        os.mkdir(image_split_dir)
        print("image splited direction:%s created." % image_split_dir)
    else:
        shutil.rmtree(image_split_dir)
        print("image splited direction:%s removed." % image_split_dir)
        os.mkdir(image_split_dir)
        print("image splited direction:%s created." % image_split_dir)

    image_ori = os.listdir("./data/image_ori")
    widgets = ['dividing: ', Percentage(), ' ', Bar('#'), ' ', Timer(), ' ']
    bar_divide = progressbar.ProgressBar(widgets=widgets, maxval=len(image_ori))
    bar_divide.start()

    label_divided = dict()
    # divide_process_pool = multiprocessing.Pool(processes=6, maxtasksperchild=1)
    file_count = 0
    for image_name in image_ori:
        image_path = "./data/image_ori/"+image_name
        image = Image.open(image_path)
        base_name = image_name.split('.')[0]
        image_ori_id = int(base_name)
        image_label_set = label_set_ori[image_ori_id]
        angle_list = []
        for polygon_id in image_label_set:
            label = image_label_set[polygon_id]
            # angle to be rotated
            area = abs(max(label[0]["y"], label[1]["y"])-min(label[2]["y"],label[3]["y"]))*abs(max(label[1]["x"],label[2]["x"])-min(label[0]["x"],label[3]["x"]))
            if area>32:
                angle = cal_rotate_angle(label[0]["x"], label[0]["y"], label[3]["x"], label[3]["y"])
                angle_list.append(angle)
        try:
            if len(angle_list) != 0:
                angle_median = np.median(angle_list)
            else:
                angle_median = 0
        except Exception as e:
            angle_median = 0
        
        #image_gray = image.convert("F")/255
        image_gray = image
        (width_ori, height_ori) = image.size
        
        if abs(angle_median)<0.0001:
            image_rotate = image_gray
        else:
            image_rotate = image_gray.rotate(angle_median)
        (w, h) = image_rotate.size
        # stepx
        w_subnum = max((w - min_overlop -1),0) // (input_shape[0]-min_overlop)+1
        # stepy
        h_subnum = max((h - min_overlop -1),0) // (input_shape[1]-min_overlop)+1
        w_new = w_subnum*(input_shape[0]-min_overlop)+min_overlop
        h_new = h_subnum*(input_shape[1]-min_overlop)+min_overlop
        w_extend = w_new - w
        h_extend = h_new - h
        w_l_extend = w_extend // 2
        w_r_extend = w_extend -w_l_extend
        h_t_extend = h_extend // 2
        h_b_extend = h_extend -h_t_extend
        # expend
        image_with_border = ImageOps.expand(image_rotate, border=(w_l_extend, h_t_extend, w_r_extend, h_b_extend) ,fill=0)

        # initialize the label divided
        for sub_i in range(h_subnum):
            for sub_j in range(w_subnum):
                key = "%d_%d_%d_%d" % (image_ori_id, sub_i, sub_j, min_overlop)
                label_divided[key]=[]


        #label processing
        for polygon_id in image_label_set:
            label = image_label_set[polygon_id]

            # rotate
            if abs(angle_median)>0.0001:
                rotate_points(label, angle_median, width_ori, height_ori)
            angle_to_rotate = cal_rotate_angle(label[0]["x"], label[0]["y"], label[3]["x"], label[3]["y"])
            # bounding box
            if abs(angle_to_rotate) >0.0001:
                bounding_points(label)
            # position mapping
            for point in label:
                point["x"] += w_l_extend
                point["y"] += h_t_extend

        # divide labels
        '''divide_process_pool.apply_async(
                divide_image, args=(
                    label_divided,
                    image_label_set,
                    image_with_border,
                    image_ori_id,
                    w_subnum,
                    h_subnum,
                    min_overlop,
                    input_shape
                )
            )
        '''
        divide_image(
                label_divided,
                image_label_set,
                image_with_border,
                image_ori_id,
                w_subnum,
                h_subnum,
                min_overlop,
                input_shape
                )

        
        file_count += 1
        bar_divide.update(file_count)
    bar_divide.finish()
    #print("waiting for the image process...")
    # divide_process_pool.close()
    # divide_process_pool.join()
    #print("process done.")

    
    # wirte labels finally
    widgets = ['label_writing: ', Percentage(), ' ', Bar('#'), ' ', Timer(), ' ']
    bar_write_label = progressbar.ProgressBar(widgets=widgets, maxval=len(label_divided))
    bar_write_label.start()
    image_count = 0
    with open("./data/train.txt", "w") as f:
        for key in label_divided:
            label_list = label_divided[key]
            if len(label_list) == 0:
                continue
            file_name = "./data/image_split/"+key+".png"
            for label in label_list:
                file_name = file_name + " %d,%d,%d,%d,%d" % (label[0], label[1], label[2], label[3], 0)
            file_name = file_name + "\n"
            f.write(file_name)
            image_count += 1
            bar_write_label.update(image_count)
    
    bar_write_label.finish()
    


if __name__ == '__main__':
    hostid = "192.168.1.100"
    labels_contributor_set = ["pi", "root"]
    input_shape = (416, 416)
    min_overlop = 208
    padding = "zero"
    host_user = "liningbo"
    host_passwd = "1a2a3a"
    database = "target"
    data_achieve(hostid, host_user, database, host_passwd, labels_contributor_set, input_shape, min_overlop, padding)
