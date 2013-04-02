from subprocess import call
import os, sys
from pylab import *
from registration.config import * 
import cv2
import collections
import networkx


def ndpi_split(stack_id, slide_num, stain='nissl'):
    os.chdir(SLIDE_FOLDER)
    if stain == 'dab':
        ndpi_prefix = 'H0{0}_DAB_{1}_'.format(str(stack_id), str(slide_num))
    else:
        ndpi_prefix = '{0}_{1}_'.format(str(stack_id), str(slide_num))
    print  os.listdir('.')
    print ndpi_prefix

    ndpi_name = [filename for filename in os.listdir('.')\
                 if not filename.startswith('.') and filename.startswith(ndpi_prefix)][0]
    new_ndpi_name = str(stack_id) + '_' + str(slide_num) + '_' + stain + '.ndpi'
    os.rename(ndpi_name, new_ndpi_name)
    try:
        cmmd = ' '.join([ '/usr/local/bin/ndpisplit', "'" + new_ndpi_name + "'"])
        print cmmd
        retcode = call(cmmd, shell=True)
        if retcode < 0:
            print >> sys.stderr, "Child was terminated by signal", -retcode
        else:
            print >> sys.stderr, "Child returned", retcode
    except OSError, e:
        print >> sys.stderr, "Execution failed:", e


def split_slide(stack_id, slide_num, starting_section_id):
    '''Segment a tif slide containing multiple section images.
    '''
#    imname = SLIDE_FOLDER + '{0}_{1}_x0.3125_z0.tif'.format(str(stack_id), str(slide_num))
    imname = '/Users/yuncong/Documents/brain images/I.tif'
    img = imread(imname)
    h, w = img.shape[:2]
    img = cv2.cvtColor(img, cv2.cv.CV_RGB2GRAY)
            
    def find_large_contours(thresh_image):
        '''
        return only the "large" contours.
        '''
        thresh_image[:, 0] = 0
        thresh_image[:, -1] = 0
        thresh_image[0, :] = 0
        thresh_image[-1, :] = 0
        
        h, w = tuple(thresh_image.shape[:2])
        contours, hierarchy = cv2.findContours(thresh_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        all_bbox = [cv2.boundingRect(i) for i in contours]
        all_area = array([cv2.contourArea(c) for c in contours])
        all_aspect_ratio = array([float(b[2]) / b[3] for b in all_bbox])
        large = find((all_area > MIN_AREA) & (all_area < h * w * 0.8) & (all_aspect_ratio < 10))
        large_contours = [contours[i] for i in large]
            
        print len(large_contours), 'large_contours'
    
        mask = zeros((h, w, 1), uint8)
        cv2.drawContours(mask, large_contours, -1, 255, -1)
#        cv2.imshow('', mask[:, :, 0])
#        cv2.waitKey()
        return large_contours, mask[:, :, 0]

    def clean(img, white_bg=True):
        '''
        clean a slide image, so that only the interesting parts are kept
        '''
#        from scipy import ndimage
        
    #    cv2.imshow('',img)
    #    cv2.waitKey()
        img_smooth = cv2.medianBlur(img, 5)
        img_smooth = cv2.GaussianBlur(img_smooth, (3, 3), 3)
        
        h, w = img.shape
        markers = zeros_like(img).astype(int16)
        markers[5, 5] = 1
        
        flooded = img_smooth.copy()
        mask = zeros((h + 2, w + 2), uint8)
        connectivity = 8
    #    flags = connectivity | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY
        flags = connectivity | cv2.FLOODFILL_FIXED_RANGE
        seed_pt = 5, 5
        lo = 5
        hi = 255
        retval, rect = cv2.floodFill(flooded, mask, seed_pt, (255, 255, 255), lo, hi, flags)
    #    cv2.imshow('floodfill', flooded)
    #    cv2.waitKey()
    
        mask = 1 - mask[1:-1, 1:-1]    
        large_contours, mask = find_large_contours(mask)
    
        if white_bg:
            white = 255 * ones((h, w), uint8)
            img_clean = (white - mask) + (img & mask)
        else:
            img_clean = img & mask
        cv2.imshow('', img_clean)
        cv2.waitKey()
        return img_clean, large_contours
    
    def show_contours(cnts, fill=False, show=False, title=''):
        '''
        A utility function that returns a white-background image showing the contour given by cnts.
        '''
        vis = zeros((h, w, 1), uint8)
        if fill:
            cv2.drawContours(vis, cnts, -1, 255, -1)
        else:
            cv2.drawContours(vis, cnts, -1, 255, 2)
        if show and DEBUG:
            cv2.imshow(title, vis[:, :, 0])
            cv2.waitKey()
        return vis[:, :, 0]
                                        
#    def rect_to_contour(b):
#        bb = ((b[0] + b[2] / 2, b[1] + b[3] / 2), (b[2], b[3]), 0)
#        return int32(cv2.cv.BoxPoints(bb)).reshape(4, 1, 2)
##            return array([(b[0],b[1]),(b[0],b[1]-b[3]),(b[0]+b[2],b[1]-b[3]),\
##                          (b[0]+b[2],b[1])]).reshape(4, 1, 2)

    def union_contours(cnts, mode):
        '''
        merge several contours, given a mode among "bbox", "hull" or "contourArea".
        '''
        mask = show_contours(cnts, fill=True, show=False)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, \
                                     cv2.CHAIN_APPROX_SIMPLE)
        if mode.startswith('bbox'):
            rects = [cv2.boundingRect(c) for c in contours]
            x_min = min([rect[0] for rect in rects])
            x_max = max([rect[0] + rect[2] for rect in rects])
            y_min = min([rect[1] for rect in rects])
            y_max = max([rect[1] + rect[3] for rect in rects])            
            v = array([(x_min, y_max), (x_min, y_min), (x_max, y_min), (x_max, y_max)])
            big_bbox = int32(v.reshape(4, 1, 2))
            if 'Area' in mode:
                return cv2.contourArea(big_bbox)
            else:
                return big_bbox
        elif mode.startswith('hull'):
            hull = cv2.convexHull(vstack(contours))
            if 'Area' in mode:
                return hull
            else:
                return cv2.contourArea(hull)
        elif mode == 'contourArea':
            return cnts, sum([cv2.contourArea(cnt) for cnt in contours])    
        
    
    def split_contour(contour_groups, bbox):
        '''
        split a contour into smaller contours of parts that are likely to belong to different sections.
        
        '''
        
        area = float32([(b[3][0][0] - b[1][0][0]) * (b[3][0][1] - b[1][0][1]) for b in bbox])
        order, area = zip(*sorted(list(enumerate(area)), key=lambda x:x[1]))
        contour_groups = [contour_groups[i] for i in order] 
        
        print 'area:', area
                
        ratio = [None] * (len(area) - 1)
        for i in range(len(area) - 1):
            ratio[i] = round(area[i + 1] / median(area))
        print ratio
        to_split_begin_possible = find(array(ratio) > 1) + 1
        
        if len(to_split_begin_possible) == 0:
            return contour_groups
        
        to_split_begin = min(to_split_begin_possible)
        to_split = range(to_split_begin, len(contour_groups))
        print 'to_split', to_split
        contour_groups_to_split = [contour_groups[k] for k in to_split]
        
        new_contours = []
        for i, group in enumerate(contour_groups_to_split):
            print 'split', i
            if len(group) > 1:
                print 'already splitted'
                new_contours += [[c] for c in group]
                print 'new_contours', len(new_contours)
                contour_groups = [i for j, i in enumerate(contour_groups) if j not in to_split]
                contour_groups = list(contour_groups) + new_contours
                print 'contour_groups', len(contour_groups)
                continue
            orig_mask = show_contours(group, fill=True, show=True)
            element = cv2.getStructuringElement(STRUCTURE_ELEMENT, (MORPH_ELEMENT_SIZE, MORPH_ELEMENT_SIZE))
            eroded_mask = orig_mask
            it = 0
            while 1:
    #            eroded_mask = cv2.morphologyEx(eroded_mask, cv2.MORPH_OPEN, element, iterations=15)
                eroded_mask = cv2.erode(eroded_mask, element)
                indiv_contours, hierarchy = cv2.findContours(eroded_mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)                
                it += 1
#                cv2.imshow('',eroded_mask)
#                cv2.waitKey()
                indiv_bbox = [cv2.boundingRect(i) for i in indiv_contours]
                indiv_bbox_area = array([b[2] * b[3] for b in indiv_bbox])
                valid = find(indiv_bbox_area > MIN_AREA)
                if len(valid) == 0:
                    print '!!!!!!!!!!!!!!!!!!!!', it
                    return None
                
                indiv_bbox = [indiv_bbox[i] for i in valid]
                indiv_bbox_area = indiv_bbox_area[valid]
                indiv_contours = [indiv_contours[i] for i in valid]
                
#                print 'indiv_bbox_area', indiv_bbox_area
                area_std = std(indiv_bbox_area)
#                print 'area_std', area_std
#                show_contours(indiv_contours, fill=False, show=True)
                
                if (len(indiv_contours) > 1 and area_std < SPLIT_STOP_STD):
                    break
    
            print 'indiv_contours', len(indiv_contours)
    
            # restore individual contour
            dilated_mask = [cv2.dilate(show_contours([c], fill=True), element, iterations=it) \
                            for c in indiv_contours]
            dilated_mask_sum = reduce(lambda x, y: x + y, dilated_mask)
            dilated_mask_remain = orig_mask - dilated_mask_sum
            dilated_mask_new = [dilated_mask_remain + m for m in dilated_mask]
            indiv_dilated_contours = []
            for m in dilated_mask_new:
                indiv_dilated_contour_noisy, hierarchy = cv2.findContours(m.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                indiv_dilated_contour_big = [c for c in indiv_dilated_contour_noisy if cv2.contourArea(c) > MIN_AREA]
                indiv_dilated_contour_biggest = sorted(indiv_dilated_contour_big, key=cv2.contourArea)[-1]                
                indiv_dilated_contours.append(indiv_dilated_contour_biggest)
#                cv2.imshow('m', m)
#                cv2.waitKey()
                show_contours([indiv_dilated_contour_biggest], show=True)
            
            new_contours = new_contours + [[c] for c in indiv_dilated_contours]
            print 'new_contours', len(new_contours)
            contour_groups = [i for j, i in enumerate(contour_groups) if j not in to_split]
            contour_groups = list(contour_groups) + new_contours
            print 'contour_groups', len(contour_groups)
        
        return contour_groups
             
    def find_groupings(contours):
        '''
        find all possible groupings given the contours. 
        Formulate as a maximal independent set problem in graph.
        '''
        show_contours(contours, show=True)
        n = len(contours)
#        area = [cv2.contourArea(c) for c in contours]
#        print 'area', area                                
                                
        close_pairs = set()
        distance = array([[inf] * n] * n)   
        for i1, b1 in enumerate(contours):
            for i2, b2 in enumerate(contours):
                distance[i1, i2] = max([cv2.pointPolygonTest(b2, tuple(v[0]), True) for v in b1])
        
#        print distance
        
        min_dist = [0] * n   
        for r, row in enumerate(distance):
            min_dist[r] = min([-i for i in row if i < 0])
                
#        distance = float32(distance)
#        min_dist = min(distance, axis = 1)
#        a = median(min_dist)
#        b = std(min_dist)
#        print a, b
#        FAR_THRESH_ADAPT = a - 2*b
        if stack_id == 4:
            FAR_THRESH_ADAPT = 110
        elif stack_id == 6:
            FAR_THRESH_ADAPT = 30
        elif stack_id == 9:
            FAR_THRESH_ADAPT = 40
        elif stack_id == 3:
            FAR_THRESH_ADAPT = 60
        else:
            FAR_THRESH_ADAPT = median(min_dist)
        print 'FAR_THRESH_ADAPT', FAR_THRESH_ADAPT
        
#        FAR_THRESH_ADAPT = 
                    
        for i1 in range(n):
            for i2 in range(n):
                if distance[i1, i2] > -FAR_THRESH_ADAPT and i1 != i2:
                    close_pairs.add(frozenset([i1, i2]))
        
        print len(close_pairs), 'close_pairs', close_pairs 
#        quit()
                        
        close_pairs = [pair for pair in close_pairs \
                       if union_contours([contours[i] for i in pair], 'bboxArea') < h * w / 4]
        print len(close_pairs), 'close_pairs', close_pairs 

        import itertools
#        far = set([frozenset(p) for p in itertools.product(range(n), repeat=2)]) - set(close_pairs)
#        far = [f for f in far if len(f) > 1]
#        print 'far', len(far), far
        
        close_matrix = zeros([n, n])
        for i, j in close_pairs:
            close_matrix[i, j] = 1
        G = networkx.from_numpy_matrix(close_matrix)
        clique_generator = networkx.algorithms.find_cliques(G)
        overlap_groups = [frozenset(c) for c in clique_generator if len(c) > 1]
        for group in overlap_groups[:]:
            for r in range(2, len(group)):
                for comb in itertools.combinations(group, r):     
                    overlap_groups.append(frozenset(comb))
        
        overlap_groups = list(set(overlap_groups)) 
        print len(overlap_groups), 'overlap_groups', overlap_groups

        if not overlap_groups:
            return set([frozenset([])])
        
        groupings = set()       
        conflict_matrix = array([[len(set(i) & set(j)) > 0 and i != j for i in overlap_groups] \
                              for j in overlap_groups])
        G = networkx.from_numpy_matrix(conflict_matrix)
        for i in range(1000):
            indep = networkx.algorithms.maximal_independent_set(G)
            groupings.add(frozenset([overlap_groups[i] for i in indep]))

        print 'Total', len(groupings), 'groupings'
#        for grouping in  groupings:
#            print grouping
        
        for i, grouping in enumerate(groupings.copy()):
            for r in range(len(grouping)):
                for subg in itertools.combinations(grouping, r):
                    groupings.add(frozenset(subg))

#        groupings = [frozenset(subg) for grouping in groupings for r in range(len(grouping)) for subg in itertools.combinations(grouping, r)]
#        groupings = set(groupings)
#        groupings = sorted(list(groupings))
        
        print 'After considering sub-groups,', len(groupings), 'groupings' 
        return groupings
        
    def compare_groupings(contours, groupings):
        '''
        Compare the possible groupings, find the best grouping
        '''
    
        area_std = []
        grouped_contours_list = []
        bboxes_list = []

        for k, grouping in enumerate(groupings):
            print 'grouping', k, grouping
            to_exclude = [i for group in grouping for i in group]
            grouping_idx = [[i] for i in range(len(contours)) if i not in to_exclude] + \
                            list(grouping)
            print 'grouping_idx', grouping_idx
                            
            grouped_contours = [[contours[i] for i in group] for group in grouping_idx]
                
            print len(grouped_contours), 'grouped_contours'
            
            bboxes = [union_contours(group, 'bbox') for group in grouped_contours]
            area0 = [cv2.contourArea(group[0]) if len(group) == 1 else union_contours(group, 'contourArea')[1] for group in grouped_contours]
            area_std0 = std(area0)
                    
#            centers = array([array([b[1][0]+b[3][0]])/2 for b in bboxes])
#            diff = centers[1:,0] - centers[:-1,0]
#            print 'center_diff', diff
#            print 'center_std', std(diff)
            
            show_contours(bboxes, fill=False, show=True)
            
            ret = split_contour(grouped_contours[:], bboxes)
            if ret is None:
                grouped_contours_list.append(grouped_contours)
                bboxes_list.append(bboxes)
                area_std.append(area_std0)
                continue
            else:
                grouped_contours = ret

            
            bboxes = [union_contours(group, 'bbox') for group in grouped_contours]
            show_contours(bboxes, fill=False, show=True)
            
            grouped_contours_list.append(grouped_contours)
            bboxes_list.append(bboxes)
            
            area = [cv2.contourArea(group[0]) if len(group) == 1 else union_contours(group, 'contourArea')[1] for group in grouped_contours]
            print area
            print std(area)
            area_std.append(std(area))
        
        print 'area_std', area_std
        
        best_idx = argmin(area_std)
        print best_idx
        best_grouping = grouped_contours_list[best_idx]
        best_bbox = bboxes_list[best_idx]
        print len(best_bbox)
    
        return best_grouping, best_bbox
                
    def sort_bboxes(bboxes):
#        print 'bboxes'
#        for b in bboxes:
#            print b
            
        print 'boxes', len(bboxes)
        idxes = sorted(range(len(bboxes)), key=lambda i: bboxes[i][0][0][0])
        xs = array([bboxes[i][0][0][0] for i in idxes])
        delta = xs[1:] - xs[:-1]
        gaps = find(delta > 110)
        nrow = bincount(gaps[1:] - gaps[:-1]).argmax() if len(gaps) > 1 else 1
        ncol = len(bboxes) / nrow
        print 'nrow, ncol', nrow, ncol
        
        idxes_sorted = []
        for col in range(ncol):    
            idxes_sorted += sorted(idxes[col * nrow:(col + 1) * nrow], key=lambda i: bboxes[i][0][0][1])
            
        return idxes_sorted
    
    
    img_thresh, large_contours = clean(img, white_bg=False)
    groupings = find_groupings(large_contours)
    grouped_contours, bboxes = compare_groupings(large_contours, groupings)
    
    idxes_sorted = sort_bboxes(bboxes)
    print 'idxes_sorted', idxes_sorted
    
    section_id = starting_section_id
    for i in idxes_sorted:
            
        group = grouped_contours[i]
        bbox = bboxes[i]
        mask = show_contours(group, fill=True)
        masked_img = img_thresh & mask
        sub_img = masked_img[bbox[2][0][1]:bbox[0][0][1], bbox[0][0][0]:bbox[2][0][0]]

        print 'section_id', section_id
#        cv2.imwrite(SECTION_FOLDER + str(stack_id) + '/' + str(stack_id) + '_' + str(starting_section_id) + '.tif', sub_img)
        section_id += 1
        cv2.imshow('', sub_img)
        cv2.waitKey()
        
    return len(bboxes)

        
if __name__ == '__main__':
    
    #    set_printoptions(threshold='nan')

    stack_ids = [3, 4, 5, 6, 9, 10]
    nslides = [10, 5, 4, 5, 7, 7]
    nslides = dict(zip(stack_ids, nslides))

    stack_id = 4
    nslide = nslides[stack_id]
        
#    for slide_num in range(1, nslide+1):
#        ndpi_split(stack_id, slide_num)
    
    section_id = 0
    for slide_num in range(1, nslide + 1):
#    slide_num = 2
        print 'slide', slide_num
        if str(stack_id) not in os.listdir(SECTION_FOLDER):
            os.mkdir(SECTION_FOLDER + str(stack_id))
        n = split_slide(stack_id, slide_num, section_id)
        section_id += n

        
#    import allen
#    dataset_id = 100048576
#    secs = allen.get_secnums(dataset_id)
#    os.chdir(ALLEN_FOLDER)
#    for i in secs:
#        f = allen.get_filename(dataset_id, i, clean=False)
#        imname = str(dataset_id) + '/' + f
#        print 'clean', imname
#        img = cv2.imread(imname, 0)
#        clean_image = clean(img)
#        cv2.imwrite(str(dataset_id) + '_clean/' + f[:-4] + '_clean.jpg', clean_image)

#    stack_id = 4
#    nsection = 40
#    os.chdir(SECTION_FOLDER)
#    if str(stack_id) + '_clean' not in os.listdir('.'):
#        os.mkdir(str(stack_id) + '_clean')
#    for sec in range(nsection):
#        clean_image = clean('{0}/{0}_{1}.tif'.format(str(stack_id), str(sec)))
#        cv2.imwrite('{0}_clean/{0}_{1}.tif'.format(str(stack_id), str(sec)), clean_image)
