#!/usr/bin/env python

'''
Functions for downloading and querying the Allen atlas images
'''

import simplejson
import urllib
import os
from registration import config
from registration import util
import numpy as np

def response_to_matrix(a, d):
    '''
    Utility function that interprets the web API response to a transform matrix
    @param a: response array
    @param d: 2 for 2D transform (6 parameters); 3 for 3D transform (12 parameters)
    '''
    if d == 2:
        return np.float32([[a['tsv_00'], a['tsv_01'], a['tsv_04']],
                  [a['tsv_02'], a['tsv_03'], a['tsv_05']]])
    elif d == 3:
        return np.float32([[a['tvr_00'], a['tvr_01'], a['tvr_02'], a['tvr_09']],
                  [a['tvr_03'], a['tvr_04'], a['tvr_05'], a['tvr_10']],
                  [a['tvr_06'], a['tvr_07'], a['tvr_08'], a['tvr_11']]])
        
def query_atlas_info(atlas_name):
    '''
    Get information of atlas. If cached, directly load from local; otherwise query web API.
    @param atlas_name: Name of atlas
    '''
    atlas_id, prefix = config.atlas_menu[atlas_name]
    if prefix == 'specimen':
        url = "http://api.brain-map.org/api/v2/data/SectionDataSet/query.json?" + \
            "criteria=products[abbreviation$eq'MouseRef'],treatments[name$eq'NISSL'],plane_of_section[name$eq'coronal']" + \
            "&include=section_images,alignment3d,section_images(alignment2d)" + \
            "&only=section_images.specimen_id,section_images.id,section_images.section_number"
    elif prefix == 'dataset':
        url = "http://api.brain-map.org/api/v2/data/SectionDataSet/" + str(atlas_id) + ".json?" + \
            "&include=section_images[annotated$eqtrue],alignment3d,section_images(alignment2d)"
    
    atlas_id, prefix = config.atlas_menu[atlas_name]
    pickle_filename = config.ALLEN_FOLDER + '%s_%d.p' % (prefix, atlas_id)
    try:
        atlas_info = util.pickle_load(pickle_filename)
    except:
        js = simplejson.loads(urllib.urlopen(url).read())
        images_info = dict([])
        for image_info in js['msg'][0]['section_images']:
            section_id = image_info['id']
            section_number = image_info['section_number']
            base_name = '%s_%d_%d_%d' % (prefix, atlas_id, section_number, section_id)
            images_info[section_number] = {'id':section_id, 'tier_count':image_info['tier_count'],
                                           'alignment2d':response_to_matrix(image_info['alignment2d'], 2),
                                           'filename':base_name}
    #    images_info = sorted(images_info, key=lambda x: x['section_number'])
        atlas_info = {'alignment3d':response_to_matrix(js['msg'][0]['alignment3d'], 3), 'section_images':images_info}
        util.pickle_save(atlas_info, pickle_filename)    
    return atlas_info
            
def download_atlas(atlas_name, downsample=4):
    '''
    Download the images of an atlas
    @param atlas_name: Name of atlas
    @param downsample: downsample level, default is 4
    '''
    print 'download', atlas_name
    atlas_info = query_atlas_info(atlas_name)
        
    atlas_id, prefix = config.atlas_menu[atlas_name]
    downloadAnnotated =  prefix == 'specimen'
    os.chdir(config.ALLEN_FOLDER)
    if not os.path.exists(str(atlas_id)):
        os.mkdir(str(atlas_id))
    anno_folder = str(atlas_id)+'_annotated'
    if downloadAnnotated and not os.path.exists(anno_folder):
        os.mkdir(anno_folder)
        
    for section_number, image_info in atlas_info['section_images'].iteritems():
        tier_count = image_info['tier_count']
        section_id = image_info['id']
        base_name = image_info['filename']
        image_name = '%d/%s_%d.jpg' % (atlas_id, base_name, downsample)
        image_url = "http://api.brain-map.org/api/v2/section_image_download/" + str(section_id) + "?downsample=" + str(downsample)
        print 'downloading', image_name
        urllib.urlretrieve(image_url, image_name)
        
        if downloadAnnotated:
            annotate_name = '%s/%s_%d_annotated.jpg' % (anno_folder, base_name, downsample)
            image_url = "http://api.brain-map.org/api/v2/atlas_image_download/" + str(section_id) + "?downsample=" + str(downsample) + "&annotation=true"
            print 'downloading', annotate_name
            urllib.urlretrieve(image_url, annotate_name)
    print 'done'        
                    
if __name__ == '__main__':
    specimen_info = query_atlas_info('p56_coronal')
    download_atlas('p56_coronal')
