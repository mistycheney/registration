#!/usr/bin/env python

'''
Functions for downloading and querying the Allen atlas images
'''

import simplejson
import urllib2
import urllib
import os
from registration import config
from registration import util
#from registration import sift
import numpy as np

def get_api_response(url):
    js = simplejson.loads(urllib.urlopen(url).read())
    return js

def get_filename(dataset_id, section_num, clean=True):
    if clean:
        for file in os.listdir(config.ALLEN_FOLDER + str(dataset_id) + '_clean'):
            if str(dataset_id) + '_' + str(section_num) + '_' in file and '_clean' in file:
                return file
    else:
        for files in os.listdir(config.ALLEN_FOLDER + str(dataset_id)):
            if str(dataset_id) + '_' + str(section_num) + '_' in files:
                return files
    raise ValueError

def download_atlas(image_id, downsample, image_name):
    image_url = "http://api.brain-map.org/api/v2/atlas_image_download/" + str(image_id) + "?downsample=" + str(downsample) + "&annotation=true"
    urllib.urlretrieve(image_url, image_name)

def download_image(image_id, downsample, image_name):
    image_url = "http://api.brain-map.org/api/v2/section_image_download/" + str(image_id) + "?downsample=" + str(downsample)
    urllib.urlretrieve(image_url, config.ALLEN_FOLDER + image_name)

def response_to_matrix(a,d):
    if d == 2:
        return np.float32([[a['tsv_00'],a['tsv_01'],a['tsv_04']],
                  [a['tsv_02'],a['tsv_03'],a['tsv_05']]])
    elif d == 3:
        return np.float32([[a['tvr_00'],a['tvr_01'],a['tvr_02'],a['tvr_09']],
                  [a['tvr_03'],a['tvr_04'],a['tvr_05'],a['tvr_10']],
                  [a['tvr_06'],a['tvr_07'],a['tvr_08'],a['tvr_11']]])
        
#def response_to_matrix_inv(a,d):
#    if d == 2:
#        return float32([[a['tvs_00'],a['tvs_01'],a['tvs_04']],
#                  [a['tvs_02'],a['tvs_03'],a['tvs_05']]])
#    elif d == 3:
#        return float32([[a['trv_00'],a['trv_01'],a['trv_06']],
#                  [a['trv_02'],a['trv_03'],a['trv_07']],
#                  [a['trv_04'],a['trv_05'],a['trv_08']]])
    

def query_dataset(dataset_id):
    import cPickle as pickle
    try:
        dataset_info = pickle.load('dataset_%d.p'%dataset_id)
    except:
        url = "http://api.brain-map.org/api/v2/data/SectionDataSet/" + str(dataset_id) + ".json?" + \
            "&include=section_images[annotated$eqtrue],alignment3d,section_images(alignment2d)"
    #        "&include=section_images,alignment3d,section_images(alignment2d)"
    #    print url
        js = get_api_response(url)
    #    images_info = [{'section_number':im['section_number'], 'id':im['id'], 'tier_count':im['tier_count'],
    #                    'alignment2d':response_to_matrix(im['alignment2d'],2)}
    #                   for im in js['msg'][0]['section_images']]
        downsample = 4
        images_info = dict([(im['section_number'], {'id':im['id'], 'tier_count':im['tier_count'],
                    'alignment2d':response_to_matrix(im['alignment2d'],2),
                    'filename':'dataset_' + str(dataset_id) + '_' + str(im['section_number']) +\
                        '_' + str(im['id']) + '_' + str(downsample) + '.jpg'})
                   for im in js['msg'][0]['section_images']])
    #    images_info = sorted(images_info, key=lambda x: x['section_number'])
        dataset_info = {'alignment3d':response_to_matrix(js['msg'][0]['alignment3d'],3), 'section_images':images_info}
        pickle.dump(dataset_info, open('dataset_%d.p'%dataset_id,'wb'))
    return dataset_info

def download_dataset(dataset_id):
    dataset_info = util.conditional_load(\
        'dataset_' + str(dataset_id), query_dataset, [dataset_id], regenerate=False)
    for section_number, image_info in dataset_info['section_images'].iteritems():
        tier_count = image_info['tier_count']
        section_id = image_info['id']
        print 'downloading section image', section_number, '...',
        downsample = 4
        image_name = 'dataset_' + str(dataset_id) + '_' + str(section_number) +\
                    '_' + str(section_id) + '_' + str(downsample) + '.jpg'
        download_image(section_id, downsample, config.ALLEN_FOLDER + str(dataset_id) + '/' + image_name)
        atlas_name = 'dataset_' + str(dataset_id) + '_' + str(section_number) +\
                    '_' + str(section_id) + '_' + str(downsample) + '_atlas.jpg'
        download_atlas(section_id, downsample, config.ALLEN_FOLDER + str(dataset_id) + '_atlas/' + atlas_name)
        print 'done'


def point_to_reference_imageid(image_id, x, y, downsample, tier_count):
    x = x * 2**(tier_count - downsample)
    y = y * 2**(tier_count - downsample)
    url = "http://api.brain-map.org/api/v2/image_to_reference/" +\
     str(image_id) + ".json?x=" + str(x) + "&y=" + str(y)
    d = get_api_response(url)['msg']['image_to_reference']
    return np.float32([d['x'], d['y'], d['z']])

def points_to_reference(dataset_id, section_number, kp_list, downsample):
    dataset_info = util.conditional_load('dataset_' + str(dataset_id), query_dataset, [dataset_id], regenerate=False)
    images_info = dataset_info['section_images']
    tier_count = images_info[section_number]['tier_count']

    scaling = 2**(tier_count - downsample)
    kp_list = kp_list * scaling        #kp_list:n*2
    n = len(kp_list)
    kp_homo_list = np.hstack([kp_list, np.ones((n,1))])
    kp_align2d_list = np.dot(images_info[section_number]['alignment2d'], kp_homo_list.T)
    kp_align2d_homo_list = np.hstack([kp_align2d_list.T, 25*section_number*np.ones((n, 1)), np.ones((n, 1))])
    kp_align3d_list = np.dot(dataset_info['alignment3d'], kp_align2d_homo_list.T)
#    print kp_align3d_list.T
    return kp_align3d_list.T

def point_to_reference_sectionnum_api(dataset_id, section_number, kp_list, downsample):
    dataset_info = util.conditional_load('dataset_' + str(dataset_id), query_dataset, [dataset_id], regenerate=False)
    image_id = dataset_info['section_images'][section_number]['id']
    tier_count = dataset_info['section_images'][section_number]['tier_count']
    p_ref_list = [point_to_reference_imageid(image_id, kp[0], kp[1], downsample, tier_count) for kp in kp_list]
    return p_ref_list


def get_secnums(dataset_id):
    dataset_info = util.conditional_load('dataset_' + str(dataset_id), query_dataset, [dataset_id], regenerate=False)
    return dataset_info['section_images'].keys()

def download_specimen(specimen_id):
    specimens = util.conditional_load('specimen_' + str(specimen_id), retrieve_specimens, [specimen_id], regenerate=False)
    for im in specimens:
        section_number = im['section_number']
        image_id = im['id']
        downsample = 4
        image_name = 'specimen_' + str(specimen_id) + '_' + str(section_number) + '_' +str(image_id)+'_' + str(downsample) + '.jpg'
        print image_name
        download_image(image_id, downsample, image_name)

def retrieve_specimens(specimen_id):
    import cPickle as pickle
    try:
        specimen_info = pickle.load('specimen_%d.p'%specimen_id)
    except:
        url = "http://api.brain-map.org/api/v2/data/SectionDataSet/query.json?" + \
        "criteria=products[abbreviation$eq'MouseRef'],treatments[name$eq'NISSL'],plane_of_section[name$eq'coronal']" + \
        "&include=section_images,alignment3d,section_images(alignment2d)" + \
        "&only=section_images.specimen_id,section_images.id,section_images.section_number"
    
        js = get_api_response(url)
        downsample = 4
        images_info = dict([(im['section_number'], {'id':im['id'], 'tier_count':im['tier_count'],
                        'alignment2d':response_to_matrix(im['alignment2d'],2),
                        'filename':'specimen_' + str(specimen_id) + '_' + str(im['section_number']) +\
                            '_' + str(im['id']) + '_' + str(downsample) + '.jpg'})
                       for im in js['msg'][0]['section_images']])
        specimen_info = {'alignment3d':response_to_matrix(js['msg'][0]['alignment3d'],3), 'section_images':images_info}
        pickle.dump(specimen_info, open('specimen_%d.p'%specimen_id,'wb'))
#    images_info = [{'section_number':im['section_number'], 'id':im['id']}
#                   for im in js['msg'][0]['section_images']]
#    images_info = sorted(images_info, key=lambda x: x['section_number'])
    return specimen_info

#def query_image(image_id):
#    url = "http://api.brain-map.org/api/v2/data/SectionImage/" + str(image_id) + ".json?include=data_set&only=data_sets.reference_space_id,data_sets.id"
#    socket = urllib.urlopen(url)
#    js = simplejson.loads(socket.read())
#    return js['msg'][0]['data_set']

#def image_to_volume(image_id):
#    url = "http://api.brain-map.org/api/v2/data/SectionImage/" + str(image_id) + ".json?"+ \
#    "include=alignment2d,data_set"
#    js = simplejson.loads(urllib.urlopen(url).read())
#    a2 = js['msg'][0]['alignment2d']
#    dataset_id = js['msg'][0]['data_set']['id']
#    
#    url = "http://api.brain-map.org/api/v2/data/SectionDataSet/" + str(dataset_id) + ".json?" + \
#    "include=alignment3d,reference_space"
#    js = simplejson.loads(urllib.urlopen(url).read())
#    a3 = js['msg'][0]['alignment3d']
#    
#    A2 = float32([[a2['tsv_00'],a2['tsv_01'],a2['tsv_04']],
#                  [a2['tsv_02'],a2['tsv_03'],a2['tsv_05']]])
#    A3 = float32([[a3['tvr_00'],a3['tvr_01'],a3['tvr_02'],a3['tvr_09']],
#                  [a3['tvr_03'],a3['tvr_04'],a3['tvr_05'],a3['tvr_10']],
#                  [a3['tvr_06'],a3['tvr_07'],a3['tvr_08'],a3['tvr_11']]])
#    return A2, A3

#def get_allen_matching_multisections(imname_from, dataset_id, section_numbers):
#    d_from = sift.detect(imname_from)
#    matchings = []
#    os.chdir(config.ALLEN_FOLDER + str(dataset_id) + '_clean')
#    for sec in section_numbers:
#        print '\nAllen:', sec
#        matching = get_allen_matching(d_from, dataset_id, sec)
#        matchings.append(matching)
#    return matchings
    
#def get_allen_matching(d_from, dataset_id, section_number):
#    allen_name = get_filename(dataset_id, section_number)
#    d_to = sift.detect(allen_name)
#    return sift.get_matching(d_from, d_to)

                    
    
if __name__ == '__main__':

#    id_code = {'p56_coronal':100048576}
    
#    specimen_info = retrieve_specimens(5756)

    dataset_info = query_dataset(100048576)
    
#    dataset_id = id_code['p56_coronal']
#    p56coronal_info = util.conditional_load('dataset_' + str(dataset_id), \
#                                query_dataset, [dataset_id], regenerate=False)
    
#    section_number = 493
#    downsample = 4
#    import numpy
#    kp_list = numpy.random.random([100,2])
#    p1 = points_to_reference(dataset_id, section_number, kp_list, downsample)
#    p2 = point_to_reference_sectionnum_api(dataset_id, section_number, kp_list, downsample)
#    print all(p1.T-p2<0.001)

    
#    specimen_id = 5756
#    specimen_info = load_specimen(specimen_id)
     
#    download_dataset(id_code['p56_coronal'])
#    download_dataset('p56_sagittal')
#    A2d, A3d = image_to_volume(100960165)
#    print A2d
#    print A3d
