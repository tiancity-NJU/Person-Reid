
from __future__ import print_function, absolute_import
import os.path as osp
import os
import sys
import shutil
sys.path.insert(0,osp.abspath(osp.dirname(__file__)+osp.sep+'..'+osp.sep+'..'))


import re
from reid.utils.data import Dataset
from reid.utils.osutils import mkdir_if_missing
from reid.utils.serialization import write_json
from reid.utils.serialization import read_json



class Combined(Dataset):

    def __init__(self, root, split_id=0, num_val=300,dataset_names = ['market1501','cuhk03','dukemtmc'], download=False):
        super(Combined, self).__init__(root, split_id=split_id)

        self.global_root = osp.dirname(self.root)
        self.dataset_names = dataset_names

        if not self.check_datasets():
            raise RuntimeError("Please download all subdatasets fristly," +
                               "Then run the script")

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. " +
                               "You can use download=True to download it.")

        self.load(num_val)


    def download(self):
        """
            just return the combined dataset.
        :return:
        """

        im_dirs = []
        splits_dirs = []
        meta_dirs = []

        for dataset in self.dataset_names:
            im_dirs.append(osp.abspath(osp.join(self.global_root, dataset, 'images')))
            splits_dirs.append(osp.abspath(osp.join(self.global_root, dataset, 'splits.json')))
            meta_dirs.append(osp.abspath(osp.join(self.global_root, dataset, 'meta.json')))

        dataset_names=self.dataset_names
        save_dir = self.root
        mkdir_if_missing(save_dir)

        self.combine_trainval_sets(im_dirs,splits_dirs,meta_dirs,dataset_names,save_dir)


    def combine_trainval_sets(self, im_dirs, split_dirs, meta_dirs, dataset_names, save_dir):
        """

        :param im_dirs:
        :param split_dirs:
        :param meta_dirs:
        :param save_dir:
        :return:   create combined dataset based on origin re-id three dataset
        """

        new_images_dir=osp.join(save_dir,'images')
        mkdir_if_missing(new_images_dir)

        start_id = 0
        num_cameras = 0
        for meta_dir, dataset_name in zip(meta_dirs,dataset_names):
            num_cameras = max(num_cameras,read_json(osp.join(self.global_root, dataset_name, 'meta.json'))['num_cameras'])

        trainval_pids = []
        identities = []
        for im_dir, split_dir, meta_dir, dataset_name in zip(im_dirs, split_dirs, meta_dirs, dataset_names):

            split = read_json(split_dir)[0]
            id_map=dict(zip(split['trainval'],range(start_id,start_id+len(split['trainval']))))
            trainval_pids.extend(range(start_id,start_id+len(split['trainval'])))

            identities_tmp=[[[] for _ in range(num_cameras)] for _ in range(len(split['trainval']))]
            im_list=os.listdir(im_dir)

            for img in im_list:
                if int(img[:8]) not in id_map: continue
                new_id = id_map[int(img[:8])]
                cam = int(img[9:11])
                new_im_name = img.replace(img[:8],'{:08d}'.format(new_id))

                identities_tmp[new_id-start_id][cam].append(new_im_name)
                shutil.copy(osp.join(im_dir,img), osp.join(save_dir, 'images', new_im_name))

            start_id += len(split['trainval'])
            identities.extend(identities_tmp)

        meta = {'name': 'Combined', 'shot': 'multiple', 'num_cameras': num_cameras,
                'identities': identities}
        write_json(meta, osp.join(save_dir, 'meta.json'))

        splits =[{
            'trainval': sorted(list(trainval_pids))[:-100],
            'query': sorted(list(trainval_pids))[-100:] ,
            'gallery': sorted(list(trainval_pids))[-100:]
        }]

        write_json(splits,osp.join(save_dir, 'splits.json'))


    def check_datasets(self):
        """
            datasets is the list of subdataset. check wheather every subdatasets is exist

        :return:
        """

        for dataset in self.dataset_names:
            if not osp.isdir(osp.join(self.global_root, dataset, 'images')) and \
                   osp.isfile(osp.join(self.global_root, dataset, 'meta.json')) and \
                   osp.isfile(osp.join(self.global_root, dataset, 'splits.json')):
                return False

        return True



if __name__ == '__main__':
    data=Combined('/home/zhang/PycharmProjects/open-reid/data/combined',download=True)
