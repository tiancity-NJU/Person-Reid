
from __future__ import print_function, absolute_import
import os.path as osp
import os
import sys
import shutil
sys.path.insert(0,osp.abspath(osp.dirname(__file__)+osp.sep+'..'+osp.sep+'..'))



from reid.utils.data import Dataset
from reid.utils.osutils import mkdir_if_missing
from reid.utils.serialization import write_json
from reid.utils.serialization import read_json



class Combined(Dataset):

    def __init__(self, root, split_id=0, num_val=100, download=False):
        super(Combined, self).__init__(root, split_id=split_id)

        self.global_root=osp.dirname(self.root)

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
        import argparse

        parser = argparse.ArgumentParser(
            description="Combine Trainval Set of Market1501, CUHK03, DukeMTMC-reID")

        # Image directory and partition file of transformed datasets

        parser.add_argument(
            '--market1501_im_dir',
            type=str,
            default=osp.join(self.global_root,'market1501','images')
        )
        parser.add_argument(
            '--market1501_split_file',
            type=str,
            default=osp.join(self.global_root,'market1501','split.json')
        )
        parser.add_argument(
            '--market1501_meta_file',
            type=str,
            default=osp.join(self.global_root, 'market1501', 'meta.json')
        )



        parser.add_argument(
            '--cuhk03_im_dir',
            type=str,
            default=osp.join(self.global_root, 'cuhk03', 'images')
        )
        parser.add_argument(
            '--cuhk03_split_file',
            type=str,
            default=osp.join(self.global_root, 'cuhk03', 'split.json')
        )
        parser.add_argument(
            '--cuhk03_meta_file',
            type=str,
            default=osp.join(self.global_root, 'cuhk03', 'meta.json')
        )




        parser.add_argument(
            '--dukemtmc_im_dir',
            type=str,
            default=osp.join(self.global_root, 'dukemtmc', 'images')
        )
        parser.add_argument(
            '--dukemtmc_split_file',
            type=str,
            default=osp.join(self.global_root, 'dukemtmc', 'split.json')
        )
        parser.add_argument(
            '--dukemtmc_meta_file',
            type=str,
            default=osp.join(self.global_root, 'dukemtmc', 'meta.json')
        )

        args=parser.parse_args()

        im_dirs=[
            osp.abspath(args.market1501_im_dir),
            osp.abspath(args.cuhk03_im_dir),
            osp.abspath(args.dukemtmc_im_dir)
        ]

        split_dirs=[
            osp.abspath(args.market1501_split_file),
            osp.abspath(args.cuhk03_split_file),
            osp.abspath(args.dukemtmc_split_file)
        ]

        meta_dirs=[
            osp.abspath(args.market1501_meta_file),
            osp.abspath(args.cuhk03_meta_file),
            osp.abspath(args.dukemtmc_meta_file)
        ]

        dataset_names=[
            'market1501',
            'cuhk03',
            'dukemtmc'
        ]


        save_dir = self.root
        mkdir_if_missing(save_dir)

        self.combine_trainval_sets(im_dirs,split_dirs,meta_dirs,dataset_names,save_dir)






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
            num_cameras = max(num_cameras,read_json(osp.join(self.root, dataset_name, 'meta.json'))['num_cameras'])

        trainval_pids = []
        identities = []
        for im_dir, split_dirs, meta_dir, dataset_name in zip(im_dirs, split_dirs, meta_dirs, dataset_names):

            print('deal with ',dataset_name)
            split = read_json(osp.join(self.root, dataset_name, 'splits.json'))[0]
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
            'trainval':sorted(list(trainval_pids)),
            'query': [],
            'gallery': []
        }]

        write_json(splits,osp.join(save_dir, 'splits.json'))



if __name__ == '__main__':
    data=Combined('/home/zhang/PycharmProjects/open-reid/data/combined')
