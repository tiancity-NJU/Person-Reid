
import os
import os.path as osp
import sys
import shutil

sys.path.insert(0,osp.abspath(osp.dirname(__file__)+osp.sep+'..'+osp.sep+'..'))

from reid.utils.data import Dataset
from reid.utils.serialization import write_json
from reid.utils.serialization import read_json
from reid.utils.osutils import mkdir_if_missing



class TestData(Dataset):


    def __init__(self, root, split_id=0, num_val=0, dst=None, download=False):
        super(TestData, self).__init__(root, split_id=split_id)

        self.data_abspaths = [osp.join(self.root, path) for path in os.listdir(self.root)]
        self.num_cameras = len(self.data_abspaths)
        self.dst = dst if dst != None else self.root

        new_img_dir = osp.join(self.dst, 'images')

        mkdir_if_missing(new_img_dir)

        if download:
            self.download()

        self.load(num_val)


    def download(self):
        query_ids = []   # we do not have label. so just 0
        gallery_ids = []
        identities = [[[] for _ in range(self.num_cameras)] for _ in range(100)]

        for abspath in self.data_abspaths:
            imgs = os.listdir(abspath)
            for img in imgs:
                info = img.split('_')
                cam_id = int(info[1][1:])
                new_img_name = '{:08d}_{:02d}_{:08d}'.format(int(info[3][:-4]),int(info[1][1:]),int(info[2][1:]))+'.jpg'
                shutil.copy(osp.join(abspath,img),osp.join(self.dst,'images',new_img_name))

                identities[int(info[3][:-4])][cam_id].append(new_img_name)
                if int(info[3][:-4]) not in query_ids:
                    query_ids.append(int(info[3][:-4]))
                    gallery_ids.append(int(info[3][:-4]))

        meta = {'name': 'Test', 'shot': 'multiple', 'num_cameras': self.num_cameras,
                'identities': identities}
        write_json(meta, osp.join(self.dst, 'meta.json'))

        splits = [{
            'trainval': [],
            'query': query_ids,
            'gallery': gallery_ids
        }]

        write_json(splits, osp.join(self.dst, 'splits.json'))


if __name__ == '__main__':

    testdata=TestData('/home/ztc/zhang/PycharmProjects/bbox_reid/cam',dst='/home/ztc/zhang/PycharmProjects/open-reid/data/test',download=True)



