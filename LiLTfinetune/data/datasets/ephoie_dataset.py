# coding=utf-8

import json
import os
from glob import glob
import cv2

import datasets
import random
import numpy as np
from LiLTfinetune.data.utils import normalize_bbox

def check_normal(value):
    if value < 0:
        return 0
    elif value >= 1000:
        return 1000
    return value

def normalize_bbox(bbox, size):
    b0 = check_normal(int(1000 * bbox[0] / size[0]))
    b1 = check_normal(int(1000 * bbox[1] / size[1]))
    b2 = check_normal(int(1000 * bbox[2] / size[0]))
    b3 = check_normal(int(1000 * bbox[3] / size[1]))
    return [
        min(b0, b2),
        min(b1, b3),
        max(b0, b2),
        max(b1, b3)
    ]

def read_json(fname):
    n_to_lb = {
       0: "其他",
       1: "年级",
       2: "科目",
       3: "学校",
       4: "考试时间",
       5: "班级",
       6: "姓名",
       7: "考号",
       8: "分数",
       9: "座号",
       10: "学号",
       11: "准考证号"
    }
    with open(fname, "r", encoding="utf-8") as fd:
        ls = json.load(fd)
    if 'regions' in ls.keys():
        size = ls['size']
        ls = ls['regions']
    else:
        img = fname.replace("label/", "image/").replace(".txt", ".jpg")    
        size = cv2.imread(img).shape[:2]

#    size = (ls['size'][1], ls['size'][0])
    tokens = []
    ner_tags = []
    bboxes = []

    for i,r in ls.items():
        text = r['string']
        if len(text) == 0:
            continue

        if len(r['box']) == 8:
            box = (r['box'][0], r['box'][1], r['box'][4], r['box'][5])
        else:
            box = r['box']
        box = normalize_bbox(box, size)

        if box[0] < 0 or box[0] > 1000 \
            or box[1] < 0 or box[1] > 1000 \
            or box[2] < 0 or box[2] > 1000 \
            or box[3] < 0 or box[3] > 1000:
            print(box, fname)
        
        last_label = -1
        for st, lb in zip(r['string'], r['tag']):
            if lb == last_label:
                prf = 'I-'
            else:
                prf = 'B-'
                last_lb = lb
            bboxes.append(box)
            ner_tags.append(prf + n_to_lb[lb])
            tokens.append(st)
        continue
    return tokens, ner_tags, bboxes, np.zeros((3,224,224))

class NBIDConfig(datasets.BuilderConfig):
    # BuilderConfig for NBID.

    def __init__(self, n_augs=0, **kwargs):
        """
        BuilderConfig for NBID.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        self.n_augs = n_augs
        super(NBIDConfig, self).__init__(**kwargs)

class NBID(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        NBIDConfig(name=f"ephoie aug 0 loo {i}", n_augs=0, version=datasets.Version("2.0.0"), description="EPHOIE dataset")
            for i in range(1,312)
    ]
    BUILDER_CONFIGS += [
        NBIDConfig(name=f"ephoie aug 1 loo {i}", n_augs=1, version=datasets.Version("2.0.0"), description="EPHOIE dataset")
            for i in range(1, 312)
    ]
    BUILDER_CONFIGS += [
        NBIDConfig(name=f"ephoie aug 1 part {i}", n_augs=1, version=datasets.Version("2.0.0"), description="EPHOIE dataset")
            for i in range(1, 11)
    ]
    BUILDER_CONFIGS += [
        NBIDConfig(name=f"ephoie aug 2 part {i}", n_augs=2, version=datasets.Version("2.0.0"), description="EPHOIE dataset")
            for i in range(1, 11)
    ]
    BUILDER_CONFIGS += [
        NBIDConfig(name=f"ephoie aug 3 part {i}", n_augs=3, version=datasets.Version("2.0.0"), description="EPHOIE dataset")
            for i in range(1, 11)
    ]


    #BUILDER_CONFIGS = [
    #    NBIDConfig(name="ephoie aug 0 part 1", n_augs=0, version=datasets.Version("1.0.0"), description="NBID dataset"),
    #    NBIDConfig(name="ephoie aug 0 part 2", n_augs=0, version=datasets.Version("1.0.0"), description="NBID dataset"),
    #    NBIDConfig(name="ephoie aug 0 part 3", n_augs=0, version=datasets.Version("1.0.0"), description="NBID dataset"),
    #    NBIDConfig(name="ephoie aug 0 part 4", n_augs=0, version=datasets.Version("1.0.0"), description="NBID dataset"),
    #    NBIDConfig(name="ephoie aug 0 part 5", n_augs=0, version=datasets.Version("1.0.0"), description="NBID dataset"),
    #    NBIDConfig(name="ephoie aug 0 part 6", n_augs=0, version=datasets.Version("1.0.0"), description="NBID dataset"),
    #    NBIDConfig(name="ephoie aug 0 part 7", n_augs=0, version=datasets.Version("1.0.0"), description="NBID dataset"),
    #    NBIDConfig(name="ephoie aug 0 part 8", n_augs=0, version=datasets.Version("1.0.0"), description="NBID dataset"),
    #    NBIDConfig(name="ephoie aug 0 part 9", n_augs=0, version=datasets.Version("1.0.0"), description="NBID dataset"),
    #    NBIDConfig(name="ephoie aug 0 part 10", n_augs=0, version=datasets.Version("1.0.0"), description="NBID dataset"),
    #    NBIDConfig(name="ephoie aug 1 part 1", n_augs=1, version=datasets.Version("1.0.0"), description="NBID dataset"),
    #    NBIDConfig(name="ephoie aug 1 part 2", n_augs=1, version=datasets.Version("1.0.0"), description="NBID dataset"),
    #    NBIDConfig(name="ephoie aug 1 part 3", n_augs=1, version=datasets.Version("1.0.0"), description="NBID dataset"),
    #    NBIDConfig(name="ephoie aug 1 part 4", n_augs=1, version=datasets.Version("1.0.0"), description="NBID dataset"),
    #    NBIDConfig(name="ephoie aug 1 part 5", n_augs=1, version=datasets.Version("1.0.0"), description="NBID dataset"),
    #    NBIDConfig(name="ephoie aug 1 part 6", n_augs=1, version=datasets.Version("1.0.0"), description="NBID dataset"),
    #    NBIDConfig(name="ephoie aug 1 part 7", n_augs=1, version=datasets.Version("1.0.0"), description="NBID dataset"),
    #    NBIDConfig(name="ephoie aug 1 part 8", n_augs=1, version=datasets.Version("1.0.0"), description="NBID dataset"),
    #    NBIDConfig(name="ephoie aug 1 part 9", n_augs=1, version=datasets.Version("1.0.0"), description="NBID dataset"),
    #    NBIDConfig(name="ephoie aug 1 part 10", n_augs=1, version=datasets.Version("1.0.0"), description="NBID dataset"),
    #    NBIDConfig(name="ephoie aug 2 part 1", n_augs=2, version=datasets.Version("1.0.0"), description="NBID dataset"),
    #    NBIDConfig(name="ephoie aug 2 part 2", n_augs=2, version=datasets.Version("1.0.0"), description="NBID dataset"),
    #    NBIDConfig(name="ephoie aug 2 part 3", n_augs=2, version=datasets.Version("1.0.0"), description="NBID dataset"),
    #    NBIDConfig(name="ephoie aug 2 part 4", n_augs=2, version=datasets.Version("1.0.0"), description="NBID dataset"),
    #    NBIDConfig(name="ephoie aug 2 part 5", n_augs=2, version=datasets.Version("1.0.0"), description="NBID dataset"),
    #    NBIDConfig(name="ephoie aug 2 part 6", n_augs=2, version=datasets.Version("1.0.0"), description="NBID dataset"),
    #    NBIDConfig(name="ephoie aug 2 part 7", n_augs=2, version=datasets.Version("1.0.0"), description="NBID dataset"),
    #    NBIDConfig(name="ephoie aug 2 part 8", n_augs=2, version=datasets.Version("1.0.0"), description="NBID dataset"),
    #    NBIDConfig(name="ephoie aug 2 part 9", n_augs=2, version=datasets.Version("1.0.0"), description="NBID dataset"),
    #    NBIDConfig(name="ephoie aug 2 part 10", n_augs=2, version=datasets.Version("1.0.0"), description="NBID dataset"),
    #    NBIDConfig(name="ephoie aug 3 part 1", n_augs=3, version=datasets.Version("1.0.0"), description="NBID dataset"),
    #    NBIDConfig(name="ephoie aug 3 part 2", n_augs=3, version=datasets.Version("1.0.0"), description="NBID dataset"),
    #    NBIDConfig(name="ephoie aug 3 part 3", n_augs=3, version=datasets.Version("1.0.0"), description="NBID dataset"),
    #    NBIDConfig(name="ephoie aug 3 part 4", n_augs=3, version=datasets.Version("1.0.0"), description="NBID dataset"),
    #    NBIDConfig(name="ephoie aug 3 part 5", n_augs=3, version=datasets.Version("1.0.0"), description="NBID dataset"),
    #    NBIDConfig(name="ephoie aug 3 part 6", n_augs=3, version=datasets.Version("1.0.0"), description="NBID dataset"),
    #    NBIDConfig(name="ephoie aug 3 part 7", n_augs=3, version=datasets.Version("1.0.0"), description="NBID dataset"),
    #    NBIDConfig(name="ephoie aug 3 part 8", n_augs=3, version=datasets.Version("1.0.0"), description="NBID dataset"),
    #    NBIDConfig(name="ephoie aug 3 part 9", n_augs=3, version=datasets.Version("1.0.0"), description="NBID dataset"),
    #    NBIDConfig(name="ephoie aug 3 part 10", n_augs=3, version=datasets.Version("1.0.0"), description="NBID dataset"),
        #NBIDConfig(name="ephoie imageless 1", n_augs=1, version=datasets.Version("1.0.0"), description="NBID dataset"),
        #NBIDConfig(name="ephoie imageless 2", n_augs=2, version=datasets.Version("1.0.0"), description="NBID dataset"),
        #NBIDConfig(name="ephoie imageless 3", n_augs=3, version=datasets.Version("1.0.0"), description="NBID dataset"),
    #]

    def _info(self):
        return datasets.DatasetInfo(
            description="NBID dataset",
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "B-其他", "I-其他",
                                "B-年级", "I-年级",
                                "B-科目", "I-科目",
                                "B-学校", "I-学校",
                                "B-考试时间", "I-考试时间",
                                "B-班级", "I-班级",
                                "B-姓名", "I-姓名",
                                "B-考号", "I-考号",
                                "B-分数", "I-分数",
                                "B-座号", "I-座号",
                                "B-学号", "I-学号",
                                "B-准考证号", "I-准考证号"
                            ]
                        )
                    ),
                    "image": datasets.Array3D(shape=(3, 224, 224), dtype="uint8"),
                    "fname": datasets.Value("string")
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        # Returns SplitGenerators.
        # downloaded_file = dl_manager.download_and_extract("https://guillaumejaume.github.io/FUNSD/dataset.zip")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": f"EPHOIE/train.txt", "split": "train"}
            ),
            datasets.SplitGenerator(
                name="validation", gen_kwargs={"filepath": f"EPHOIE/train.txt", "split": "valid"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"filepath": f"EPHOIE/test.txt", "split": "test"}
            ),
        ]

    def _generate_examples(self, filepath, split):
        n_augs = self.config.n_augs
        if "loo" in self.config.name:
            pt = int(self.config.name.split(" ")[-1])
            with open(f"EPHOIE/prots/loo_{pt}_aug_{n_augs}.json", "r") as fd:
                js = json.load(fd)
        else:
            pt = int(self.config.name.split(" ")[-1])
            with open(f"EPHOIE/prots/prot_{pt}_aug_{n_augs}.json", "r") as fd:
                js = json.load(fd)
        fs = js[split]
        """
        if 'imageless' not in self.config.name or 'test' in filepath:
            with open(filepath, "r") as fd:
                fs = ["./EPHOIE/label/" + x.strip() + ".txt" for x in fd.readlines()]
        else:
            fs = []
        if self.config.n_augs > 0 and 'test' not in filepath:
            afs = glob("./EPHOIE/ephoie_augmented/*")[:1200*self.config.n_augs]
        else:
            afs = []
        """
        #fs = glob(f"{filepath}/*.json")
        #random.shuffle(fs)
        
        #fs = sorted(os.listdir(filepath))
        #random.shuffle(fs)

        if split == "train" or split == "test":
            fs = [fs[0]]
        for i, f in enumerate(fs):
            #ff = filepath + "/" + f
            tokens, ner_tags, bboxes, image = read_json(f)
            yield i, {"id": str(i), "tokens": tokens, "bboxes": bboxes, "ner_tags": ner_tags, "image": image, "fname": f}

