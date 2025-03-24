# coding=utf-8

import json
import os
from glob import glob

import datasets
import random

def check_normal(value):
    if value < 0:
        return 0
    elif value > 1000:
        return 1000
    return value

def normalize_bbox(bbox, size):
    b0 = int(1000 * bbox[0] / size[0])
    b1 = int(1000 * bbox[1] / size[1])
    b2 = int(1000 * bbox[2] / size[0])
    b3 = int(1000 * bbox[3] / size[1])
    return [
        check_normal(b0),
        check_normal(b1),
        check_normal(b2),
        check_normal(b3)
    ]

def read_json(fname):
    with open(fname, "r", encoding="utf-8") as fd:
        ls = json.load(fd)
    
    size = (ls['size'][1], ls['size'][0])
    if "node_type" in ls.keys():
        node_type = ls['node_type']
    else:
        node_type = "other"
    tokens = []
    ner_tags = []
    bboxes = []

    if type(ls['regions']) == dict:
        regs = [{'tag': i, 'box': v['box'], 'text': v['text']} for i,v in ls['regions'].items()]
    else:
        regs = ls['regions']
    
    for reg in regs:
        label = reg['tag']
        text = reg['text']
        if len(text) == 0:
            continue

        box = reg['box']

        box = normalize_bbox([int(box[0][0]),
                              int(box[0][1]),
                              int(box[1][0]),
                              int(box[1][1])],
                              size)
        if box[0] < 0 or box[0] > 1000 \
            or box[1] < 0 or box[1] > 1000 \
            or box[2] < 0 or box[2] > 1000 \
            or box[3] < 0 or box[3] > 1000:
            print(box, fname)
        box[0], box[2] = min(box[0], box[2]), max(box[0], box[2])
        box[1], box[3] = min(box[1], box[3]), max(box[1], box[3])

        wd_per_char = 1
        box_width = box[2] - box[0]
        words = [w for w in text.split(" ") if w.strip() != ""]
        new_box = box

        n_chars = 0
        if label == "other":
            for w in words:
                tokens.append(w)
                ner_tags.append("O")
                print("Added O")
                new_box = [int(box[0] + wd_per_char*n_chars),
                           box[1],
                           int(box[2] + wd_per_char*(n_chars + len(w))),
                           box[3]]
                n_chars += len(w) + 1
                new_box = box
                bboxes.append(new_box)
        else:
            tokens.append(words[0])
            ner_tags.append("B-" + label.upper())

            new_box = [int(box[0] + wd_per_char*n_chars),
                        box[1],
                        int(box[0] + wd_per_char*(n_chars + len(words[0]))),
                        box[3]]
            n_chars += len(words[0]) + 1
            new_box = box
            bboxes.append(new_box)

            for w in words[1:]:
                tokens.append(w)
                ner_tags.append("B-" + label.upper())

                new_box = [int(box[0] + wd_per_char*n_chars),
                            box[1],
                            int(box[0] + wd_per_char*(n_chars + len(w))),
                            box[3]]
                n_chars += len(w) + 1
                new_box = box
                bboxes.append(new_box)

    return tokens, ner_tags, bboxes, node_type

class NBIDConfig(datasets.BuilderConfig):
    # BuilderConfig for NBID.

    def __init__(self, **kwargs):
        """
        BuilderConfig for NBID.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """

        super(NBIDConfig, self).__init__(**kwargs)

class NBID(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        NBIDConfig(name="inbid", version=datasets.Version("1.0.0"), description="NBID dataset"),
    ]

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
                                # FRONT
                                "B-NOME", "I-NOME",
                                "B-FILIACAO1", "I-FILIACAO1",
                                "B-FILIACAO2", "I-FILIACAO2",
                                "B-DATANASC", "I-DATANASC",
                                "B-NATURALIDADE", "I-NATURALIDADE",
                                "B-ORGAOEXP", "I-ORGAOEXP",
                                "B-SERIAL", "I-SERIAL",
                                "B-CODSEC", "I-CODSEC",
                                "B-RH", "I-RH",
                                "B-OBS", "I-OBS",
                                
                                # Back
                                "B-RG", "I-RG",
                                "B-DATAEXP", "I-DATAEXP",
                                "B-CPF", "I-CPF",
                                "B-REGCIVIL", "I-REGCIVIL",
                                "B-DNI", "I-DNI",
                                "B-TE", "I-TE",
                                "B-CTPS", "I-CTPS",
                                "B-SERIE", "I-SERIE",
                                "B-UF", "I-UF",
                                "B-PIS", "I-PIS",
                                "B-PROFISSIONAL", "I-PROFISSIONAL",
                                "B-CNH", "I-CNH",
                                "B-CNS", "I-CNS",
                                "B-MILITAR", "I-MILITAR",

                                # Other
                                "O"
                            ]
                        )
                    ),
                    "node_type": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        # Returns SplitGenerators.
        # downloaded_file = dl_manager.download_and_extract("https://guillaumejaume.github.io/FUNSD/dataset.zip")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": f"ibid/train/"}
            ),
            datasets.SplitGenerator(
                name="validation", gen_kwargs={"filepath": f"ibid/valid/"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"filepath": f"ibid/test/"}
            ),
        ]

    def _generate_examples(self, filepath):

        fs = glob(f"{filepath}/*.json")
        random.shuffle(fs)
        
        fs = sorted(os.listdir(filepath))
        #random.shuffle(fs)

        for i, f in enumerate(fs):
            ff = filepath + "/" + f
        #for i,f in enumerate(fs):
            tokens, ner_tags, bboxes, node_type = read_json(ff)
            #print(i, tokens, bboxes, ner_tags)
            yield i, {"id": str(i), "tokens": tokens, "bboxes": bboxes, "ner_tags": ner_tags, "node_type": node_type}



# def nbid_generator():
#     ftrain = "./nbid/prot80/front_train.txt"

#     filenames = []
#     with open(ftrain, "r") as fd:
#         filenames = [x.split(",")[-1].strip() for x in fd.readlines()]
    
#     for i, f in enumerate(filenames):
#         with open("./nbid/front/" + f) as fd:
#             ls = fd.readlines()
#         tokens = []
#         bboxes = []
#         ner_tags = []

#         for line in ls:
#             decoded = line.split(',')

#             label = decoded[-1]
#             text = decoded[-2]

#             words = [w for w in text.split(" ") if w.strip() != ""]
#             if label == "other":
#                 for w in words:
#                     tokens.append(w)
#                     ner_tags.append("O")
#                     bboxes.append(bbox)
#             else:
#                 tokens.append(words[0])
#                 ner_tags.append("B-" + label.upper())
#                 bboxes.append(bbox)

#                 for w in words[1:]:
#                     tokens.append(w)
#                     ner_tags.append("I-" + label.upper())
#                     bboxes.append(bbox)

#         yield i, {"id": str(i), "tokens": tokens, "bboxes": bboxes, "ner_tags": ner_tags}

# dts = NBID()
# # datasets = datasets.load_dataset(os.path.abspath(dts.__file__))
# # dts = datasets.Dataset.from_generator(nbid_generator)
# print(dts)

"""
DatasetDict({
    train: Dataset({
        features: ['id', 'tokens', 'bboxes', 'ner_tags'],
        num_rows: 149
    })
    test: Dataset({
        features: ['id', 'tokens', 'bboxes', 'ner_tags'],
        num_rows: 50
    })
})
"""
