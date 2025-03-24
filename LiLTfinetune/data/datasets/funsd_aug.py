# coding=utf-8
import random

import json
import os

# Imported by me
from glob import glob
import numpy as np

import datasets

from LiLTfinetune.data.utils import load_image, normalize_bbox


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@article{Jaume2019FUNSDAD,
  title={FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents},
  author={Guillaume Jaume and H. K. Ekenel and J. Thiran},
  journal={2019 International Conference on Document Analysis and Recognition Workshops (ICDARW)},
  year={2019},
  volume={2},
  pages={1-6}
}
"""

_DESCRIPTION = """\
https://guillaumejaume.github.io/FUNSD/
"""


class FunsdConfig(datasets.BuilderConfig):
    """BuilderConfig for FUNSD"""

    def __init__(self, n_augs=0, **kwargs):
        """BuilderConfig for FUNSD.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        self.n_augs=n_augs
        super(FunsdConfig, self).__init__(**kwargs)


class Funsd(datasets.GeneratorBasedBuilder):
    """Conll2003 dataset."""

    BUILDER_CONFIGS = [
        FunsdConfig(name=f"funsd aug 0 loo {i}", n_augs=0, version=datasets.Version("2.0.0"), description="FUNSD dataset")
		    for i in range(1,51)
    ]
    BUILDER_CONFIGS += [
        FunsdConfig(name=f"funsd aug 1 loo {i}", n_augs=1, version=datasets.Version("2.0.0"), description="FUNSD dataset")
            for i in range(1, 51)

    #]
        #FunsdConfig(name="funsd aug 0 part 1", n_augs=0, version=datasets.Version("2.0.0"), description="FUNSD dataset"),
        #FunsdConfig(name="funsd aug 0 part 2", n_augs=0, version=datasets.Version("2.0.0"), description="FUNSD dataset"),
        #FunsdConfig(name="funsd aug 0 part 3", n_augs=0, version=datasets.Version("2.0.0"), description="FUNSD dataset"),
        #FunsdConfig(name="funsd aug 0 part 4", n_augs=0, version=datasets.Version("2.0.0"), description="FUNSD dataset"),
        #FunsdConfig(name="funsd aug 0 part 5", n_augs=0, version=datasets.Version("2.0.0"), description="FUNSD dataset"),
        #FunsdConfig(name="funsd aug 1 part 1", n_augs=1, version=datasets.Version("2.0.0"), description="FUNSD dataset"),
        #FunsdConfig(name="funsd aug 1 part 2", n_augs=1, version=datasets.Version("2.0.0"), description="FUNSD dataset"),
        #FunsdConfig(name="funsd aug 1 part 3", n_augs=1, version=datasets.Version("2.0.0"), description="FUNSD dataset"),
        #FunsdConfig(name="funsd aug 1 part 4", n_augs=1, version=datasets.Version("2.0.0"), description="FUNSD dataset"),
        #FunsdConfig(name="funsd aug 1 part 5", n_augs=1, version=datasets.Version("2.0.0"), description="FUNSD dataset"),
        #FunsdConfig(name="funsd aug 2 part 1", n_augs=2, version=datasets.Version("2.0.0"), description="FUNSD dataset"),
        #FunsdConfig(name="funsd aug 2 part 2", n_augs=2, version=datasets.Version("2.0.0"), description="FUNSD dataset"),
        #FunsdConfig(name="funsd aug 2 part 3", n_augs=2, version=datasets.Version("2.0.0"), description="FUNSD dataset"),
        #FunsdConfig(name="funsd aug 2 part 4", n_augs=2, version=datasets.Version("2.0.0"), description="FUNSD dataset"),
        #FunsdConfig(name="funsd aug 2 part 5", n_augs=2, version=datasets.Version("2.0.0"), description="FUNSD dataset"),
        #FunsdConfig(name="funsd aug 3 part 1", n_augs=3, version=datasets.Version("2.0.0"), description="FUNSD dataset"),
        #FunsdConfig(name="funsd aug 3 part 2", n_augs=3, version=datasets.Version("2.0.0"), description="FUNSD dataset"),
        #FunsdConfig(name="funsd aug 3 part 3", n_augs=3, version=datasets.Version("2.0.0"), description="FUNSD dataset"),
        #FunsdConfig(name="funsd aug 3 part 4", n_augs=3, version=datasets.Version("2.0.0"), description="FUNSD dataset"),
        #FunsdConfig(name="funsd aug 3 part 5", n_augs=3, version=datasets.Version("2.0.0"), description="FUNSD dataset"),
        #FunsdConfig(name="funsd aug 4 part 1", n_augs=4, version=datasets.Version("2.0.0"), description="FUNSD dataset"),
        #FunsdConfig(name="funsd aug 4 part 2", n_augs=4, version=datasets.Version("2.0.0"), description="FUNSD dataset"),
        #FunsdConfig(name="funsd aug 4 part 3", n_augs=4, version=datasets.Version("2.0.0"), description="FUNSD dataset"),
        #FunsdConfig(name="funsd aug 4 part 4", n_augs=4, version=datasets.Version("2.0.0"), description="FUNSD dataset"),
        #FunsdConfig(name="funsd aug 4 part 5", n_augs=4, version=datasets.Version("2.0.0"), description="FUNSD dataset"),
        #FunsdConfig(name="funsd aug 5 part 1", n_augs=5, version=datasets.Version("2.0.0"), description="FUNSD dataset"),
        #FunsdConfig(name="funsd aug 5 part 2", n_augs=5, version=datasets.Version("2.0.0"), description="FUNSD dataset"),
        #FunsdConfig(name="funsd aug 5 part 3", n_augs=5, version=datasets.Version("2.0.0"), description="FUNSD dataset"),
        #FunsdConfig(name="funsd aug 5 part 4", n_augs=5, version=datasets.Version("2.0.0"), description="FUNSD dataset"),
        #FunsdConfig(name="funsd aug 5 part 5", n_augs=5, version=datasets.Version("2.0.0"), description="FUNSD dataset"),
        #FunsdConfig(name="funsd aug 10", n_augs=10, version=datasets.Version("2.0.0"), description="FUNSD dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=["O", "B-HEADER", "I-HEADER", "B-QUESTION", "I-QUESTION", "B-ANSWER", "I-ANSWER"]
                        )
                    ),
                    "image": datasets.Array3D(shape=(3, 224, 224), dtype="uint8"),
                }
            ),
            supervised_keys=None,
            homepage="https://guillaumejaume.github.io/FUNSD/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        #downloaded_file = dl_manager.download_and_extract("https://guillaumejaume.github.io/FUNSD/dataset.zip")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": f"./augmentsv3/all_train/", "split": "train", "run_type": "loo"}
            ),
            datasets.SplitGenerator(
                name="validation", gen_kwargs={"filepath": f"./augmentsv3/all_train/", "split": "valid", "run_type": "loo"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"filepath": f"./augmentsv3/test/", "split": "test", "run_type": "loo"}
            ),
        ]

    def _generate_examples(self, filepath, split, run_type="loo"):
        if "loo" in self.config.name:
            pt = int(self.config.name.split(" ")[-1])
        n_augs = self.config.n_augs
        #with open(f"augmentsv3/prots/prot_{pt}_aug_{n_augs}.json", "r") as fd:
        with open(f"augmentsv3/prots/loo_{pt}_aug_{n_augs}.json", "r") as fd:
            js = json.load(fd)
        fs = js[split]
        #if len(fs) > 1:
        random.shuffle(fs)

        #fs = []
        #print(filepath)
        #if 'train' in filepath:
        #    basefs = sorted(glob(f"{filepath}/*_4.json"))
            #print(basefs)
        #    for f in basefs:
        #        bs = f[:-7]
        #        fs += [bs + ".json"] + [f"{bs}_{n}.json" for n in range(0, n_augs)]
        #else:
        #    fs = sorted(glob(f"{filepath}/*.json"))

        #logger.info("⏳ Generating examples from = %s", filepath)
        #ann_dir = os.path.join(filepath, "annotations")
        #img_dir = os.path.join(filepath, "images")
        for guid, file in enumerate(fs):
            tokens = []
            bboxes = []
            ner_tags = []
            file_path = file
            #file_path = os.path.join(ann_dir, file)
            with open(file_path, "r", encoding="utf8") as f:
                data = json.load(f)
            if 'form' in data.keys():
                image_path = file_path.replace("json", "png")
                image, size = load_image(image_path)
            else:
                size = data['size']
                image = np.zeros((3,224,224))
                data['form'] = data['regions']
            for item in data["form"]:
                words, label = item["words"], item['label']
                words = [w for w in words if w["text"].strip() != ""]
                if len(words) == 0:
                    continue
                if label == "other":
                    for w in words:
                        tokens.append(w["text"])
                        ner_tags.append("O")
                        bboxes.append(normalize_bbox(item["box"], size))
                else:
                    tokens.append(words[0]["text"])
                    ner_tags.append("B-" + label.upper())
                    bboxes.append(normalize_bbox(item["box"], size))
                    for w in words[1:]:
                        tokens.append(w["text"])
                        ner_tags.append("I-" + label.upper())
                        bboxes.append(normalize_bbox(item["box"], size))

            yield guid, {"id": str(guid), "tokens": tokens, "bboxes": bboxes, "ner_tags": ner_tags, "image": image}
