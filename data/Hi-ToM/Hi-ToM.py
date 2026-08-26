import json
from datasets import DatasetBuilder, DatasetInfo, SplitGenerators, DownloadManager
from datasets.features import Features, Value
import datasets

class MyCustomDataset(DatasetBuilder):
    VERSION = datasets.Version("1.0.0")

    def _info(self):
        # Ensure CITATION is a string containing the bibliographic citation for your dataset

        return DatasetInfo(
            description="My custom dataset for tracking objects.",
            features=Features({
                "prompting_type": Value("string"),
                "deception": Value("bool"),
                "story_length": Value("int32"),
                "question_order": Value("int32"),
                "sample_id": Value("int32"),
                "story": Value("string"),
                "question": Value("string"),
                "choices": Value("string"),
                "answer": Value("string"),
            }),
            supervised_keys=None,
            homepage="https://github.com/ying-hui-he/Hi-ToM_dataset",
        )

    def _split_generators(self, dl_manager: DownloadManager):
        # Using the raw content URL for GitHub
        downloaded_files = dl_manager.download_and_extract({
            "data_file": "https://raw.githubusercontent.com/ying-hui-he/Hi-ToM_dataset/main/Hi-ToM_data.json"
        })

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": downloaded_files["data_file"],
                },
            ),
        ]

    def _generate_examples(self, filepath):
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)["data"]  # Accessing the 'data' field directly
            for id, item in enumerate(data):
                yield id, {
                    "prompting_type": item["prompting_type"],
                    "deception": item["deception"],
                    "story_length": item["story_length"],
                    "question_order": item["question_order"],
                    "sample_id": item["sample_id"],
                    "story": item["story"],
                    "question": item["question"],
                    "choices": item["choices"],
                    "answer": item["answer"],
                }
# Replace 'MyCustomDataset' with a suitable name that follows Python class naming conventions.
