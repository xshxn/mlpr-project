"""Librispeech automatic speech recognition dataset."""
import csv
import os

import datasets
from datasets.tasks import AutomaticSpeechRecognition

from huggingface_hub import list_repo_files


import pyarrow.parquet as pq
import pyarrow as pa


_CITATION = """\
@inproceedings{panayotov2015librispeech,
  title={Librispeech: an ASR corpus based on public domain audio books},
  author={Panayotov, Vassil and Chen, Guoguo and Povey, Daniel and Khudanpur, Sanjeev},
  booktitle={Acoustics, Speech and Signal Processing (ICASSP), 2015 IEEE International Conference on},
  pages={5206--5210},
  year={2015},
  organization={IEEE}
}
"""

_DESCRIPTION = """\
LibriSpeech is a corpus of approximately 1000 hours of read English speech with sampling rate of 16 kHz,
prepared by Vassil Panayotov with the assistance of Daniel Povey. The data is derived from read
audiobooks from the LibriVox project, and has been carefully segmented and aligned.87
"""

_URL = "http://www.openslr.org/12"
_TRANSCRIPT_URL =  "https://huggingface.co/datasets/distil-whisper/whisper_transcriptions_greedy/resolve/main/librispeech_asr/"

_DATA_REPO_ID = "sanchit-gandhi/librispeech-data"

_TRANSCRIPT_URLS = {
    "clean": {
        "dev": _TRANSCRIPT_URL + "validation-clean-transcription.csv",
        "test": _TRANSCRIPT_URL + "test-clean-transcription.csv",
        "train.100": _TRANSCRIPT_URL + "train-clean-100-transcription.csv",
        "train.360": _TRANSCRIPT_URL + "train-clean-360-transcription.csv",
    },
    "other": {
        "test": _TRANSCRIPT_URL + "test-other-transcription.csv",
        "dev": _TRANSCRIPT_URL + "validation-other-transcription.csv",
        "train.500": _TRANSCRIPT_URL + "train-other-500-transcription.csv",
    },
    "all": {
        "dev.clean": _TRANSCRIPT_URL + "validation-clean-transcription.csv",
        "dev.other": _TRANSCRIPT_URL + "validation-other-transcription.csv",
        "test.clean": _TRANSCRIPT_URL + "test-clean-transcription.csv",
        "test.other": _TRANSCRIPT_URL + "test-other-transcription.csv",
        "train.clean.100": _TRANSCRIPT_URL + "train-clean-100-transcription.csv",
        "train.clean.360": _TRANSCRIPT_URL + "train-clean-360-transcription.csv",
        "train.other.500": _TRANSCRIPT_URL + "train-other-500-transcription.csv",
    },
}


class LibrispeechASRConfig(datasets.BuilderConfig):
    """BuilderConfig for LibriSpeechASR."""

    def __init__(self, **kwargs):
        """
        Args:
          data_dir: `string`, the path to the folder containing the files in the
            downloaded .tar
          citation: `string`, citation for the data set
          url: `string`, url for information about the data set
          **kwargs: keyword arguments forwarded to super.
        """
        super(LibrispeechASRConfig, self).__init__(version=datasets.Version("2.1.0", ""), **kwargs)


class LibriSpeechASR(datasets.ArrowBasedBuilder):
    """Librispeech dataset."""

    DEFAULT_WRITER_BATCH_SIZE = 256
    DEFAULT_CONFIG_NAME = "all"
    BUILDER_CONFIGS = [
        LibrispeechASRConfig(name="clean", description="'Clean' speech."),
        LibrispeechASRConfig(name="other", description="'Other', more challenging, speech."),
        LibrispeechASRConfig(name="all", description="Combined clean and other dataset."),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "file": datasets.Value("string"),
                    "audio": datasets.Audio(sampling_rate=16_000),
                    "text": datasets.Value("string"),
                    "speaker_id": datasets.Value("int64"),
                    "chapter_id": datasets.Value("int64"),
                    "id": datasets.Value("string"),
                    "whisper_transcript": datasets.Value("string"),
                }
            ),
            supervised_keys=("file", "text"),
            homepage=_URL,
            citation=_CITATION,
            task_templates=[AutomaticSpeechRecognition(audio_column="audio", transcription_column="text")],
        )

    def _split_generators(self, dl_manager):
        data_repo_download = f"https://huggingface.co/datasets/{_DATA_REPO_ID}/resolve/main/"
        all_files = list_repo_files(_DATA_REPO_ID, repo_type="dataset")

        train_clean_100_files = [file for file in all_files if file.startswith("data/train.clean.100")]
        train_clean_360_files = [file for file in all_files if file.startswith("data/train.clean.360")]
        train_other_500_files = [file for file in all_files if file.startswith("data/train.other.500")]
        validation_clean_files = [file for file in all_files if file.startswith("data/validation.clean")]
        validation_other_files = [file for file in all_files if file.startswith("data/validation.other")]
        test_clean_files = [file for file in all_files if file.startswith("data/test.clean")]
        test_other_files = [file for file in all_files if file.startswith("data/test.other")]

        split_to_ids = {
            "train.clean.100": train_clean_100_files,
            "train.clean.360": train_clean_360_files,
            "train.other.500": train_other_500_files,
            "dev.clean": validation_clean_files,
            "dev.other": validation_other_files,
            "test.clean": test_clean_files,
            "test.other": test_other_files,
        }

        dl_urls = {}
        for split, split_ids in split_to_ids.items():
            dl_urls[split] = [data_repo_download + source_id for source_id in split_ids]
        archive_paths = dl_manager.download(dl_urls)

        local_extracted_archive_paths = (
            dl_manager.extract(archive_paths)
            if not dl_manager.is_streaming
            else {split: [None] * len(archive_paths[split]) for split in split_to_ids}
        )

        transcript_archive_path = dl_manager.download(_TRANSCRIPT_URLS[self.config.name])
        # (Optional) In non-streaming mode, we can extract the archive locally to have actual local transcription files:
        # local_extracted_transcript_archive = dl_manager.extract(transcript_archive_path) if not dl_manager.is_streaming else {}

        if self.config.name == "clean":
            train_splits = [
                datasets.SplitGenerator(
                    name="train.100",
                    gen_kwargs={
                        "local_extracted_archive_paths": local_extracted_archive_paths.get("train.clean.100"),
                        "archives": [dl_manager.iter_files(path) for path in archive_paths["train.clean.100"]],
                        #"local_extracted_transcript_archive": local_extracted_transcript_archive.get("train.100"),
                        "transcript_files": transcript_archive_path["train.100"],
                    },
                ),
                datasets.SplitGenerator(
                    name="train.360",
                    gen_kwargs={
                        "local_extracted_archive_paths": local_extracted_archive_paths.get("train.360"),
                        "archives": [dl_manager.iter_files(path) for path in archive_paths["train.360"]],
                        #"local_extracted_transcript_archive": local_extracted_transcript_archive.get("train.360"),
                        "transcript_files": transcript_archive_path["train.360"],
                    },
                ),
            ]
            dev_splits = [
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "local_extracted_archive_paths": local_extracted_archive_paths.get("dev"),
                        "archives": [dl_manager.iter_files(path) for path in archive_paths["dev"]],
                        #"local_extracted_transcript_archive": local_extracted_transcript_archive.get("dev"),
                        "transcript_files": transcript_archive_path["dev"],
                    },
                )
            ]
            test_splits = [
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "local_extracted_archive_paths": local_extracted_archive_paths.get("test"),
                        "archives": [dl_manager.iter_files(path) for path in archive_paths["test"]],
                        #"local_extracted_transcript_archive": local_extracted_transcript_archive.get("test"),
                        "transcript_files": transcript_archive_path["test"],
                    },
                )
            ]
        elif self.config.name == "other":
            train_splits = [
                datasets.SplitGenerator(
                    name="train.500",
                    gen_kwargs={
                        "local_extracted_archive_paths": local_extracted_archive_paths.get("train.500"),
                        "archives": [dl_manager.iter_files(path) for path in archive_paths["train.500"]],
                        #"local_extracted_transcript_archive": local_extracted_transcript_archive.get("train.500"),
                        "transcript_files": transcript_archive_path["train.500"],
                    },
                )
            ]
            dev_splits = [
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "local_extracted_archive_paths": local_extracted_archive_paths.get("dev"),
                        "archives": [dl_manager.iter_files(path) for path in archive_paths["dev"]],
                        #"local_extracted_transcript_archive": local_extracted_transcript_archive.get("dev"),
                        "transcript_files": transcript_archive_path["dev"],
                    },
                )
            ]
            test_splits = [
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "local_extracted_archive_paths": local_extracted_archive_paths.get("test"),
                        "archives": [dl_manager.iter_files(path) for path in archive_paths["test"]],
                        #"local_extracted_transcript_archive": local_extracted_transcript_archive.get("test"),
                        "transcript_files": transcript_archive_path["test"],
                    },
                )
            ]
        elif self.config.name == "all":
            train_splits = [
                datasets.SplitGenerator(
                    name="train.clean.100",
                    gen_kwargs={
                        "local_extracted_archive_paths": local_extracted_archive_paths.get("train.clean.100"),
                        "archives": [dl_manager.iter_files(path) for path in archive_paths["train.clean.100"]],
                        #"local_extracted_transcript_archive": local_extracted_transcript_archive.get("train.clean.100"),
                        "transcript_files": transcript_archive_path["train.clean.100"],
                    },
                ),
                datasets.SplitGenerator(
                    name="train.clean.360",
                    gen_kwargs={
                        "local_extracted_archive_paths": local_extracted_archive_paths.get("train.clean.360"),
                        "archives": [dl_manager.iter_files(path) for path in archive_paths["train.clean.360"]],
                        #"local_extracted_transcript_archive": local_extracted_transcript_archive.get("train.clean.360"),
                        "transcript_files": transcript_archive_path["train.clean.360"],
                    },
                ),
                datasets.SplitGenerator(
                    name="train.other.500",
                    gen_kwargs={
                        "local_extracted_archive_paths": local_extracted_archive_paths.get("train.other.500"),
                        "archives": [dl_manager.iter_files(path) for path in archive_paths["train.other.500"]],
                        #"local_extracted_transcript_archive": local_extracted_transcript_archive.get("train.other.500"),
                        "transcript_files": transcript_archive_path["train.other.500"],
                    },
                ),
            ]
            dev_splits = [
                datasets.SplitGenerator(
                    name="validation.clean",
                    gen_kwargs={
                        "local_extracted_archive_paths": local_extracted_archive_paths.get("dev.clean"),
                        "archives": [dl_manager.iter_files(path) for path in archive_paths["dev.clean"]],
                        #"local_extracted_transcript_archive": local_extracted_transcript_archive.get("dev.clean"),
                        "transcript_files": transcript_archive_path["dev.clean"],
                    },
                ),
                datasets.SplitGenerator(
                    name="validation.other",
                    gen_kwargs={
                        "local_extracted_archive_paths": local_extracted_archive_paths.get("dev.other"),
                        "archives": [dl_manager.iter_files(path) for path in archive_paths["dev.other"]],
                        #"local_extracted_transcript_archive": local_extracted_transcript_archive.get("dev.other"),
                        "transcript_files": transcript_archive_path["dev.other"],
                    },
                ),
            ]
            test_splits = [
                datasets.SplitGenerator(
                    name="test.clean",
                    gen_kwargs={
                        "local_extracted_archive_paths": local_extracted_archive_paths.get("test.clean"),
                        "archives": [dl_manager.iter_files(path) for path in archive_paths["test.clean"]],
                        #"local_extracted_transcript_archive": local_extracted_transcript_archive.get("test.clean"),
                        "transcript_files": transcript_archive_path["test.clean"],
                    },
                ),
                datasets.SplitGenerator(
                    name="test.other",
                    gen_kwargs={
                        "local_extracted_archive_paths": local_extracted_archive_paths.get("test.other"),
                        "archives": [dl_manager.iter_files(path) for path in archive_paths["test.other"]],
                        #"local_extracted_transcript_archive": local_extracted_transcript_archive.get("test.other"),
                        "transcript_files": transcript_archive_path["test.other"],
                    },
                ),
            ]

        return train_splits + dev_splits + test_splits

    def _generate_tables(self, local_extracted_archive_paths, archives, transcript_files):
        whisper_transcriptions = dict()
        with open(transcript_files, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=",")
            for line in reader:
                whisper_transcriptions[line["file_id"]] = line["whisper_transcript"]

        idx = 0
        for local_extracted_archive_path, archive in zip(local_extracted_archive_paths, archives):
            # Here we iterate over all the files within the TAR archive:
            for audio_file in archive:
                with open(audio_file, "rb") as f:
                    pf = pq.ParquetFile(f)
                    for record_batch in pf.iter_batches():
                        pa_table = pa.Table.from_batches([record_batch])
                    whisper_transcript = [whisper_transcriptions.get(str(file_id), None) for file_id in pa_table["id"]]
                    whisper_transcript = pa.array(whisper_transcript, pa.string())
                    pa_table = pa_table.append_column("whisper_transcript", whisper_transcript)
                    yield idx, pa_table
                    idx += 1