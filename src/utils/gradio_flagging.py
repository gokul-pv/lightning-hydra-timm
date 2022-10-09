from __future__ import annotations

import csv
import datetime
import io
import json
import os
import uuid
from typing import TYPE_CHECKING, Any, List, Optional

import gradio as gr
from gradio import FlaggingCallback, encryptor, utils

if TYPE_CHECKING:
    from gradio.components import IOComponent

import boto3

from src.utils import pylogger

log = pylogger.get_pylogger(__name__)


class CSVLoggerS3(FlaggingCallback):
    """The default implementation of the FlaggingCallback abstract class.

    Each flagged
    sample (both the input and output data) is logged to a CSV file with headers on the machine running the gradio app and to S3.
    Example:
        import gradio as gr
        def image_classifier(inp):
            return {'cat': 0.3, 'dog': 0.7}
        demo = gr.Interface(fn=image_classifier, inputs="image", outputs="label",
                            flagging_callback=CSVLogger())
    Guides: using_flagging
    """

    def __init__(self, s3_dir: str, to_s3: bool = False):
        self.to_s3 = to_s3
        self.dir = s3_dir

    def setup(
        self,
        components: list[IOComponent],
        flagging_dir: str,
        encryption_key: str | None = None,
    ):
        self.components = components
        self.flagging_dir = flagging_dir
        self.encryption_key = encryption_key
        os.makedirs(flagging_dir, exist_ok=True)

    def flag(
        self,
        flag_data: list[Any],
        flag_option: str | None = None,
        flag_index: int | None = None,
        username: str | None = None,
    ) -> int:
        flagging_dir = self.flagging_dir
        log_filepath = os.path.join(flagging_dir, "log.csv")
        is_new = not os.path.exists(log_filepath)

        if flag_index is None:
            csv_data = []
            for idx, (component, sample) in enumerate(zip(self.components, flag_data)):
                save_dir = os.path.join(
                    flagging_dir,
                    utils.strip_invalid_filename_characters(component.label or f"component {idx}"),
                )
                csv_data.append(
                    component.deserialize(
                        sample,
                        save_dir=save_dir,
                        encryption_key=self.encryption_key,
                    )
                    if sample is not None
                    else ""
                )
            csv_data.append(flag_option if flag_option is not None else "")
            csv_data.append(username if username is not None else "")
            csv_data.append(str(datetime.datetime.now()))
            if is_new:
                headers = [
                    component.label or f"component {idx}"
                    for idx, component in enumerate(self.components)
                ] + [
                    "flag",
                    "username",
                    "timestamp",
                ]

        def replace_flag_at_index(file_content):
            file_content = io.StringIO(file_content)
            content = list(csv.reader(file_content))
            header = content[0]
            flag_col_index = header.index("flag")
            content[flag_index][flag_col_index] = flag_option
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerows(utils.sanitize_list_for_csv(content))
            return output.getvalue()

        if self.encryption_key:
            output = io.StringIO()
            if not is_new:
                with open(log_filepath, "rb", encoding="utf-8") as csvfile:
                    encrypted_csv = csvfile.read()
                    decrypted_csv = encryptor.decrypt(self.encryption_key, encrypted_csv)
                    file_content = decrypted_csv.decode()
                    if flag_index is not None:
                        file_content = replace_flag_at_index(file_content)
                    output.write(file_content)
            writer = csv.writer(output)
            if flag_index is None:
                if is_new:
                    writer.writerow(utils.sanitize_list_for_csv(headers))
                writer.writerow(utils.sanitize_list_for_csv(csv_data))
            with open(log_filepath, "wb", encoding="utf-8") as csvfile:
                csvfile.write(encryptor.encrypt(self.encryption_key, output.getvalue().encode()))
        else:
            if flag_index is None:
                with open(log_filepath, "a", newline="", encoding="utf-8") as csvfile:
                    writer = csv.writer(csvfile)
                    if is_new:
                        writer.writerow(utils.sanitize_list_for_csv(headers))
                    writer.writerow(utils.sanitize_list_for_csv(csv_data))
            else:
                with open(log_filepath, encoding="utf-8") as csvfile:
                    file_content = csvfile.read()
                    file_content = replace_flag_at_index(file_content)
                with open(
                    log_filepath, "w", newline="", encoding="utf-8"
                ) as csvfile:  # newline parameter needed for Windows
                    csvfile.write(utils.sanitize_list_for_csv(file_content))
        if self.to_s3:
            # Upload the file
            os.system(f"aws s3 cp flagged/ {self.dir} --recursive")

            # s3_client = boto3.client('s3')
            # try:
            #     response = s3_client.upload_file("flagged", "myemlobucket", "logs")
            # except ClientError as e:
            #     log.error(e)
        with open(log_filepath, encoding="utf-8") as csvfile:
            line_count = len([None for row in csv.reader(csvfile)]) - 1
        return line_count
