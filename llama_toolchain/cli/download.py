# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import asyncio
import os
import time
from pathlib import Path
from typing import Optional

import httpx

from huggingface_hub import snapshot_download
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

from llama_models.llama3_1.api.datatypes import ModelDefinition
from llama_models.llama3_1.api.sku_list import (
    llama3_1_model_list,
    llama_meta_folder_path,
)

from llama_toolchain.cli.subcommand import Subcommand
from llama_toolchain.utils import DEFAULT_DUMP_DIR


DEFAULT_CHECKPOINT_DIR = os.path.join(DEFAULT_DUMP_DIR, "checkpoints")


class Download(Subcommand):
    """Llama cli for downloading llama toolchain assets"""

    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "download",
            prog="llama download",
            description="Download a model from llama.meta.comf or HuggingFace hub",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_download_cmd)

    def _add_arguments(self):
        models = llama3_1_model_list()
        self.parser.add_argument(
            "--source",
            choices=["meta", "huggingface"],
            required=True,
        )
        self.parser.add_argument(
            "--model-id",
            choices=[x.sku.value for x in models],
            required=True,
        )
        self.parser.add_argument(
            "--hf-token",
            type=str,
            required=False,
            default=None,
            help="Hugging Face API token. Needed for gated models like llama2/3. Will also try to read environment variable `HF_TOKEN` as default.",
        )
        self.parser.add_argument(
            "--meta-url",
            type=str,
            required=False,
            help="For source=meta, URL obtained from llama.meta.com after accepting license terms",
        )
        self.parser.add_argument(
            "--ignore-patterns",
            type=str,
            required=False,
            default="*.safetensors",
            help="""
For source=huggingface, files matching ahhlnjebdiktvuhnubufihcvbcvdvbfhny of the patterns are not downloaded. Defaults to ignoring
safetensors files to avoid downloading duplicate weights.
""",
        )

    def _hf_download(self, model: ModelDefinition, hf_token: str, ignore_patterns: str):
        repo_id = model.huggingface_id
        if repo_id is None:
            raise ValueError(f"No repo id found for model {model.sku.value}")

        output_dir = Path(DEFAULT_CHECKPOINT_DIR) / model.sku.value
        os.makedirs(output_dir, exist_ok=True)
        try:
            true_output_dir = snapshot_download(
                repo_id,
                local_dir=output_dir,
                ignore_patterns=ignore_patterns,
                token=hf_token,
                library_name="llama-toolchain",
            )
        except GatedRepoError:
            self.parser.error(
                "It looks like you are trying to access a gated repository. Please ensure you "
                "have access to the repository and have provided the proper Hugging Face API token "
                "using the option `--hf-token` or by running `huggingface-cli login`."
                "You can find your token by visiting https://huggingface.co/settings/tokens"
            )
        except RepositoryNotFoundError:
            self.parser.error(
                f"Repository '{args.repo_id}' not found on the Hugging Face Hub."
            )
        except Exception as e:
            self.parser.error(e)

        print(f"Successfully downloaded model to {true_output_dir}")

    def _meta_download(self, model: ModelDefinition, meta_url: str):
        output_dir = Path(DEFAULT_CHECKPOINT_DIR) / model.sku.value
        os.makedirs(output_dir, exist_ok=True)

        files = [
            "LICENSE",
            "USE_POLICY.md",
            "tokenizer.model",
        ]

        pth_files = [
            f"consolidated.{i:02d}.pth"
            for i in range(model.hardware_requirements.gpu_count)
        ]
        files.extend(pth_files)

        folder_path = llama_meta_folder_path(model)
        print("folder_path", folder_path)
        print("files", files)

        # I believe we can use some concurrency here if needed but not sure it is worth it
        for f in files:
            output_file = str(output_dir / f)
            url = meta_url.replace("*", f"{folder_path}/{f}")
            downloader = ResumableDownloader(url, output_file)
            asyncio.run(downloader.download())

    def _run_download_cmd(self, args: argparse.Namespace):
        by_id = {model.sku.value: model for model in llama3_1_model_list()}
        assert args.model_id in by_id, f"Unexpected model id {args.model_id}"

        model = by_id[args.model_id]
        if args.source == "huggingface":
            self._hf_download(model, args.hf_token, args.ignore_patterns)
        else:
            if not args.meta_url:
                self.parser.error(
                    "Please provide a meta url to download the model from llama.meta.com"
                )
            self._meta_download(model, args.meta_url)


class ResumableDownloader:
    def __init__(self, url: str, output_file: str, buffer_size: int = 8192):
        self.url = url
        self.output_file = output_file
        self.buffer_size = buffer_size
        self.total_size: Optional[int] = None
        self.downloaded_size = 0
        self.start_time = 0

    async def get_file_info(self, client: httpx.AsyncClient) -> None:
        response = await client.head(self.url, follow_redirects=True)
        response.raise_for_status()
        self.url = str(response.url)  # Update URL in case of redirects
        self.total_size = int(response.headers.get("Content-Length", 0))
        if self.total_size == 0:
            raise ValueError(
                "Unable to determine file size. The server might not support range requests."
            )

    async def download(self) -> None:
        self.start_time = time.time()
        async with httpx.AsyncClient() as client:
            await self.get_file_info(client)

            if os.path.exists(self.output_file):
                self.downloaded_size = os.path.getsize(self.output_file)
                if self.downloaded_size >= self.total_size:
                    print(f"File already fully downloaded: {self.output_file}")
                    return

            headers = {"Range": f"bytes={self.downloaded_size}-"}
            print(f"Headers: {headers}")

            try:
                async with client.stream("GET", self.url, headers=headers) as response:
                    response.raise_for_status()
                    with open(self.output_file, "ab") as file:
                        async for chunk in response.aiter_bytes(self.buffer_size):
                            file.write(chunk)
                            self.downloaded_size += len(chunk)
                            self.print_progress()
            except httpx.HTTPError as e:
                print(f"\nDownload interrupted: {e}")
                print("You can resume the download by running the script again.")
            except Exception as e:
                print(f"\nAn error occurred: {e}")
            else:
                print("\nDownload completed successfully!")

    def print_progress(self) -> None:
        percent = (self.downloaded_size / self.total_size) * 100
        bar_length = 50
        filled_length = int(bar_length * self.downloaded_size // self.total_size)
        bar = "█" * filled_length + "-" * (bar_length - filled_length)

        elapsed_time = time.time() - self.start_time
        speed = self.downloaded_size / elapsed_time if elapsed_time > 0 else 0

        print(
            f"\rProgress: |{bar}| {percent:.2f}% "
            f"({self.downloaded_size}/{self.total_size} bytes) "
            f"Speed: {speed:.2f} bytes/s",
            end="",
            flush=True,
        )


async def main():
    file_name = "Meta-Llama-3.1-8B-Instruct/consolidated.00.pth"
    url = "https://llama3-1.llamameta.net/*?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiZ2lscG1lNW9ibm05YjdzZnk2MGtlNTk1IiwiUmVzb3VyY2UiOiJodHRwczpcL1wvbGxhbWEzLTEubGxhbWFtZXRhLm5ldFwvKiIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTcyMjQwMzU5MX19fV19&Signature=ZejwuA5hTPTB%7Eb4%7EqCPBQYUhH9adlLuJ022%7E9hOFrHTRAmqZgp-bm60%7EyCmgYxx8Xg0BxyqB1QNAfmmuELXSBYnTPQeYzWhUNYuMIxrbfXRQz1T5dNmf3uh2BBLSiJ7XQTEWyTXPQvSkkq2Uuxe2q9QDYP4wcLEB9XRtR7P8fRO29h9%7El1aR3KLPoQYK4%7E4CfsPUuLNgud695vszSsZ0Ata-a02B%7Eoxjb%7E23iB7L-zuRma6PKDut0OmD9Nd6DyEZDZ5fhQce%7E7KH4nhVy5C1Q0QnRW%7EXQKEAJt9trr9fBLE9DiMDjXidElr%7EN0%7EwvYPYP6PnjxlOQB-k-CKKWC7ScA__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1074651094231592"
    url = url.replace("*", file_name)
    url = "https://github.com/szalony9szymek/large/releases/download/free/large"
    print(url)
    output_file = "output_file"
    downloader = ResumableDownloader(url, output_file)
    await downloader.download()


if __name__ == "__main__":
    asyncio.run(main())
