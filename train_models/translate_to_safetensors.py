import argparse
import json

import pytorch_lightning  # pycl: ignore
import torch
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description="Translate saved model to safetensors")
    parser.add_argument("input", type=str, help="The input filename")
    parser.add_argument("output", type=str, help="The output file basename")

    args = parser.parse_args()

    save_items = torch.load(args.input)

    # Remove lightning dependency:
    save_items["hparams"] = dict(save_items["hparams"])

    tensor_keys = ["w_avg", "w2_avg", "pre_D"]
    json_keys = ["hparams", "swa_params"]

    with open(args.output + ".json", "w") as f:
        json_parts = {key: save_items[key] for key in json_keys}
        json.dump(json_parts, f)

    tensor_parts = {key: save_items[key] for key in tensor_keys}
    save_file(tensor_parts, args.output + ".safetensors")

if __name__ == "__main__":
    main()
