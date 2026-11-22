import argparse
import hashlib
import json
from pathlib import Path


def main():
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.source.resolve() == args.output.resolve():
        raise ValueError("Use a new candidate path; preserve the original model")
    model = onnx.load(str(args.source))
    initializers = {weight.name: weight for weight in model.graph.initializer}
    converted = set()
    nodes = []
    for node in model.graph.node:
        weight = initializers.get(node.input[0]) if node.input else None


        if (node.op_type == "Gather" and weight is not None and weight.data_type == TensorProto.FLOAT
                and len(weight.dims) == 2 and weight.dims[0] >= 512
                and all(other.op_type == "Gather" for other in model.graph.node if weight.name in other.input)):
            name = weight.name + "_fp16"
            if name not in converted:
                original = numpy_helper.to_array(weight)
                array = original.astype(np.float16)
                if not np.array_equal(original, array.astype(np.float32)):
                    raise ValueError(f"Embedding {weight.name} cannot be stored losslessly as FP16")
                model.graph.initializer.append(numpy_helper.from_array(array, name))
                converted.add(name)
            node.input[0] = name
            original_output = node.output[0]
            node.output[0] = original_output + "_fp16"
            nodes.extend([node, helper.make_node("Cast", [node.output[0]], [original_output],
                                                 name=node.name + "/restore_fp32", to=TensorProto.FLOAT)])
        else:
            nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    used = {name for node in nodes for name in node.input} | {value.name for value in model.graph.output}
    kept = [weight for weight in model.graph.initializer if weight.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    if not converted:
        raise ValueError("No eligible embedding tables")
    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.output))
    print(json.dumps({"source_bytes": args.source.stat().st_size,
                      "candidate_bytes": args.output.stat().st_size, "tables": sorted(converted),
                      "candidate_sha256": hashlib.file_digest(args.output.open("rb"), "sha256").hexdigest()}))


if __name__ == "__main__":
    main()
