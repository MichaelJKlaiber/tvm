# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
from os.path import join
from typing import Union, List, Dict, Any

import pytest
from tests.python.relay.aot.test_crt_aot_usmp import (
    _check_for_no_tvm_backendallocworkspace_calls,
    MOBILENET_V1_URL,
)
from tvm.contrib.download import download_testdata
from tvm.micro.testing.aot_test_utils import AOT_DEFAULT_RUNNER
from tvm.relay import testing, transform
from tvm.testing.aot import (
    run_and_check,
    AOTTestModel,
    AOTCompiledTestModel,
    AOTTestRunner,
    create_relay_module_and_inputs_from_tflite_file,
    generate_ref_data,
    compile_and_run,
)

import tvm
from test_uma_vanilla_accelerator import VanillaAcceleratorBackend
from tvm import relay, IRModule
import numpy as np
from collections import OrderedDict
from tvm.micro import model_library_format as mlf

import onnx
from tvm.testing.aot import compile_models


@pytest.mark.parametrize(
    "interface_api,use_unpacked_api,test_runner,groups,weight_shape",
    [("c", True, AOT_DEFAULT_RUNNER, 1, 32)],
)
def test_conv2d(interface_api, use_unpacked_api, test_runner, groups, weight_shape):
    """Test a subgraph with a single conv2d operator."""
    mod, inputs, output_list, test_runner = create_conv2d(groups, test_runner, weight_shape)

    uma_backend = VanillaAcceleratorBackend()
    uma_backend.register()
    mod = uma_backend.partition(mod)
    target = tvm.target.Target("vanilla_accelerator", host=tvm.target.Target("c"))
    target_c = tvm.target.Target("c")

    compile_and_run(
        AOTTestModel(module=mod, inputs=inputs, outputs=output_list),
        test_runner,
        interface_api,
        use_unpacked_api,
        #target=target,
        target=[target_c, target],
    )


def create_conv2d(groups=1, test_runner=AOT_DEFAULT_RUNNER, weight_shape=32):
    dtype = "float32"
    ishape = (1, 32, 14, 14)
    wshape = (32, weight_shape, 3, 3)
    pass_config = {"tir.usmp.enable": True}
    test_runner = AOTTestRunner(
        makefile=test_runner.makefile,
        prologue=test_runner.prologue,
        epilogue=test_runner.epilogue,
        includes=test_runner.includes,
        parameters=test_runner.parameters,
        pass_config=pass_config,
    )
    data0 = relay.var("data", shape=ishape, dtype=dtype)
    weight0 = relay.var("weight", shape=wshape, dtype=dtype)
    out = relay.nn.conv2d(data0, weight0, kernel_size=(3, 3), padding=(1, 1), groups=groups)
    main_f = relay.Function([data0, weight0], out)
    mod = tvm.IRModule()
    mod["main"] = main_f
    mod = transform.InferType()(mod)
    i_data = np.random.uniform(0, 1, ishape).astype(dtype)
    w1_data = np.random.uniform(0, 1, wshape).astype(dtype)
    inputs = OrderedDict([("data", i_data), ("weight", w1_data)])
    output_list = generate_ref_data(mod, inputs)
    return mod, inputs, output_list, test_runner


def test_tflite_model_u1_usecase(model_url, usmp_algo, workspace_size, constant_size):
    """
    This checks for ML models and the memory used by them
    when using USMP with different algorithms
    """
    pytest.importorskip("tflite")

    import tvm.relay.testing.tf as tf_testing  # pylint: disable=import-outside-toplevel

    use_unpacked_api = True
    interface_api = "c"
    test_runner = AOTTestRunner(
        pass_config={"tir.usmp.enable": True, "tir.usmp.algorithm": usmp_algo}
    )

    tflite_model_file = tf_testing.get_workload_official(
        model_url[0],
        model_url[1],
    )
    mod, inputs, params = create_relay_module_and_inputs_from_tflite_file(tflite_model_file)
    output_list = generate_ref_data(mod, inputs, params)

    uma_backend = VanillaAcceleratorBackend()
    uma_backend.register()
    mod = uma_backend.partition(mod)
    target = tvm.target.Target("vanilla_accelerator", host=tvm.target.Target("c"))

    aotmodel = AOTTestModel(module=mod, inputs=inputs, outputs=output_list, params=params)
    compiled_test_mods = compile_models(
        aotmodel,
        interface_api=interface_api,
        use_unpacked_api=use_unpacked_api,
        pass_config=test_runner.pass_config,
        target=target,
    )

    for compiled_model in compiled_test_mods:
        _check_for_no_tvm_backendallocworkspace_calls(compiled_model.executor_factory.lib)

    # Checking the workspace size reported in model library format
    mlf_memory_map = mlf._build_function_memory_map(
        compiled_test_mods[0].executor_factory.function_metadata
    )
    assert mlf_memory_map["main"][0]["workspace_size_bytes"] == workspace_size
    assert mlf_memory_map["main"][0]["constants_size_bytes"] == constant_size
    # That should match to workspace size that will be codegen'd to the entry point.
    allocated_pool_info_size = sum(
        [
            _.allocated_size
            for _ in list(
                dict(
                    compiled_test_mods[0].executor_factory.executor_codegen_metadata.pool_inputs
                ).values()
            )
        ]
    )
    assert allocated_pool_info_size == workspace_size + constant_size

    run_and_check(
        models=compiled_test_mods, runner=test_runner, interface_api=interface_api, target=target
    )


def download_and_import_onnx_model(model_url: str) -> [IRModule, dict, dict, dict]:
    """
    Download an ONNX NN model from `url`  and import it using the TVM onnx frontend
    """

    def _get_shape(io):
        shape = []
        dimensions = io.type.tensor_type.shape.dim
        for dim in dimensions:
            shape.append(dim.dim_value)
        return shape

    filename = model_url.split("/")[-1]
    model_url = "".join([model_url])
    model_path = download_testdata(model_url, filename, module="onnx")
    onnx_model = onnx.load(model_path)
    graph_input = onnx_model.graph.input
    assert len(graph_input) == 1
    input_name = graph_input[0].name
    input_shape = _get_shape(graph_input[0])
    graph_output = onnx_model.graph.output
    assert len(graph_output) == 1
    output_name = graph_output[0].name
    output_shape = _get_shape(graph_output[0])
    input_shape_dict = {input_name: input_shape}
    output_shape_dict = {output_name: output_shape}
    mod, params = relay.frontend.from_onnx(onnx_model, input_shape_dict)
    return mod, params, input_shape_dict, output_shape_dict


def test_vanilla_accelerator_integration(url: str):
    mod, params, input_shapes, output_shapes = download_and_import_onnx_model(url)
    _vanilla_accelerator_run(mod, params, input_shapes, output_shapes)


def _generate_runtime_data(
    input_shapes: dict, output_shapes: dict
) -> [OrderedDict, OrderedDict]:
    assert len(input_shapes) == 1
    assert len(output_shapes) == 1

    iname = list(input_shapes.keys())[0]
    oname = list(output_shapes.keys())[0]
    ishape = input_shapes[iname]
    oshape = output_shapes[oname]
    i_data = np.random.uniform(0, 1, ishape).astype("float32")
    o_data = np.random.uniform(0, 1, oshape).astype("float32")
    oname = "output"  # name set by relay.build in executor_codegen_metadata.outputs
    inputs = OrderedDict([(iname, i_data)])
    outputs = OrderedDict([(oname, o_data)])
    return inputs, outputs


def _vanilla_accelerator_run(mod, params, input_shapes, output_shapes):
    interface_api = "c"
    use_unpacked_api = True

    # uma_backend = VanillaAcceleratorBackend()
    # uma_backend.register()
    # mod = uma_backend.partition(mod)
    # target = tvm.target.Target("vanilla_accelerator", host=tvm.target.Target("c"))
    target_c = tvm.target.Target("c")

    input_list, output_list = _generate_runtime_data(input_shapes, output_shapes)
    aot_test_model = AOTTestModel(module=mod, inputs=input_list, outputs=output_list)
    test_runner = AOT_DEFAULT_RUNNER

    compile_and_run(aot_test_model, test_runner, interface_api, use_unpacked_api,
                    target=target_c)
    #target=[target_c, target])

    return
    compiled_test_mods = compile_models(
        models=AOTTestModel(module=mod, inputs=input_list, outputs=output_list),
        interface_api=interface_api,
        use_unpacked_api=use_unpacked_api,
        pass_config=test_runner.pass_config,
        target=target,
    )

    for compiled_model in compiled_test_mods:
        _check_for_no_tvm_backendallocworkspace_calls(compiled_model.executor_factory.lib)

    run_and_check(
        models=compiled_test_mods,
        runner=test_runner,
        interface_api=interface_api,
    )

    aot_test_model = AOTTestModel(module=mod, inputs=input_list, outputs=output_list, params=params)

    runner = AOTTestRunner(pass_config={"tir.usmp.enable": True})

    compiled_test_mods = compile_models(
        aot_test_model,
        interface_api=interface_api,
        use_unpacked_api=use_unpacked_api,
        pass_config=runner.pass_config,
        target=target,
    )

    for compiled_model in compiled_test_mods:
        _check_for_no_tvm_backendallocworkspace_calls(compiled_model.executor_factory.lib)

    mlf_memory_map = mlf._build_function_memory_map(
        compiled_test_mods[0].executor_factory.function_metadata
    )

    run_and_check(compiled_test_mods, runner, interface_api)

def test_two_layers():
    RELAY_MODEL = """
    #[version = "0.0.5"]  
    def @main(%Input3: Tensor[(1, 1, 28, 28), float32] /* ty=Tensor[(1, 1, 28, 28), float32] */) -> Tensor[(1, 10), float32] {
          %0 = nn.pad(%Input3, 0f /* ty=float32 */, pad_width=[[0i64, 0i64], [0i64, 0i64], [2i64, 2i64], [2i64, 2i64]]) /* ty=Tensor[(1, 1, 32, 32), float32] */;
          %1 = nn.conv2d(%0, meta[relay.Constant][0] /* ty=Tensor[(8, 1, 5, 5), float32] */, padding=[0, 0, 0, 0], channels=8, kernel_size=[5, 5]) /* ty=Tensor[(1, 8, 28, 28), float32] */;
          %2 = add(%1, meta[relay.Constant][1] /* ty=Tensor[(8, 1, 1), float32] */) /* ty=Tensor[(1, 8, 28, 28), float32] */;
          nn.relu(%2) /* ty=Tensor[(1, 8, 28, 28), float32] */;
    """
    mod = tvm.parser.fromtext(RELAY_MODEL)
    params, input_shape_dict, output_shape_dict = None, {"Input3": [1, 1, 28, 28]}, {"output": [1, 1, 28, 28]}

    return mod, params, input_shape_dict, output_shape_dict


@pytest.mark.parametrize(
    ["debug_calculated_workspaces", "workspace_byte_alignment"], [(False, 1)]
)
def test_mobilenet(debug_calculated_workspaces, workspace_byte_alignment):
    """Full network test with Mobilenet"""
    use_unpacked_api = True
    interface_api = "c"
    test_runner = AOT_DEFAULT_RUNNER

    # TODO(@Mousius) - Enable memory planning to take into account debug information
    debugging_memory_overhead = 1024 * 1024

    mod, params = testing.mobilenet.get_workload(batch_size=1)

    uma_backend = VanillaAcceleratorBackend()
    uma_backend.register()
    mod = uma_backend.partition(mod)
    target = tvm.target.Target("vanilla_accelerator", host=tvm.target.Target("c"))
    target_c = tvm.target.Target("c")

    data_shape = [int(x) for x in mod["main"].checked_type.arg_types[0].shape]
    data = np.random.uniform(size=data_shape).astype("float32")
    input_list = {"data": data}
    output_list = generate_ref_data(mod, input_list, params)
    aot_test_model = AOTTestModel(module=mod, inputs=input_list, outputs=output_list, params=params)

    compile_and_run(
        aot_test_model,
        test_runner,
        interface_api,
        use_unpacked_api,
        workspace_byte_alignment=workspace_byte_alignment,
        debug_calculated_workspaces=debug_calculated_workspaces,
        target=[target_c, target]
    )


if __name__ == "__main__":
    # mod, params, input_shapes, output_shapes = test_two_layers()
    # _vanilla_accelerator_run(mod, params, input_shapes, output_shapes)

    # test_vanilla_accelerator_integration(
    #     "https://github.com/onnx/models/raw/main/vision/classification/mnist/model/mnist-12.onnx"
    # )

    # test_vanilla_accelerator_integration(
    #     "https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet50-v1-12.onnx"
    # )
    # test_tflite_model_u1_usecase(MOBILENET_V1_URL, "greedy_by_size", 4845696, 8468008)
    # test_conv2d("c", True, AOT_DEFAULT_RUNNER, 1, 32)

    test_mobilenet(False, 1)
