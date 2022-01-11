import functools
import io
import logging
import os
import signal
import subprocess
import tempfile

import numpy as np
import torch

from torchdynamo.optimizations.subgraph import SubGraph

log = logging.getLogger(__name__)
BACKENDS = dict()
_NP_DTYPE = {
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.bool: np.bool_,
}


def create_backend(fn):
    @functools.wraps(fn)
    def inner(model, example_inputs=None, **kwargs):
        if model is None:
            return None

        if not isinstance(model, SubGraph):
            with tempfile.TemporaryDirectory() as tmp:
                return inner(SubGraph(model, example_inputs, tmp), **kwargs)
        else:
            assert example_inputs is None

        try:
            return fn(model, **kwargs)
        except KeyboardInterrupt:
            raise
        except Exception:
            log.exception(f"{fn.__name__} error")
            return None

    BACKENDS[fn.__name__] = inner
    return inner


@create_backend
def eager(subgraph):
    return subgraph.model


@create_backend
def ts(subgraph):
    return subgraph.scripted


def reload_jit_model(subgraph):
    tmp = io.BytesIO()
    torch.jit.save(subgraph.scripted, tmp)
    tmp.seek(0)
    model = torch.jit.load(tmp)
    # populate cache
    for _ in range(3):
        model(*subgraph.example_inputs)
    return model


@create_backend
def nnc(subgraph):
    with torch.jit.fuser("fuser1"):
        return reload_jit_model(subgraph)


@create_backend
def nvfuser(subgraph):
    with torch.jit.fuser("fuser2"):
        return reload_jit_model(subgraph)


@create_backend
def ofi(subgraph):
    return torch.jit.optimize_for_inference(subgraph.scripted)


@create_backend
def static_runtime(subgraph):
    scripted = subgraph.scripted
    if hasattr(scripted, "_c"):
        static_module = torch._C._jit_to_static_module(scripted._c)
    else:
        static_module = torch._C._jit_to_static_module(scripted.graph)
    return subgraph.wrap_returns(static_module)


def onnxrt_common(subgraph, provider, onnx_filename=None):
    import onnxruntime

    assert provider in onnxruntime.get_available_providers()
    session = onnxruntime.InferenceSession(
        onnx_filename or subgraph.onnx_filename, providers=[provider]
    )
    input_names = subgraph.input_names
    output_names = subgraph.output_names
    create_outputs = subgraph.empty_outputs_factory()
    is_cpu = subgraph.is_cpu

    def _call(*args):
        binding = session.io_binding()
        args = [a.contiguous() for a in args]
        for name, value in zip(input_names, args):
            dev = value.device
            binding.bind_input(
                name,
                dev,
                dev.index or 0,
                _NP_DTYPE[value.dtype],
                value.size(),
                value.data_ptr(),
            )
        outputs = create_outputs()
        for name, value in zip(output_names, outputs):
            dev = value.device
            binding.bind_output(
                name,
                dev,
                dev.index or 0,
                _NP_DTYPE[value.dtype],
                value.size(),
                value.data_ptr(),
            )
        session.run_with_iobinding(binding)
        if is_cpu:
            binding.copy_outputs_to_cpu()
        return outputs

    return subgraph.wrap_returns(_call)


@create_backend
def onnxrt_cpu(subgraph):
    return onnxrt_common(subgraph, provider="CPUExecutionProvider")


@create_backend
def onnxrt_cuda(subgraph):
    return onnxrt_common(subgraph, provider="CUDAExecutionProvider")


@create_backend
def onnx2tensorrt(subgraph):
    return onnxrt_common(subgraph, provider="TensorrtExecutionProvider")


@create_backend
def onnxrt_cpu_numpy(subgraph, provider="CPUExecutionProvider"):
    """Alternate version that integrates via numpy"""
    import onnxruntime

    assert provider in onnxruntime.get_available_providers()
    ort_session = onnxruntime.InferenceSession(
        subgraph.onnx_filename, providers=[provider]
    )

    def to_numpy(x):
        try:
            return x.numpy()
        except RuntimeError:
            return x.detach().numpy()

    def _call(*args):
        res = ort_session.run(
            None, {f"i{i}": to_numpy(arg) for i, arg in enumerate(args)}
        )
        res = [torch.from_numpy(x) for x in res]
        return res

    return subgraph.wrap_returns(_call)


@create_backend
def onnxrt(subgraph):
    if subgraph.is_cuda:
        return onnxrt_cuda(subgraph)
    else:
        return onnxrt_cpu(subgraph)


@functools.lru_cache(None)
def _init_tensorflow():
    import tensorflow as tf

    # prevent tensorflow from eating all the GPU memory
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    return tf


@create_backend
def onnx2tf(subgraph):
    import onnx
    from onnx_tf.backend import prepare

    tf = _init_tensorflow()
    filename = subgraph.filename("tensorflow")
    input_names = subgraph.input_names
    output_names = subgraph.output_names
    device = "/CPU:0" if subgraph.is_cpu else f"/GPU:{subgraph.device_index}"
    with tf.device(device):
        if not os.path.exists(filename):
            prepare(onnx.load(subgraph.onnx_filename)).export_graph(filename)
        tf_module = tf.saved_model.load(filename)
        tf_module = tf.function(tf_module, jit_compile=True)

    def run(*args):
        args = [a.contiguous() for a in args]
        with tf.device(device):
            outs = tf_module(
                **{
                    name: tf.experimental.dlpack.from_dlpack(
                        torch.utils.dlpack.to_dlpack(args[idx])
                    )
                    for idx, name in enumerate(input_names)
                }
            )
            return [
                torch.utils.dlpack.from_dlpack(
                    tf.experimental.dlpack.to_dlpack(outs[name])
                )
                for name in output_names
            ]

    return subgraph.wrap_returns(run)


@create_backend
def taso(subgraph):
    taso_filename = subgraph.filename("taso")
    subprocess.check_call(
        [
            os.path.expanduser("~/conda/envs/taso/bin/python"),
            "-c",
            "import taso,onnx; onnx.save(taso.export_onnx(taso.optimize("
            f"taso.load_onnx('{subgraph.onnx_filename}'))), '{taso_filename}')",
        ]
    )
    return onnxrt_common(
        subgraph, provider="CUDAExecutionProvider", onnx_filename=taso_filename
    )


@create_backend
def ipex(subgraph):
    import intel_extension_for_pytorch

    return intel_extension_for_pytorch.optimize(subgraph.scripted)


def _raise_timeout(signum, frame):
    raise TimeoutError()


@create_backend
def fx2trt(subgraph):
    from torch.fx.experimental.fx2trt.fx2trt import InputTensorSpec
    from torch.fx.experimental.fx2trt.fx2trt import TRTInterpreter
    from torch.fx.experimental.fx2trt.trt_module import TRTModule
    import torch.fx.experimental.fx_acc.acc_tracer as acc_tracer

    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.alarm(120)  # fx2trt infinite loops sometimes
    try:
        logging.getLogger("torch.fx.experimental.fx_acc.acc_tracer").setLevel(
            logging.ERROR
        )

        model = subgraph.model
        inputs = subgraph.example_inputs

        model = acc_tracer.trace(model, inputs)
        input_specs = InputTensorSpec.from_tensors(inputs)
        interp = TRTInterpreter(model, input_specs, explicit_precision=True)
        result = interp.run(fp16_mode=False, max_batch_size=len(inputs[0]))
        trt_mod = TRTModule(result.engine, result.input_names, result.output_names)
        return subgraph.wrap_returns(trt_mod)
    finally:
        signal.alarm(0)


@create_backend
def torch2trt(subgraph):
    from torch2trt import torch2trt

    inputs = subgraph.example_inputs
    trt_mod = torch2trt(
        subgraph.model,
        inputs,
        max_batch_size=len(inputs[0]),
        strict_type_constraints=True,
    )
    return subgraph.wrap_returns(trt_mod)


@create_backend
def tensorrt(subgraph):
    model = onnx2tensorrt(subgraph)
    if model is None:
        model = torch2trt(subgraph)
    return model


@create_backend
def onnx2tensorrt_alt(subgraph):
    import tensorrt as trt
    from torch.fx.experimental.fx2trt.trt_module import TRTModule

    inputs = subgraph.example_inputs

    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    assert isinstance(inputs, (list, tuple))
    inputs = tuple(inputs)
    input_names = subgraph.input_names
    output_names = subgraph.output_names
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    success = parser.parse(open(subgraph.onnx_filename, "rb").read())
    for idx in range(parser.num_errors):
        print(parser.get_error(idx))
    assert success

    config.max_workspace_size = 1 << 25
    config.set_flag(trt.BuilderFlag.STRICT_TYPES)
    builder.max_batch_size = len(inputs[0])

    engine = builder.build_engine(network, config)
    assert engine

    trt_mod = TRTModule(engine, input_names, output_names)
    return subgraph.wrap_returns(trt_mod)


@create_backend
def cudagraphs(subgraph):
    model = subgraph.model
    inputs = subgraph.example_inputs

    assert subgraph.is_cuda
    assert isinstance(inputs, (list, tuple))
    static_inputs = [torch.randn_like(x) for x in inputs]

    # warmup
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        model(*inputs)
    stream.synchronize()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    # record
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        static_outputs = model(*static_inputs)
    if not isinstance(static_outputs, (list, tuple)):
        static_outputs = (static_outputs,)

    def run(*new_inputs):
        assert len(static_inputs) == len(new_inputs)
        for dst, src in zip(static_inputs, new_inputs):
            dst.copy_(src)
        graph.replay()
        return [x.clone() for x in static_outputs]

    return subgraph.wrap_returns(run)


def tvm_compile(jit_mod, example_inputs, log_file=None, **kwargs):
    if jit_mod is None:
        return None
    try:
        return tvm_compile_inner(jit_mod, example_inputs, log_file, **kwargs)
    except Exception as e:
        if log_file and os.path.exists(log_file):
            os.unlink(log_file)
        if isinstance(e, KeyboardInterrupt):
            raise
        log.exception("tvm error")
        return None


@create_backend
def tvm(subgraph):
    return subgraph.wrap_returns(
        tvm_compile_inner(
            subgraph.scripted, subgraph.example_inputs, None, cuda=subgraph.is_cuda
        )
    )


@create_backend
def ansor(subgraph):
    """
    WARNING: this backend takes hours or days to train and
    often produces a slower result than the default schedule.
    """
    return subgraph.wrap_returns(
        tvm_compile_inner(
            subgraph.scripted,
            subgraph.example_inputs,
            subgraph.filename("ansor"),
            cuda=subgraph.is_cuda,
        )
    )


@functools.lru_cache(None)
def llvm_target():
    if "avx512" in open("/proc/cpuinfo").read():
        return "llvm -mcpu=skylake-avx512"
    return "llvm -mcpu=core-avx2"


def tvm_compile_inner(jit_mod, example_inputs, log_file, trials=20000, cuda=False):
    # based on functorch version in eager_compile.py
    import tvm
    from tvm import relay, auto_scheduler
    from tvm.contrib import graph_executor

    shape_list = [(f"inp_{idx}", i.shape) for idx, i in enumerate(example_inputs)]
    mod, params = relay.frontend.from_pytorch(jit_mod, shape_list)
    if cuda:
        dev = tvm.cuda(0)
        target = tvm.target.cuda()
    else:
        dev = tvm.cpu(0)
        target = tvm.target.Target(llvm_target())
    if log_file is not None:
        if not os.path.exists(log_file):
            tasks, task_weights = auto_scheduler.extract_tasks(
                mod["main"], params, target
            )
            for task in tasks:
                print(task.compute_dag)
            else:
                print("No tasks")
            if len(tasks) != 0:
                tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
                if not os.path.exists(log_file):
                    assert trials > 0
                    tune_option = auto_scheduler.TuningOptions(
                        num_measure_trials=trials,
                        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
                        early_stopping=2000,
                    )
                    try:
                        tuner.tune(tune_option)
                    except Exception:
                        if os.path.exists(log_file):
                            os.unlink(log_file)
                        raise

        with auto_scheduler.ApplyHistoryBest(log_file):
            with tvm.transform.PassContext(
                opt_level=3, config={"relay.backend.use_auto_scheduler": True}
            ):
                lib = relay.build(mod, target=target, params=params)
    else:
        # no autotuning (for debugging)
        with tvm.transform.PassContext(opt_level=10):
            lib = relay.build(mod, target=target, params=params)

    m = graph_executor.GraphModule(lib["default"](dev))

    def exec_tvm(*args):
        args = [a.contiguous() for a in args]
        for idx, arg in enumerate(args, 0):
            if arg.dim() != 0:
                m.set_input(
                    f"inp_{idx}",
                    tvm.nd.from_dlpack(torch.utils.dlpack.to_dlpack(arg)),
                )
        m.run()
        return [
            torch.utils.dlpack.from_dlpack(m.get_output(i).to_dlpack())
            for i in range(m.get_num_outputs())
        ]

    return exec_tvm