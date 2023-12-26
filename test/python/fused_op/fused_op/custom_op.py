import json
import hashlib
import torch
import importlib
from importlib import resources
from .run_tuned_json_graph import get_device_source


package_name = "fused_op"

dtype_mapping = {
      'float64': torch.float64,
      'float32': torch.float32,
      'float16': torch.float16,
      'int64': torch.int64,
      'int32': torch.int32,
      'int16': torch.int16,
      'int8': torch.int8,
    }
def generate_welder_graph(ir, feed_list, extra_outputs, tags=""):
  input_dict, kwargs = {}, {}
  for k, i, shape, dtype in feed_list:
    input_dict[k] = {
      'dtype': str(dtype).split('.')[1],
      'shape': list(shape)
    }

  ir = ir.replace('"', '`').replace('\n', ' ').strip()
  input_dict = json.dumps(input_dict)
  extra_outputs = ', '.join(['"%s"' % x for x in extra_outputs])
  expression = f'- einstein_v2("{ir}", input_dict={input_dict}, extra_outputs=[{extra_outputs}]) ## @: {tags}'

  nodes = []
  edges = [[id, 0] for id in range(len(feed_list))]
  node_id = len(feed_list)
  nodes.append([node_id, expression, "fused_op", edges])
  nodes.append([node_id + 1, "", "Result", [[node_id, 0]]])

  return json.dumps(nodes, indent=2)


def load_kernel(arch, op_name, hash_key):
  with resources.open_text(f"{package_name}.kernel.{arch}.{op_name}", f"{hash_key}.json") as f:
    raw_json_graph = json.load(f)
  with resources.open_text(f"{package_name}.kernel.{arch}.{op_name}", f"{hash_key}.kernel.json") as f:
    tuned_json_graph = json.load(f)
  
  inputs_outputs_info = []
  device_source = get_device_source(raw_json_graph, tuned_json_graph, inputs_outputs_info)

  backend = 'c-cuda'
  lib_name = 'antares_custom_torch_v2_%s' % backend.replace('-', '_')
  try:
    custom_lib = importlib.import_module(lib_name)
  except:
    print(f'Failed to import {lib_name}.\nPlease install Custom Plugin for backend in advance: BACKEND={backend} antares torch-setup')
  custom_key = custom_lib.inject(device_source)
  return custom_lib, custom_key, inputs_outputs_info

class CompiledKernel:
  def __init__(self, custom_lib, custom_key, inout_info):
    self.custom_lib = custom_lib
    self.custom_key = custom_key
    self.inout_info = inout_info

KERNEL_CACHE = {}

class CustomOp(torch.nn.Module):
  # def __init__(self, kernel_file, device=None):
  #   super(CustomOp, self).__init__()
  #   if device is None:
  #     self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  #   else:
  #     self.device = device
  #   self.custom_lib, self.custom_key, inout_info = load_kernel(kernel_file)
  #   self.output_list = inout_info[1]

  
  def __init__(self, ir, input_orders, extra_outputs=[], tags="", op_name="default", steps=1, arch='g3090', device=None):
    super(CustomOp, self).__init__()
    if device is None:
      self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
      self.device = device
    ir = ir.replace('"', '`').replace('\n', ' ').strip()
    input_list, index = [], 0
    for k in input_orders:
      if isinstance(input_orders[k], tuple):
        input_list += [(k, index, input_orders[k][2], input_orders[k][1])]
      else:
        input_list += [(k, index, input_orders[k].shape, input_orders[k].dtype)]
      index += 1

    self.input_orders = sorted(input_list, key=lambda x: x[0])
    self.graph = generate_welder_graph(ir, input_list, extra_outputs, tags)

    self.hash_key = hashlib.sha256((ir + str(input_list) + str(extra_outputs) + str(tags)).encode()).hexdigest()
    if self.hash_key in KERNEL_CACHE:
      cache = KERNEL_CACHE[self.hash_key]
      self.custom_lib = cache.custom_lib
      self.custom_key = cache.custom_key
      self.output_list = cache.inout_info[1]
      return
      
    self.custom_lib, self.custom_key, inout_info = load_kernel(arch, op_name, self.hash_key)
    self.output_list = inout_info[1]
    KERNEL_CACHE[self.hash_key] = CompiledKernel(self.custom_lib, self.custom_key, inout_info)

  def input_info(self):
    return self.input_list
  
  def forward(self, inputs):
    ordered_inputs = []
    for i in range(len(inputs)):
      inp = inputs[i]
      ordered_inputs.append(inp.contiguous().to(self.device))

    outputs = []
    for info in self.output_list:
      out = torch.empty(info[1]['shape'], device=self.device, dtype=dtype_mapping[info[1]['dtype']])
      outputs.append(out)
    self.custom_lib.forward(self.custom_key, ordered_inputs + outputs)
    outputs = outputs[0] if len(outputs) == 1 else tuple(outputs)
    return outputs
