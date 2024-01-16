import onnx_graphsurgeon as gs
import numpy as np
import onnx

def CreateModel():
  # graph = gs.Graph(opset=11)
  input_ = gs.Variable(name="input", dtype=np.float32, shape=(6, 256, 36, 60))
  grid_ = gs.Variable(name="grid", dtype=np.float32, shape=(6, 900, 1, 2))
  output_ = gs.Variable(name="output", dtype=np.float32, shape=(6, 256, 900, 1))
  node = gs.Node(op="grid_sampler", attrs={"align_corners": 0, "interpolation_mode":0, "padding_mode":0}, inputs=[input_, grid_], outputs=[output_])
# , "xxxx":[1,2,3,4,5,6,7,8,9]
  graph = gs.Graph(nodes=[node], inputs=[input_, grid_], outputs=[output_], opset=13)
  onnx.save(gs.export_onnx(graph), "./gridsampler.onnx")

def main():
  CreateModel()

if __name__ == '__main__':
  main()
