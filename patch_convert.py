import tensorflow as tf
from tensorflow.python.platform import gfile

# 1. Register DeePMD custom operators
try:
    import deepmd
    import os
    deepmd_path = os.path.dirname(deepmd.__file__)
    op_lib_path = os.path.join(deepmd_path, "lib", "deepmd_op.dll")
    if os.path.exists(op_lib_path):
        tf.load_op_library(op_lib_path)
        print("Successfully registered DeePMD-kit custom operators.")
    else:
        from deepmd.tf.env import tf as dptf
        print("Registered operators via internal package fallback.")
except Exception as e:
    print(f"Warning while loading custom operator library: {e}")

tf.compat.v1.disable_eager_execution()

input_pb = "graph-compress.pb"
output_dir = "graph-compress.savedmodel"

print(f"Reading frozen graph: {input_pb}")
with tf.compat.v1.Session() as sess:
    with gfile.FastGFile(input_pb, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name="")
    
    graph = tf.compat.v1.get_default_graph()
    all_tensor_names = [tensor.name for op in graph.get_operations() for tensor in op.outputs]
    
    def find_tensor(possible_names):
        for name in possible_names:
            if name in all_tensor_names:
                return graph.get_tensor_by_name(name)
        for name in all_tensor_names:
            if any(p.replace(":0", "") in name.lower() for p in possible_names):
                return graph.get_tensor_by_name(name)
        raise KeyError(f"Could not find any tensor matching patterns: {possible_names}")

    inputs = {
        "coord": find_tensor(["coord:0", "t_coord:0"]),
        "type": find_tensor(["type:0", "model_type:0"]),
        "box": find_tensor(["box:0", "t_box:0"])
    }
    outputs = {
        "energy": find_tensor(["energy:0", "o_energy:0"]),
        "force": find_tensor(["force:0", "o_force:0"])
    }
    
    print(f"\nMatched Tensors:")
    print(f"  Inputs:  coord -> {inputs['coord'].name}, type -> {inputs['type'].name}, box -> {inputs['box'].name}")
    print(f"  Outputs: energy -> {outputs['energy'].name}, force -> {outputs['force'].name}")

    print(f"\nExporting directly to SavedModel directory at: {output_dir}")
    
    # Clean old broken attempts if they exist to avoid directory conflicts
    import shutil
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        
    b = tf.compat.v1.saved_model.Builder(output_dir)
    
    sig_def = tf.compat.v1.saved_model.signature_def_utils.predict_signature_def(
        inputs=inputs, outputs=outputs
    )
    
    # FIXED: Using explicit V1 namespace constants to bypass TF2 attribute errors
    b.add_meta_graph_and_variables(
        sess,
        tags=[tf.compat.v1.saved_model.tag_constants.SERVING],
        signature_def_map={
            "serving_default": sig_def
        }
    )
    b.save()

print("SavedModel generation completed successfully!")
