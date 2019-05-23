import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from tensorflow.python.tools.freeze_graph import freeze_graph_with_def_protos

from inpaint_model import InpaintCAModel

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    https://stackoverflow.com/questions/45466020/how-to-export-keras-h5-to-tensorflow-pb
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph

checkpoint_dir = './model_logs/hasT_350k/'

sess_config = tf.ConfigProto()                                           
sess_config.gpu_options.allow_growth = True                              
sess = tf.Session(config=sess_config)                                    
                                                                         
model = InpaintCAModel()                                                 
input_image_ph = tf.placeholder(                                         
    tf.float32, shape = (1,None,None,3), name = 'INPUT'
)                                                                    
output = model.build_server_graph(input_image_ph)                        
output = (output + 1.) * 127.5                                           
output = tf.reverse(output, [-1])                                        
output = tf.saturate_cast(output, tf.uint8, name='OUTPUT')                              

vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)             
assign_ops = []                                                          
for var in vars_list:                                                    
    vname = var.name                                                     
    from_name = vname                                                    
    var_value = tf.contrib.framework.load_variable(                      
        checkpoint_dir, from_name
    )                                                                
    assign_ops.append(tf.assign(var, var_value))                         
sess.run(assign_ops)                                                     

frozen_graph = freeze_session(
    sess, output_names=['OUTPUT']
)

tf.io.write_graph(frozen_graph, './dset4paper/', 'cnet.pb', as_text=False)
