class BaseModel(object):
  model_type = []
  
  def __init__(self, input_shape, classes, config, with_bn, with_dp,hiddens,last_act_layer):
    self.hiddens = hiddens
    self.input_shape = input_shape
    self.with_bn = with_bn
    self.with_dp = with_dp
    self.classes = classes
    self.model_output_path = config.model_output_path
    self.model_output_path2 = config.model_tfevents_path
    
    self.last_act_layer=last_act_layer
  
  def vis(self):
    from keras.utils import vis_utils
    # vis_utils.plot_model(self.model,show_shapes=True, to_file=self.model_output_path + '/model.png')
    try:
      vis_utils.plot_model(self.model, show_shapes=True, to_file=self.model_output_path + '/model.pdf')
    except Exception as inst:
      print inst
  
  def save(self):
    self.model.save(self.model_output_path + '/model.h5')
    self.model.save(self.model_output_path2 + '/model.h5')
