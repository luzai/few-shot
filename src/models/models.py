class BaseModel(object):
    model_type = []

    def __init__(self, input_shape, classes, config, with_bn, with_dp):
        self.input_shape = input_shape
        self.with_bn = with_bn
        self.with_dp = with_dp
        self.classes = classes
        self.model_output_path = config.model_output_path



    def vis(self):
        from keras.utils import vis_utils
        vis_utils.plot_model(self.model,show_shapes=True, to_file=self.model_output_path + '/model.png')

    def save(self):
        self.model.save(self.model_output_path + '/model.h5')
