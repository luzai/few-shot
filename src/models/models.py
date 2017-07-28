class BaseModel(object):
    model_type = []

    def __init__(self, input_shape, classes, type, with_bn, with_dp):
        self.input_shape = input_shape
        self.with_bn = with_bn
        self.with_dp = with_dp
        self.classes = classes

    def vis(self):
        from keras.utils import vis_utils
        vis_utils.plot_model(self.model, to_file='model.png')
